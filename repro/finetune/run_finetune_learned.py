#!/usr/bin/env python3
"""Usage: VTR_WORK=... python repro/finetune/run_finetune_learned.py --method dtps --rate 0.7 --data ROOT [--freeze]"""
import argparse
import csv
import importlib.util
import math
import os
import sys
import time
import types
from functools import partial

import torch
import torch.nn as nn
from timm.data import create_transform
from torchvision.datasets import ImageFolder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "repro", "official"))
sys.path.insert(0, HERE)
import run_official as ro
from recipe import drop_path_active
from run_finetune import COLS, MEAN, OUT, STD, add_recipe_args, evaluate, label_map, make_recipe

MACS_OUT = os.path.join(REPO, "repro", "results_finetune_macs.csv")
D, H, DEPTH, N0 = 384, 4 * 384, 12, 197
FIXED_MACS = 196 * 768 * 384 + 384 * 1000


def block_macs(n):
    return 4 * n * D * D + 2 * n * n * D + 2 * n * D * H


def load_deit_s():
    sd = torch.load(os.path.join(ro.SCRATCH, "weights", "deit_small_patch16_224.pth"), map_location="cpu",
                    weights_only=True)
    return sd.get("model", sd)


def build_dtps(rate):
    root = os.path.join(ro.SCRATCH, "official", "TPS-CVPR2023", "torch_codebase", "models")
    pkg = types.ModuleType("tps_models")
    pkg.__path__ = [root]
    sys.modules["tps_models"] = pkg
    spec = importlib.util.spec_from_file_location("tps_models.dynamic_vit", os.path.join(root, "dynamic_vit.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tps_models.dynamic_vit"] = mod
    spec.loader.exec_module(mod)
    model = mod.DynamicVit(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6), no_embed_class=False,
                           token_merge_module=mod.TPSModule(temperature=1), prune_loc=[3, 6, 9], keep_ratio=rate)
    missing, unexpected = model.load_state_dict(load_deit_s(), strict=False)
    assert not unexpected and all(k.startswith("score_predictor.") for k in missing), (missing, unexpected)
    keep = model.keep_ratio_list
    counts = [N0] * 3 + [int(196 * keep[0]) + 1] * 3 + [int(196 * keep[1]) + 1] * 3 + [int(196 * keep[2]) + 1] * 3
    print("dTPS tokens per block:", counts, "GMACs %.3f" % ((sum(block_macs(n) for n in counts) + FIXED_MACS) / 1e9))
    return model, keep


def build_avit(hp):
    root = os.path.join(ro.SCRATCH, "official", "A-ViT")
    sys.path.insert(0, root)
    for k in [m for m in sys.modules if m == "timm" or m.startswith("timm.")]:
        del sys.modules[k]
    from timm.models.act_vision_transformer import VisionTransformer
    assert VisionTransformer.__module__.startswith("timm.") and root in sys.modules["timm"].__file__
    args = types.SimpleNamespace(act_mode=4, **hp)
    model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), args=args)
    model.load_state_dict(load_deit_s(), strict=True)
    return model


def avit_macs(counter):
    n = torch.stack([(counter >= i + 1).sum(1) for i in range(DEPTH)], 1).double()
    return (4 * n * D * D + 2 * n * n * D + 2 * n * D * H).sum(1) + FIXED_MACS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dtps", "avit"], required=True)
    ap.add_argument("--rate", type=float, default=0.7, help="dtps: base keep ratio")
    ap.add_argument("--freeze", action="store_true", help="dtps: train the predictors only")
    ap.add_argument("--gate-scale", type=float, default=10.0); ap.add_argument("--gate-center", type=float, default=75.0)
    ap.add_argument("--ponder", type=float, default=0.0005); ap.add_argument("--prior-alpha", type=float, default=0.001)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5, help="peak lr of the backbone")
    ap.add_argument("--lr-predictor", type=float, default=2.5e-4, help="dtps: peak lr of the predictors")
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--ratio-weight", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--val-limit", type=int, default=0)
    add_recipe_args(ap)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    if args.method == "dtps":
        model, keep = build_dtps(args.rate)
        tag = f"keep={args.rate}, " + ("predictors only" if args.freeze else "full")
    else:
        hp = dict(gate_scale=args.gate_scale, gate_center=args.gate_center, ponder_token_scale=args.ponder,
                  distr_prior_alpha=args.prior_alpha)
        model = build_avit(hp)
        tag = f"gate {args.gate_scale:g}/{args.gate_center:g}, ponder {args.ponder:g}, prior {args.prior_alpha:g}, lr {args.lr:g}"
    model.to(dev)
    train_tf, mixup, ce, sd = make_recipe(args, model)
    print(f"recipe {args.aug}: mixup {mixup is not None}, stochastic depth modules {len(sd)}", flush=True)

    val_tf = create_transform(input_size=(3, 224, 224), interpolation="bicubic", crop_pct=0.9, mean=MEAN, std=STD)
    train = ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
    val = ImageFolder(os.path.join(args.data, "val"), transform=val_tf)
    tmap_train, tmap_val = label_map(train), label_map(val)
    if args.val_limit:
        g = torch.Generator().manual_seed(0)
        val = torch.utils.data.Subset(val, torch.randperm(len(val), generator=g)[: args.val_limit].tolist())
    tl = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    vl = torch.utils.data.DataLoader(val, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)

    groups = {("pred", True): [], ("pred", False): [], ("bone", True): [], ("bone", False): []}
    for n, p in model.named_parameters():
        pred = n.startswith("score_predictor.")
        if args.freeze and not pred:
            p.requires_grad_(False)
            continue
        groups[(("pred" if pred else "bone"), not (p.ndim <= 1 or "token" in n or "pos_embed" in n))].append(p)
    base = {"pred": args.lr_predictor, "bone": args.lr}
    opt = torch.optim.AdamW([{"params": ps, "weight_decay": args.wd if dec else 0.0, "lr": base[k], "base_lr": base[k]}
                             for (k, dec), ps in groups.items() if ps], betas=(0.9, 0.999))
    print("trainable parameters:", sum(p.numel() for g in opt.param_groups for p in g["params"]), flush=True)
    steps_per_epoch = args.max_steps or len(tl)
    total, warm = args.epochs * steps_per_epoch, max(1, int(0.2 * steps_per_epoch))

    def scale_at(s):
        return (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    def loss_fn(out, y):
        if args.method == "dtps":
            pred, _, _, hard, _ = out
            ratio = sum(((h.mean(1) - keep[i]) ** 2).mean() for i, h in enumerate(hard)) / len(hard)
            return ce(pred.float(), y) + args.ratio_weight * ratio
        loss = ce(out.float(), y) + args.ponder * torch.mean(model.rho_token)
        if args.prior_alpha > 0:
            distr = torch.stack(model.halting_score_layer)
            distr = torch.clamp(distr / torch.sum(distr), 0.01, 0.99)
            prior = args.prior_alpha * model.kl_loss(distr.log(), model.distr_target)
            if prior.item() > 0:
                loss = loss + prior
        return loss

    def eval_macs():
        if args.method == "dtps":
            return float("nan"), float("nan")
        macs, depth, n = 0.0, 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, _ in vl:
                model(x.to(dev, non_blocking=True))
                c = model.counter_token.detach()
                macs += avit_macs(c).sum().item()
                depth += c.double().mean(1).sum().item()
                n += c.shape[0]
        return macs / n / 1e9, depth / n

    def log(epoch, top1, loss, t0):
        row = dict(method=args.method, setting=tag, model="deit_small_patch16_224", epoch=epoch, top1=round(top1, 2),
                   train_loss=round(loss, 4) if loss == loss else "",
                   lr=f"{args.lr_predictor:g}/{args.lr:g}" if args.method == "dtps" else f"{args.lr:g}",
                   batch=args.batch, aug=args.aug, seconds=round(time.time() - t0, 1))
        gmacs, depth = eval_macs()
        print(row, "" if gmacs != gmacs else f"GMACs {gmacs:.3f} mean depth {depth:.2f}", flush=True)
        if args.max_steps or args.val_limit:
            return
        for path, cols, r in [(OUT, COLS, row), (MACS_OUT, ["method", "setting", "epoch", "gmacs", "mean_depth"],
                                                   dict(method=args.method, setting=tag, epoch=epoch,
                                                        gmacs=round(gmacs, 3), mean_depth=round(depth, 3)))]:
            if path == MACS_OUT and gmacs != gmacs:
                continue
            new = not os.path.exists(path)
            with open(path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                if new:
                    w.writeheader()
                w.writerow(r)

    t0 = time.time()
    model.eval()
    log(0, evaluate(model, vl, tmap_val, dev), float("nan"), t0)
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        drop_path_active(sd, True)
        t0, tot, cnt = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tl):
            if args.max_steps and i >= args.max_steps:
                break
            for gr in opt.param_groups:
                gr["lr"] = gr["base_lr"] * scale_at(step)
            x, y = x.to(dev, non_blocking=True), tmap_train[y].to(dev, non_blocking=True)
            if mixup is not None:
                x, y = mixup(x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, cnt, step = tot + loss.item(), cnt + 1, step + 1
            if i % 500 == 0:
                print(f"epoch {epoch} step {i}/{steps_per_epoch} loss {loss.item():.4f} "
                      f"{(i + 1) * args.batch / (time.time() - t0):.0f} img/s", flush=True)
        model.eval()
        drop_path_active(sd, False)
        log(epoch, evaluate(model, vl, tmap_val, dev), tot / max(cnt, 1), t0)
    if not (args.max_steps or args.val_limit):
        d = os.path.join(ro.SCRATCH, "finetune")
        os.makedirs(d, exist_ok=True)
        name = (f"dtps_{args.rate}" + ("_frozen" if args.freeze else "")) if args.method == "dtps" else \
            f"avit_lr{args.lr:g}_ponder{args.ponder:g}"
        torch.save(model.state_dict(), os.path.join(d, name + ("_weakaug" if args.aug == "weak" else "") + ".pth"))


if __name__ == "__main__":
    main()
