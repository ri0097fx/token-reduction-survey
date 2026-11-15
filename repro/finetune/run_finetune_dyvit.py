#!/usr/bin/env python3
"""Usage: VTR_WORK=... python repro/finetune/run_finetune_dyvit.py --rate 0.7 --data IMAGENET_ROOT [--freeze] [--distill]"""
import argparse
import csv
import math
import os
import sys
import time
import types

import torch
from timm.data import create_transform
from torchvision.datasets import ImageFolder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "repro", "official"))
sys.path.insert(0, HERE)
import run_official as ro
from recipe import drop_path_active
from run_finetune import COLS, MEAN, OUT, STD, add_recipe_args, evaluate, label_map, make_recipe


def build(rate, distill, device):
    ro.add_path(os.path.join(ro.SCRATCH, "official", "DynamicViT"))
    sys.modules.setdefault("torch._six", types.SimpleNamespace(inf=math.inf))
    from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
    keep = [rate, rate ** 2, rate ** 3]
    model = VisionTransformerDiffPruning(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                         qkv_bias=True, pruning_loc=[3, 6, 9], token_ratio=keep, distill=True)
    sd = torch.load(os.path.join(ro.SCRATCH, "weights", "deit_small_patch16_224.pth"), map_location="cpu",
                    weights_only=True)
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("score_predictor.") for k in missing), (missing, unexpected)
    teacher = None
    if distill:
        teacher = VisionTransformerTeacher(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                           qkv_bias=True)
        teacher.load_state_dict(sd, strict=True)
        teacher.eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
    return model.to(device), teacher, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, required=True, help="base keep ratio, 0.7 or 0.5")
    ap.add_argument("--freeze", action="store_true", help="train the predictors only")
    ap.add_argument("--distill", action="store_true", help="the released distillation loss with the DeiT-S teacher")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5, help="peak lr of the backbone")
    ap.add_argument("--lr-predictor", type=float, default=2.5e-4, help="peak lr of the predictors")
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--ratio-weight", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--val-limit", type=int, default=0)
    add_recipe_args(ap)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    model, teacher, keep = build(args.rate, args.distill, dev)
    tag = f"keep={args.rate}, " + ("predictors only" if args.freeze else "full") + (", distill" if args.distill else "")
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
    if args.distill:
        from losses import DistillDiffPruningLoss_dynamic
        crit = DistillDiffPruningLoss_dynamic(teacher, ce, clf_weight=1.0, keep_ratio=keep, mse_token=True,
                                              ratio_weight=args.ratio_weight, distill_weight=0.5, print_mode=False)

    def loss_fn(x, out, y):
        if args.distill:
            l = crit(x, out, y)
            return l[0] if isinstance(l, (tuple, list)) else l
        pred, _, _, scores = out
        ratio = sum(((s.mean(1) - keep[i]) ** 2).mean() for i, s in enumerate(scores)) / len(scores)
        return ce(pred.float(), y) + args.ratio_weight * ratio

    def log(epoch, top1, loss, t0):
        row = dict(method="dynamicvit", setting=tag, model="deit_small_patch16_224", epoch=epoch, top1=round(top1, 2),
                   train_loss=round(loss, 4) if loss == loss else "", lr=f"{args.lr_predictor:g}/{args.lr:g}",
                   batch=args.batch, aug=args.aug, seconds=round(time.time() - t0, 1))
        print(row, flush=True)
        if args.max_steps or args.val_limit:
            return
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow(row)

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
                loss = loss_fn(x, out, y)
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
        name = f"dynamicvit_{args.rate}" + ("_frozen" if args.freeze else "") + ("_distill" if args.distill else "")
        torch.save(model.state_dict(), os.path.join(d, name + ("_weakaug" if args.aug == "weak" else "") + ".pth"))


if __name__ == "__main__":
    main()
