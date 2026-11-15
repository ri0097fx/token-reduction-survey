#!/usr/bin/env python3
"""Usage: VTR_WORK=... python repro/finetune/run_finetune.py --method tome --setting 16 --data IMAGENET_ROOT [--epochs 5]"""
import argparse
import csv
import math
import os
import sys
import time
import types

import torch
from timm.data import create_transform
from timm.loss import SoftTargetCrossEntropy
from torchvision.datasets import ImageFolder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "repro", "official"))
sys.path.insert(0, HERE)
import run_official as ro
from recipe import MEAN, STD, drop_path_active, make_mixup, set_drop_path, train_transform, weak_transform

OUT = os.path.join(REPO, "repro", "results_finetune.csv")
COLS = ["method", "setting", "model", "epoch", "top1", "train_loss", "lr", "batch", "aug", "seconds"]


def add_recipe_args(ap):
    ap.add_argument("--aug", choices=["deit", "weak"], default="deit",
                    help="deit: RandAugment, erasing, Mixup/CutMix, stochastic depth 0.1; weak: crop and flip only")
    ap.add_argument("--drop-path", type=float, default=0.1, help="stochastic depth rate under --aug deit")


def make_recipe(args, model, num_classes=1000):
    if args.aug == "weak":
        return weak_transform(), None, torch.nn.CrossEntropyLoss(label_smoothing=0.1), []
    return train_transform(), make_mixup(num_classes), SoftTargetCrossEntropy(), set_drop_path(model, args.drop_path)


def label_map(ds):
    wnids = [l.strip() for l in open(os.path.join(ro.HERE, "imagenet_wnids.txt"))]
    return torch.tensor([wnids.index(c) for c in ds.classes])


def evaluate(forward, loader, tmap, dev):
    correct = n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader:
            out = forward(x.to(dev, non_blocking=True))
            out = out[0] if isinstance(out, (tuple, list)) else out
            correct += (out.float().argmax(-1).cpu() == tmap[y]).sum().item()
            n += y.numel()
    return 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--setting", default="")
    ap.add_argument("--model", default="deit_small_patch16_224"); ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after this many steps (smoke test)")
    ap.add_argument("--val-limit", type=int, default=0, help="evaluate on this many val images (smoke test)")
    add_recipe_args(ap)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    a = types.SimpleNamespace(method=args.method, setting=args.setting, model=args.model, device=args.device,
                              weights=os.path.join(ro.SCRATCH, "weights"), repos=os.path.join(ro.SCRATCH, "official"))
    model, forward, label, _ = ro.ADAPTERS[args.method](a)
    model.eval().to(dev)
    train_tf, mixup, crit, sd = make_recipe(args, model)
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

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "token" in n or "pos_embed" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.999))
    steps_per_epoch = args.max_steps or len(tl)
    total, warm = args.epochs * steps_per_epoch, max(1, int(0.2 * steps_per_epoch))

    def lr_at(s):
        return args.lr * (s + 1) / warm if s < warm else 0.5 * args.lr * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    def log(epoch, top1, loss, t0):
        row = dict(method=args.method, setting=label, model=args.model, epoch=epoch, top1=round(top1, 2),
                   train_loss=round(loss, 4) if loss == loss else "", lr=args.lr, batch=args.batch, aug=args.aug,
                   seconds=round(time.time() - t0, 1))
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
    log(0, evaluate(forward, vl, tmap_val, dev), float("nan"), t0)
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0, tot, cnt = time.time(), 0.0, 0
        drop_path_active(sd, True)
        for i, (x, y) in enumerate(tl):
            if args.max_steps and i >= args.max_steps:
                break
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step)
            x, y = x.to(dev, non_blocking=True), tmap_train[y].to(dev, non_blocking=True)
            if mixup is not None:
                x, y = mixup(x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward(x)
                out = out[0] if isinstance(out, (tuple, list)) else out
                loss = crit(out.float(), y)
            if not loss.requires_grad:
                raise RuntimeError("the loss has no gradient: the forward of this method runs without autograd")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, cnt, step = tot + loss.item(), cnt + 1, step + 1
            if i % 500 == 0:
                print(f"epoch {epoch} step {i}/{steps_per_epoch} loss {loss.item():.4f} "
                      f"{(i + 1) * args.batch / (time.time() - t0):.0f} img/s", flush=True)
        drop_path_active(sd, False)
        log(epoch, evaluate(forward, vl, tmap_val, dev), tot / max(cnt, 1), t0)
    if not (args.max_steps or args.val_limit):
        d = os.path.join(ro.SCRATCH, "finetune")
        os.makedirs(d, exist_ok=True)
        name = f"{args.method}_{args.setting or 'default'}" + ("_weakaug" if args.aug == "weak" else "")
        torch.save(model.state_dict(), os.path.join(d, name + ".pth"))


if __name__ == "__main__":
    main()
