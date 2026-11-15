#!/usr/bin/env python3
"""Usage: VTR_WORK=... python repro/transfer/finetune_transfer.py --dataset cub200 [--epochs 100] [--lr 5e-5]"""
import argparse
import csv
import math
import os
import sys
import time

import timm
import torch
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from transfer_data import NUM_CLASSES, eval_transform, get_dataset, train_transform

WORK = os.environ.get("VTR_WORK", os.path.join(os.path.dirname(REPO), "work"))
OUT = os.path.join(REPO, "repro", "results_transfer_finetune.csv")
COLS = ["dataset", "method", "setting", "model", "epoch", "top1", "train_loss", "lr", "batch", "seconds"]


def build_deit_s(num_classes, drop_path):
    model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=num_classes,
                              drop_path_rate=drop_path)
    sd = torch.load(os.path.join(WORK, "weights", "deit_small_patch16_224.pth"), map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.get("model", sd).items() if not k.startswith("head.")}
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys and set(res.missing_keys) == {"head.weight", "head.bias"}, res
    return model


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = n = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader:
            out = model(x.to(dev, non_blocking=True))
            out = out[0] if isinstance(out, (tuple, list)) else out
            correct += (out.float().argmax(-1).cpu() == y).sum().item()
            n += y.numel()
    return 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(NUM_CLASSES))
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-epochs", type=int, default=5); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1); ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.8); ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--aa", default="rand-m9-mstd0.5-inc1"); ap.add_argument("--reprob", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="", help="suffix of the weight file and the setting column")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: steps per epoch")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    K = NUM_CLASSES[args.dataset]
    model = build_deit_s(K, args.drop_path).to(dev)
    setting = f"lr={args.lr:g}, {args.epochs} epochs" + (f", {args.tag}" if args.tag else "")

    train = get_dataset(args.dataset, True, train_transform(args.aa, args.reprob))
    test = get_dataset(args.dataset, False, eval_transform())
    tl = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=True)
    vl = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"{args.dataset}: {len(train)} train, {len(test)} test, {K} classes, {len(tl)} steps per epoch", flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or "token" in n or "pos_embed" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.999))
    steps = args.max_steps or len(tl)
    total, warm = args.epochs * steps, args.warmup_epochs * steps

    def lr_at(s):
        if s < warm:
            return args.min_lr + (args.lr - args.min_lr) * (s + 1) / warm
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    mixup = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=1.0, switch_prob=0.5, mode="batch",
                  label_smoothing=args.smoothing, num_classes=K) if args.mixup > 0 or args.cutmix > 0 else None
    crit = SoftTargetCrossEntropy() if mixup else torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    def log(epoch, top1, loss, t0):
        row = dict(dataset=args.dataset, method="none", setting=setting, model="deit_small_patch16_224", epoch=epoch,
                   top1=round(top1, 2), train_loss=round(loss, 4) if loss == loss else "", lr=f"{args.lr:g}",
                   batch=args.batch, seconds=round(time.time() - t0, 1))
        print(row, flush=True)
        if args.max_steps:
            return
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow(row)

    t0 = time.time()
    log(0, evaluate(model, vl, dev), float("nan"), t0)
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tl):
            if args.max_steps and i >= args.max_steps:
                break
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if mixup:
                x, y = mixup(x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                out = out[0] if isinstance(out, (tuple, list)) else out
                loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, cnt, step = tot + loss.item(), cnt + 1, step + 1
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            log(epoch, evaluate(model, vl, dev), tot / max(cnt, 1), t0)
    if not args.max_steps:
        d = os.path.join(WORK, "transfer")
        os.makedirs(d, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(d, f"deit_s_{args.dataset}" + (f"_{args.tag}" if args.tag else "") + ".pth"))


if __name__ == "__main__":
    main()
