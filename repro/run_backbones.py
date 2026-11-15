#!/usr/bin/env python3
"""Usage: VTR_WORK=/path/to/work python3 repro/run_backbones.py VAL_DIR [--models deit_s,...] [--methods tome,...]"""
import argparse
import csv
import datetime
import math
import os
import sys
import time

import torch
from torchvision.datasets import ImageFolder
from timm.data import ImageNetInfo

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "repro"))
from backbones import BACKBONES, ReducedBackbone, block_macs_per_token, macs_of_counts
from tokred.methods import (PROP_ATTN, atc, etps, long_bat, mctf, pitome, ppt, tofu, tome, topk,
                            zero_tprune_paper)

TARGETS = (0.65, 0.50)
MERGE = ("tome", "tofu", "pitome", "mctf")
SELECT = ("topk", "evit", "etps", "ppt", "long_bat", "atc")
METHODS = ("none",) + MERGE + SELECT + ("zero_tprune",)
ZTP_S = 10


def locs_for(depth):
    return tuple(int(round(depth * f)) - 1 for f in (4 / 12, 7 / 12, 10 / 12))


def ztp_locs(depth):
    return tuple(max(1, int(round(depth * f))) for f in (1 / 12, 3 / 12, 6 / 12, 9 / 12, 11 / 12))


def cost_fraction(spec, plan, after_mlp=False):
    a_lin, mlp, hhd = block_macs_per_token(spec)
    n0 = spec.num_patches + 1
    full = sum(a_lin * n0 + 2 * n0 * n0 * hhd + mlp * n0 for _ in spec.blocks)
    n, total = n0, 0.0
    for l in range(len(spec.blocks)):
        n_out = plan(n, l)
        total += a_lin * n + 2 * n * n * hhd + mlp * (n if after_mlp else n_out)
        n = n_out
    return total / full


def choose(spec):
    depth = len(spec.blocks)
    locs = locs_for(depth)
    zl = ztp_locs(depth)
    cands = [round(0.3 + 0.01 * i, 2) for i in range(66)]
    out = {}

    def sel_plan(rho):
        return lambda n, l: 1 + int(math.ceil(rho * (n - 1))) if l in locs else n

    def ztp_plan(rho):
        def p(n, l):
            if l + 1 not in zl:
                return n
            n = n - min(ZTP_S, (n - 1) // 2)
            if zl.index(l + 1) in (1, 2, 3):
                n = 1 + max(1, int(round(rho * (n - 1))))
            return n
        return p

    for t in TARGETS:
        best = min(range(1, spec.num_patches // 2),
                   key=lambda r: abs(cost_fraction(spec, lambda n, l: n - min(r, (n - 1) // 2)) - t))
        out[("r", t)] = (best, cost_fraction(spec, lambda n, l: n - min(best, (n - 1) // 2)))
        rho = min(cands, key=lambda c: abs(cost_fraction(spec, sel_plan(c)) - t))
        out[("rho", t)] = (rho, cost_fraction(spec, sel_plan(rho)))
        zr = min(cands, key=lambda c: abs(cost_fraction(spec, ztp_plan(c), True) - t))
        out[("ztp", t)] = (zr, cost_fraction(spec, ztp_plan(zr), True))
    return locs, zl, out


def make_runs(spec, methods):
    depth = len(spec.blocks)
    locs, zl, amounts = choose(spec)
    loc_s = "/".join(str(l + 1) for l in locs)
    runs = [("none", "", None, True)] if "none" in methods else []
    for m in MERGE:
        if m not in methods:
            continue
        for t in TARGETS:
            r, _ = amounts[("r", t)]
            red = {"tome": lambda: tome(r=r), "tofu": lambda: tofu(r=r, prune_layers=depth // 2),
                   "pitome": lambda: pitome(r=r), "mctf": lambda: mctf(r=r)}[m]()
            runs.append((m, "r=%d" % r, red, PROP_ATTN[m]))
    for m in SELECT:
        if m not in methods:
            continue
        for t in TARGETS:
            rho, _ = amounts[("rho", t)]
            red = {"topk": lambda: topk(keep_rate=rho, locs=locs), "evit": lambda: topk(keep_rate=rho, locs=locs, fuse=True),
                   "etps": lambda: etps(keep_rate=rho, locs=locs), "ppt": lambda: ppt(keep_rate=rho, locs=locs),
                   "long_bat": lambda: long_bat(keep_rate=rho, locs=locs), "atc": lambda: atc(keep_rate=rho, locs=locs)}[m]()
            runs.append((m, "keep_rate=%s@%s" % (rho, loc_s), red, PROP_ATTN[m]))
    if "zero_tprune" in methods:
        for t in TARGETS:
            zr, _ = amounts[("ztp", t)]
            red = zero_tprune_paper(locs=zl, retention=(1, zr, zr, zr, 1), s=ZTP_S, depth=depth)
            runs.append(("zero_tprune", "retention=%s@%s" % (zr, "/".join(str(l) for l in zl)), red, PROP_ATTN["zero_tprune"]))
    return runs, amounts


def evaluate(spec, reducer, prop_attn, loader, target_map, device):
    model = ReducedBackbone(spec, reducer, prop_attn).to(device).eval()
    correct = total = 0
    macs = 0.0
    t0 = time.time()
    with torch.no_grad():
        for img, y in loader:
            out = model(img.to(device, non_blocking=True))
            correct += (out.argmax(-1).cpu() == target_map[y]).sum().item()
            total += y.numel()
            c = model.last_ctx
            macs += macs_of_counts(spec, c.tokens_in, c.tokens_in_sq, c.tokens_mlp) * y.numel()
    return 100.0 * correct / total, macs / total / 1e9, total, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--models", default=",".join(BACKBONES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="evaluate a fixed random subset of this size (smoke test)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "repro", "results_imagenet_backbones.csv"))
    args = ap.parse_args()
    methods = args.methods.split(",")
    log_dir = os.path.join(REPO, "repro", "logs", "imagenet_backbones")
    os.makedirs(log_dir, exist_ok=True)
    prog = open(os.path.join(log_dir, "progress.txt"), "a")
    done = set()
    if os.path.exists(args.out) and not args.limit:
        for row in csv.DictReader(open(args.out)):
            done.add((row["backbone"], row["method"], row["setting"]))
    info = ImageNetInfo()
    wnid_to_idx = {info.index_to_label_name(i): i for i in range(1000)}
    for name in args.models.split(","):
        spec = BACKBONES[name]()
        runs, amounts = make_runs(spec, methods)
        if args.plan:
            print(name, spec.label, "blocks", len(spec.blocks), "patches", spec.num_patches, "dim", spec.embed_dim)
            for k, (a, f) in amounts.items():
                print("   %-4s target %.2f -> %s (%.3f)" % (k[0], k[1], a, f))
            continue
        ds = ImageFolder(args.data, transform=spec.transform)
        target_map = torch.tensor([wnid_to_idx[c] for c in ds.classes])
        if args.limit:
            g = torch.Generator().manual_seed(0)
            ds = torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=g)[: args.limit].tolist())
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
        for method, setting, reducer, prop in runs:
            if (name, method, setting) in done:
                continue
            stamp = lambda: datetime.datetime.now().strftime("%F %T")
            prog.write("%s start %s %s %s\n" % (stamp(), name, method, setting)); prog.flush()
            try:
                top1, gmacs, n, sec = evaluate(spec, reducer, prop, loader, target_map, args.device)
            except Exception as e:
                prog.write("%s FAILED %s %s %s: %r\n" % (stamp(), name, method, setting, e)); prog.flush()
                print("FAILED", name, method, setting, repr(e), flush=True)
                continue
            row = {"backbone": name, "label": spec.label, "method": method, "setting": setting, "top1": round(top1, 2),
                   "block_gmacs": round(gmacs, 3), "images": n, "seconds": round(sec, 1),
                   "note": "%s; torch %s; %s" % (spec.note, torch.__version__, torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu")}
            print(row, flush=True)
            prog.write("%s done  %s %s %s top1=%.2f gmacs=%.3f\n" % (stamp(), name, method, setting, top1, gmacs)); prog.flush()
            if not args.limit:
                new = not os.path.exists(args.out)
                with open(args.out, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
        del spec
        torch.cuda.empty_cache()
    prog.write("%s ALL DONE (%s; %s)\n" % (datetime.datetime.now().strftime("%F %T"), args.models, args.methods))


if __name__ == "__main__":
    main()
