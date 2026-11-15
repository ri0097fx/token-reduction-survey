#!/usr/bin/env python3
"""Usage: VTR_WORK=... python repro/transfer/eval_transfer.py --datasets cub200,cars [--methods tome,...]"""
import argparse
import csv
import datetime
import os
import sys
import time

import timm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "repro"))
sys.path.insert(0, HERE)
from backbones import Spec
from run_backbones import METHODS, evaluate, make_runs
from transfer_data import NAMES, NUM_CLASSES, eval_transform, get_dataset

WORK = os.environ.get("VTR_WORK", os.path.join(os.path.dirname(REPO), "work"))
OUT = os.path.join(REPO, "repro", "results_transfer_free.csv")


def transfer_spec(dataset, tag=""):
    v = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES[dataset]).eval()
    path = os.path.join(WORK, "transfer", f"deit_s_{dataset}" + (f"_{tag}" if tag else "") + ".pth")
    v.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)

    def embed(img):
        x = v.patch_embed(img)
        x = v._pos_embed(x)
        x = v.patch_drop(x)
        return v.norm_pre(x)

    def head(x, ctx, feats):
        return v.forward_head(v.norm(x))

    return Spec("deit_s_" + dataset, "DeiT-S " + dataset, v, list(v.blocks), embed, head, eval_transform(),
                v.embed_dim, v.patch_embed.num_patches, note=f"fine-tuned on {dataset}; {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(NAMES)); ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tag", default=""); ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    methods = args.methods.split(",")
    done = set()
    if os.path.exists(args.out) and not args.limit:
        for row in csv.DictReader(open(args.out)):
            done.add((row["dataset"], row["method"], row["setting"]))
    log_dir = os.path.join(REPO, "repro", "logs", "transfer")
    os.makedirs(log_dir, exist_ok=True)
    prog = open(os.path.join(log_dir, "eval_progress.txt"), "a")
    for name in args.datasets.split(","):
        spec = transfer_spec(name, args.tag)
        runs, _ = make_runs(spec, methods)
        ds = get_dataset(name, False, spec.transform)
        target_map = torch.arange(NUM_CLASSES[name])
        if args.limit:
            g = torch.Generator().manual_seed(0)
            ds = torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=g)[: args.limit].tolist())
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
        for method, setting, reducer, prop in runs:
            if (name, method, setting) in done:
                continue
            stamp = datetime.datetime.now().strftime("%F %T")
            prog.write(f"{stamp} start {name} {method} {setting}\n"); prog.flush()
            top1, gmacs, n, sec = evaluate(spec, reducer, prop, loader, target_map, args.device)
            row = {"dataset": name, "method": method, "setting": setting, "top1": round(top1, 2),
                   "block_gmacs": round(gmacs, 3), "images": n, "seconds": round(sec, 1),
                   "note": "%s; torch %s" % (spec.note, torch.__version__)}
            print(row, flush=True)
            if not args.limit:
                new = not os.path.exists(args.out)
                with open(args.out, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
        del spec
        torch.cuda.empty_cache()
    prog.write(f"{datetime.datetime.now().strftime('%F %T')} DONE ({args.datasets})\n")


if __name__ == "__main__":
    main()
