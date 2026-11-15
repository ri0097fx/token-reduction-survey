#!/usr/bin/env python3
"""Usage: python3 repro/evaluate.py --data DIR [--limit N] [--out results.csv] [--methods tome,evit,...]"""
import argparse
import csv
import os
import sys
import time

import torch
import timm
from timm.data import ImageNetInfo, create_transform, resolve_data_config
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import SETTINGS
from tokred.methods import PROP_ATTN, REGISTRY
from tokred.vit import ReducedViT, macs_per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="deit_small_patch16_224.fb_in1k")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (tested on the CPU only)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--methods", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--extra", action="append", default=[],
                    help="learned settings as method:path, e.g. ltmp:repro/ltmp_thresholds.json")
    ap.add_argument("--setting", action="append", default=[],
                    help="run these settings instead of bench.SETTINGS, e.g. tome:r=16 or evit:keep_rate=0.5")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    vit = timm.create_model(args.model, pretrained=True).eval().to(args.device)
    tf = create_transform(**resolve_data_config({}, model=vit))
    ds = ImageFolder(args.data, transform=tf)
    info = ImageNetInfo()
    wnid_to_idx = {info.index_to_label_name(i): i for i in range(1000)}

    target_map = torch.tensor([int(c) if c.isdigit() else wnid_to_idx[c] for c in ds.classes])
    if args.limit:
        g = torch.Generator().manual_seed(0)
        ds = torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=g)[: args.limit].tolist())
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers)

    wanted = set(args.methods.split(",")) if args.methods else None
    rows = []
    def parse_extra(e):
        name, arg = e.split(":", 1)
        return name, {"path": arg}
    def parse_setting(e):
        name, arg = e.split(":", 1)
        kw = {}
        for kv in filter(None, arg.split(",")):
            k, v = kv.split("=")
            num = lambda t: int(t) if t.isdigit() else float(t)

            kw[k] = [num(t) for t in v.split("/")] if "/" in v else num(v)
        return name, kw
    base = [parse_setting(e) for e in args.setting] if args.setting else list(SETTINGS)
    settings = base + [parse_extra(e) for e in args.extra]
    for method, kw in settings:
        if wanted and method not in wanted:
            continue

        rkw = {k: v for k, v in kw.items() if k != "prop"}
        red = REGISTRY[method](**rkw)
        model = ReducedViT(vit, red, prop_attn=bool(kw.get("prop", PROP_ATTN[method]))).eval()
        correct = total = 0
        macs = 0.0
        t0 = time.time()
        with torch.no_grad():
            for img, y in dl:


                out = model(img.to(args.device))
                correct += (out.argmax(-1).cpu() == target_map[y]).sum().item()
                total += y.numel()
                macs += macs_per_image(vit, model.last_ctx) * y.numel()
        row = {"method": method, "setting": ";".join("%s=%s" % (k, "/".join(map(str, v)) if isinstance(v, list) else v)
                                    for k, v in kw.items()),
               "top1": round(100.0 * correct / total, 2), "gmacs": round(macs / total / 1e9, 3),
               "images": total, "seconds": round(time.time() - t0, 1),
               "note": "" if args.model == "deit_small_patch16_224.fb_in1k" else args.model}
        rows.append(row)
        print(row, flush=True)
    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["method", "setting", "top1", "gmacs", "images", "seconds", "note"])
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
