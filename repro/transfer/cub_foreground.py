#!/usr/bin/env python3
"""Usage: VTR_WORK=... python repro/transfer/cub_foreground.py [--limit N]"""
import argparse
import csv
import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "repro"))
sys.path.insert(0, HERE)
from backbones import ReducedBackbone
from tokred.vit import Ctx, block_forward
from run_backbones import make_runs
from eval_transfer import transfer_spec
from transfer_data import DATASETS, eval_transform, get_dataset

OUT = os.path.join(REPO, "repro", "results_transfer_cub_foreground.csv")
CUB = os.path.join(DATASETS, "CUB_200_2011")
METHODS = ["topk", "evit", "tome", "mctf"]


def boxes_by_file():
    ids = {}
    for line in open(os.path.join(CUB, "images.txt")):
        i, rel = line.split()
        ids[int(i)] = rel
    boxes = {}
    for line in open(os.path.join(CUB, "bounding_boxes.txt")):
        i, x, y, w, h = line.split()
        boxes[ids[int(i)]] = (float(x), float(y), float(w), float(h))
    return boxes


def patch_mask(path, box, grid=14, patch=16, short=256, crop=224):
    W, H = Image.open(path).size
    s = short / min(W, H)
    Wr, Hr = (short, int(short * H / W)) if W <= H else (int(short * W / H), short)
    left, top = int(round((Wr - crop) / 2.0)), int(round((Hr - crop) / 2.0))
    x0, y0, w, h = box
    m = torch.zeros(grid * grid, dtype=torch.bool)
    for gy in range(grid):
        for gx in range(grid):
            cx, cy = (left + gx * patch + patch / 2) / s, (top + gy * patch + patch / 2) / s
            m[gy * grid + gx] = (x0 <= cx < x0 + w) and (y0 <= cy < y0 + h)
    return m


class TracedBackbone(ReducedBackbone):
    def forward(self, img):
        s = self.spec
        x = s.embed(img)
        B, N, _ = x.shape
        ctx = Ctx(size=torch.ones(B, N, 1, device=x.device, dtype=x.dtype), prop_attn=self.prop_attn,
                  pos=torch.arange(N, device=x.device).expand(B, N).contiguous(),
                  source=torch.eye(N, device=x.device, dtype=x.dtype).expand(B, N, N).contiguous())
        ctx.stash["depth"] = len(s.blocks)
        for i, blk in enumerate(s.blocks):
            x = block_forward(blk, x, ctx, self.reducer)
        self.last_ctx = ctx
        return s.head(x, ctx, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    spec = transfer_spec("cub200")
    runs, _ = make_runs(spec, METHODS)
    ds = get_dataset("cub200", False, eval_transform())
    boxes = boxes_by_file()
    root = os.path.join(DATASETS, "CUB_200_2011", "images", "test")
    masks = torch.stack([patch_mask(p, boxes[os.path.relpath(p, root)]) for p, _ in ds.samples])
    if args.limit:
        g = torch.Generator().manual_seed(0)
        sel = torch.randperm(len(ds), generator=g)[: args.limit].tolist()
        ds, masks = torch.utils.data.Subset(ds, sel), masks[sel]
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    print(f"{len(ds)} test images, foreground patches per image {masks.float().sum(1).mean():.1f} of 196", flush=True)
    rows = []
    for method, setting, reducer, prop in runs:
        model = TracedBackbone(spec, reducer, prop).to(dev).eval()
        cnt = {k: 0.0 for k in ["fg", "bg", "fg_dropped", "fg_fused", "fg_alone", "fg_only", "fg_mixed", "bg_dropped",
                                "bg_fused", "bg_alone", "bg_only", "bg_mixed", "size_fg", "tok_fg"]}
        correct = n = 0
        with torch.no_grad():
            for bi, (img, y) in enumerate(loader):
                out = model(img.to(dev, non_blocking=True))
                correct += (out.argmax(-1).cpu() == y).sum().item()
                src = (model.last_ctx.source[:, :, 1:] > 0.5).cpu()
                fg = masks[n: n + y.numel()]
                n += y.numel()
                row_fg = (src & fg[:, None, :]).sum(-1)
                row_bg = (src & ~fg[:, None, :]).sum(-1)
                row_n = row_fg + row_bg
                fused = torch.zeros_like(src[:, 0, :])
                if model.last_ctx.stash.get("has_fused"):
                    fused = src[:, -1, :]
                    src = src[:, :-1, :]
                    row_fg, row_bg, row_n = row_fg[:, :-1], row_bg[:, :-1], row_n[:, :-1]
                in_tok = src.any(1)

                tok_n = (src * row_n[:, :, None]).sum(1)
                tok_mixed = (src * ((row_fg > 0) & (row_bg > 0))[:, :, None]).sum(1) > 0
                for name, m in (("fg", fg), ("bg", ~fg)):
                    cnt[name] += m.sum().item()
                    cnt[name + "_dropped"] += (m & ~in_tok & ~fused).sum().item()
                    cnt[name + "_fused"] += (m & fused).sum().item()
                    cnt[name + "_alone"] += (m & in_tok & (tok_n == 1)).sum().item()
                    cnt[name + "_mixed"] += (m & in_tok & tok_mixed).sum().item()
                    cnt[name + "_only"] += (m & in_tok & (tok_n > 1) & ~tok_mixed).sum().item()
                has_fg = row_fg > 0
                cnt["size_fg"] += (row_n * has_fg).sum().item()
                cnt["tok_fg"] += has_fg.sum().item()
        row = dict(method=method, setting=setting, top1=round(100.0 * correct / n, 2), images=n,
                   fg_patches=round(cnt["fg"] / n, 1))
        for name in ("fg", "bg"):
            for k in ("dropped", "fused", "alone", "only", "mixed"):
                row[f"{name}_{k}"] = round(100.0 * cnt[f"{name}_{k}"] / cnt[name], 1)
        row["size_fg"] = round(cnt["size_fg"] / max(cnt["tok_fg"], 1), 2)
        print(row, flush=True)
        rows.append(row)
    if not args.limit:
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
