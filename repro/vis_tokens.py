#!/usr/bin/env python3
"""Usage: python repro/vis_tokens.py --method tome --setting 16 --images DIR --names a.jpg,b.jpg --out OUT.npz"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from timm.data import create_transform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "official"))
import run_official as R


def groups_from_source(source):
    src = source[:, 1:]
    g = src.argmax(0).numpy().astype(int)
    g[src.sum(0).numpy() == 0] = -1
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--setting", default="")
    ap.add_argument("--model", default="deit_small_patch16_224")
    ap.add_argument("--images", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=os.path.join(R.SCRATCH, "weights"))
    ap.add_argument("--repos", default=os.path.join(R.SCRATCH, "official"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model, forward, label, _ = R.ADAPTERS[args.method](args)
    model.eval()
    tf = create_transform(input_size=(3, 224, 224), interpolation="bicubic", crop_pct=0.9,
                          mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    m = args.method
    if m == "tome":
        model._tome_info["trace_source"] = True
    elif m == "diffrate":
        model._diffrate_info["trace_source"] = True
    elif m == "pitome":
        model._info["trace_source"] = True
    elif m == "mctf":
        model.info["trace_source"] = 1
    seen = {}


    where = {"tome": ("_tome_info", "source"), "diffrate": ("_diffrate_info", "source"),
             "pitome": ("_info", "source"), "mctf": ("info", "student_source")}
    if m in where:
        attr, key = where[m]
        model.blocks[-1].register_forward_pre_hook(
            lambda mod, i: seen.__setitem__("source", getattr(model, attr)[key].detach().clone()))
    if m == "dynamicvit":
        import models.dyvit as dyvit
        orig = dyvit.batch_index_select

        def spy(x, idx):
            if x.dim() == 3 and x.shape[-1] == model.embed_dim:
                seen.setdefault("steps", []).append(idx[0, 1:].cpu() - 1)
            return orig(x, idx)
        dyvit.batch_index_select = spy
    if m == "papr":
        model.pos_drop.register_forward_hook(lambda mod, i, o: seen.__setitem__("all", o.detach()))
        model.blocks[0].register_forward_pre_hook(lambda mod, i: seen.__setitem__("kept", i[0].detach()))
    out, preds, counts = [], [], []
    with torch.no_grad():
        for name in args.names.split(","):
            x = tf(Image.open(os.path.join(args.images, name)).convert("RGB")).unsqueeze(0)
            if m in ("evit", "topk"):
                feats, _, idxs = model.forward_features(x, get_idx=True)
                logits = model.head(feats)


                cur, is_fused = [{p} for p in range(196)], [False] * 196
                for idx in idxs:
                    idx = idx[0].tolist()
                    drop = set(range(len(cur))) - set(idx)
                    fused = set().union(*(cur[i] for i in drop))
                    if m == "evit":
                        cur, is_fused = [cur[i] for i in idx] + [fused], [is_fused[i] for i in idx] + [True]
                    else:
                        cur, is_fused = [cur[i] for i in idx], [is_fused[i] for i in idx]
                counts.append(len(cur) + 1)
                g = -np.ones(196, dtype=int)
                for k, (members, f) in enumerate(zip(cur, is_fused)):
                    for p in members:
                        g[p] = -2 if f else k
            else:
                logits = forward(x)
                if m == "papr":
                    a, k = seen["all"][0, 1:], seen["kept"][0, 1:]
                    counts.append(k.shape[0] + 1)
                    g = -np.ones(196, dtype=int)
                    for j in range(k.shape[0]):
                        hit = (a == k[j]).all(1).nonzero()
                        g[hit[0, 0].item()] = j
                elif m == "dynamicvit":
                    cur = list(range(196))
                    for idx in seen.pop("steps"):
                        cur = [cur[i] for i in idx.tolist()]
                    counts.append(len(cur) + 1)
                    g = -np.ones(196, dtype=int)
                    for k, p in enumerate(cur):
                        g[p] = k
                else:
                    counts.append(seen["source"].shape[1])
                    g = groups_from_source(seen["source"][0].cpu())
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            out.append(g)
            preds.append(int(logits.argmax(-1)))
    np.savez(args.out, groups=np.stack(out), preds=np.array(preds), counts=np.array(counts), names=np.array(args.names.split(",")),
             label=label)
    print(args.method, label, counts, preds)


if __name__ == "__main__":
    main()
