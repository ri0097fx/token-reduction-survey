#!/usr/bin/env python3
"""Usage: python3 repro/bench.py --check                 # wrapper without reduction == timm output"""
import argparse
import os
import sys
import time

import torch
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokred.methods import PROP_ATTN, REGISTRY
from tokred.vit import ReducedViT, macs_per_image

SETTINGS = [
    ("none", {}),
    ("topk", {"keep_rate": 0.7}),
    ("evit", {"keep_rate": 0.7}),
    ("ats", {"keep_rate": 0.7}),
    ("etps", {"keep_rate": 0.7}),
    ("ppt", {"keep_rate": 0.7}),
    ("zero_tprune", {"keep_rate": 0.7}),
    ("tome", {"r": 13}),
    ("tofu", {"r": 13}),
    ("pitome", {"r": 13}),
    ("mctf", {"r": 16}),
    ("atc", {"keep_rate": 0.7}),
    ("token_pooling", {"r": 13}),
    ("long_bat", {"keep_rate": 0.7}),
]


def build(model_name, method, kw):
    vit = timm.create_model(model_name, pretrained=True).eval()
    red = REGISTRY[method](**kw)
    return ReducedViT(vit, red, prop_attn=PROP_ATTN[method])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deit_small_patch16_224.fb_in1k")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224)
    if args.check:
        ref = timm.create_model(args.model, pretrained=True).eval()
        wrap = ReducedViT(ref, None)
        with torch.no_grad():
            d = (ref(x) - wrap(x)).abs().max().item()
        print("max |timm - wrapper| without reduction: %.2e" % d)
    if args.cost or args.latency:
        print("%-12s %-18s %8s %10s  %s" % ("method", "setting", "GMACs", "ms (b=1)", "tokens entering blocks"))
        for method, kw in SETTINGS:
            m = build(args.model, method, kw)
            with torch.no_grad():
                m(x[:1])
                ctx = m.last_ctx
                g = macs_per_image(m.vit, ctx) / 1e9
                ms = float("nan")
                if args.latency:
                    for _ in range(10):
                        m(x[:1])
                    t0 = time.perf_counter()
                    for _ in range(args.reps):
                        m(x[:1])
                    ms = (time.perf_counter() - t0) / args.reps * 1000
            print("%-12s %-18s %8.2f %10.1f  %s" % (method, ",".join("%s=%s" % kv for kv in kw.items()), g, ms,
                                                   [round(t) for t in ctx.tokens_in]))


if __name__ == "__main__":
    main()
