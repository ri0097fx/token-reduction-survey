#!/usr/bin/env python3
"""Usage: VTR_WORK=/path/to/work python3 repro/bench_sync.py [--device cuda] [--out repro/bench_sync.csv]"""
import argparse
import csv
import math
import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "official"))
import run_official as ro

KEEP, BLOCKS = 0.7, (3, 6, 9)


def attention_with_cls(attn, x):
    B, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    a = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(dim=-1)
    out = attn.proj((a @ v).transpose(1, 2).reshape(B, N, C))
    return out, a[:, :, 0, 1:].mean(1)


def select(x, cls_attn, variant):
    B, N, C = x.shape
    k = math.ceil(KEEP * (N - 1))
    if variant == "item":
        k = torch.tensor(KEEP * (N - 1), device=x.device).ceil().long().item()
    idx = cls_attn.topk(k, dim=1).indices.sort(dim=1).values + 1
    idx = torch.cat([torch.zeros(B, 1, dtype=idx.dtype, device=x.device), idx], dim=1)
    if variant == "mask":
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device).scatter_(1, idx, True)
        return x[mask].reshape(B, k + 1, C)
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))


def make_forward(block, variant):
    def forward(self, x):
        y, cls_attn = attention_with_cls(self.attn, self.norm1(x))
        x = x + y
        x = select(x, cls_attn, variant)
        return x + self.mlp(self.norm2(x))
    return types.MethodType(forward, block)


def build(variant, args):
    model = ro.timm_deit(types.SimpleNamespace(model="deit_small_patch16_224", weights=args.weights))
    if variant != "unreduced":
        for b in BLOCKS:
            model.blocks[b].forward = make_forward(model.blocks[b], variant)
    return model.eval().to(args.device)


def time_model(model, dev, batch, passes, warm=10):
    x = torch.randn(batch, 3, 224, 224, device=dev)
    with torch.no_grad():
        for _ in range(warm):
            model(x)
        torch.cuda.synchronize(dev)
        times = []
        for _ in range(passes):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], sum(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=os.path.join(ro.SCRATCH, "weights"))
    ap.add_argument("--out", default=os.path.join(HERE, "bench_sync.csv"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows = []
    ref = None
    x = torch.randn(2, 3, 224, 224, device=dev, generator=torch.Generator(dev).manual_seed(0))
    for variant in ("unreduced", "gather", "mask", "item"):
        model = build(variant, args)
        if variant != "unreduced":
            with torch.no_grad():
                out = model(x)
            ref = out if ref is None else ref
            assert torch.allclose(out, ref, atol=1e-4), variant
        med1, _ = time_model(model, dev, 1, 300)
        _, tot = time_model(model, dev, 256, 20, warm=5)
        row = {"variant": variant, "ms_batch1": round(1000 * med1, 3), "img_per_s_batch256": round(20 * 256 / tot, 1)}


        try:
            torch._dynamo.reset()
            compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
            with torch.no_grad():
                out = compiled(x[:1])
                same = bool(torch.allclose(out, model(x[:1]), atol=1e-3))
            med1c, _ = time_model(compiled, dev, 1, 300, warm=20)
            row["ms_batch1_cudagraph"] = round(1000 * med1c, 3)
            row["cudagraph_same_output"] = same
        except Exception as e:
            row["ms_batch1_cudagraph"] = ""
            row["cudagraph_same_output"] = ("failed: " + str(e).splitlines()[0])[:100]
        row["note"] = "DeiT-S fp32, top-k by class attention at blocks 4, 7, 10, keep 0.7; torch %s; %s" % (
            torch.__version__, torch.cuda.get_device_name(dev))
        print(row, flush=True)
        rows.append(row)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
