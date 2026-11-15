#!/usr/bin/env python3
"""Usage: VTR_WORK=/path/to/work python3 repro/bench_sdpa.py VAL_DIR [--device cuda] [--out repro/bench_sdpa.csv]"""
import argparse
import csv
import math
import os
import sys
import time
import types

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "official"))
import run_official as ro


def qkv_of(attn, x):
    B, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def out_of(attn, o):
    B, H, N, D = o.shape
    return attn.proj(o.transpose(1, 2).reshape(B, N, H * D))


def timm_sdpa_forward(self, x):
    q, k, v = qkv_of(self, x)
    return out_of(self, F.scaled_dot_product_attention(q, k, v))


def tome_bias_forward(self, x, size=None):
    q, k, v = qkv_of(self, x)
    mask = None if size is None else size.log()[:, None, None, :, 0].to(q.dtype)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out_of(self, o), k.mean(1)


def tome_augment_forward(self, x, size=None):
    q, k, v = qkv_of(self, x)
    B, H, N, D = q.shape
    pad = (-(D + 1)) % 8
    extra_q = torch.ones(B, H, N, 1, dtype=q.dtype, device=q.device)
    bias = torch.zeros(B, N, 1, device=q.device, dtype=q.dtype) if size is None else (math.sqrt(D) * size.log()).to(q.dtype)
    extra_k = bias[:, None, :, :].expand(B, H, N, 1)
    zq = torch.zeros(B, H, N, pad, dtype=q.dtype, device=q.device)
    q2 = torch.cat([q, extra_q, zq], dim=-1)
    k2 = torch.cat([k, extra_k, zq], dim=-1)
    v2 = torch.cat([v, zq, zq[..., :1]], dim=-1)
    backends = [SDPBackend.FLASH_ATTENTION] if q.dtype in (torch.float16, torch.bfloat16) else \
        [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    with sdpa_kernel(backends):
        o = F.scaled_dot_product_attention(q2, k2, v2, scale=self.scale)[..., :D]
    return out_of(self, o), k.mean(1)


def evit_clsrow_forward(self, x, keep_rate=None, tokens=None):
    if keep_rate is None:
        keep_rate = self.keep_rate
    B, N, C = x.shape
    q, k, v = qkv_of(self, x)
    x = out_of(self, F.scaled_dot_product_attention(q, k, v))
    left_tokens = N - 1
    if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:
        left_tokens = math.ceil(keep_rate * (N - 1))
        if tokens is not None:
            left_tokens = tokens
        if left_tokens == N - 1:
            return x, None, None, None, left_tokens

        cls_attn = ((q[:, :, 0:1] @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)[:, :, 0, 1:].mean(dim=1)
        _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
        index = idx.unsqueeze(-1).expand(-1, -1, C)
        return x, index, idx, cls_attn, left_tokens
    return x, None, None, None, left_tokens


def build(method, setting, variant, args):
    a = types.SimpleNamespace(method=method, setting=setting, model="deit_small_patch16_224",
                              weights=os.path.join(ro.SCRATCH, "weights"), repos=os.path.join(ro.SCRATCH, "official"))
    model, forward, label, _ = ro.ADAPTERS[method](a)
    fn = {"released": None, "sdpa": timm_sdpa_forward, "sdpa-bias": tome_bias_forward,
          "sdpa-augment": tome_augment_forward, "sdpa-clsrow": evit_clsrow_forward}[variant]
    if fn is not None:
        for blk in model.blocks:
            blk.attn.forward = types.MethodType(fn, blk.attn)
    model.eval().to(args.device)
    return forward, label


def timed(forward, x, passes, warm, dtype):
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        for _ in range(warm):
            forward(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(passes):
            t0 = time.perf_counter()
            forward(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], sum(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--images", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(HERE, "bench_sdpa.csv"))
    args = ap.parse_args()
    from timm.data import create_transform
    from torchvision.datasets import ImageFolder
    tf = create_transform(input_size=(3, 224, 224), interpolation="bicubic", crop_pct=0.9,
                          mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ds = ImageFolder(args.data, transform=tf)
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(ds), generator=g)[: args.images].tolist()
    imgs = torch.stack([ds[i][0] for i in sel]).to(args.device)
    cases = [("none", "", ["released", "sdpa"]), ("tome", "13", ["released", "sdpa-bias", "sdpa-augment"]),
             ("evit", "0.7", ["released", "sdpa-clsrow"])]
    rows = []
    for method, setting, variants in cases:
        for dtype_name, dtype in (("fp32", None), ("fp16", torch.float16)):
            ref = None
            for variant in variants:
                forward, label = build(method, setting, variant, args)
                row = {"method": method, "setting": label, "variant": variant, "dtype": dtype_name}
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                        out = torch.cat([forward(imgs[i:i + 64]).float() for i in range(0, len(imgs), 64)])
                    if ref is None:
                        ref = out
                    row["max_abs_diff"] = round(float((out - ref).abs().max()), 5)
                    row["pred_agreement"] = round(float((out.argmax(-1) == ref.argmax(-1)).float().mean()), 4)
                    med1, _ = timed(forward, imgs[:1], 300, 20, dtype)
                    _, tot = timed(forward, imgs[:256], 20, 5, dtype)
                    row["ms_batch1"] = round(1000 * med1, 3)
                    row["img_per_s_batch256"] = round(20 * 256 / tot, 1)
                except Exception as e:
                    row["max_abs_diff"] = ("failed: " + str(e).splitlines()[0])[:100]
                row["note"] = "DeiT-S; torch %s; %s" % (torch.__version__, torch.cuda.get_device_name(0))
                print(row, flush=True)
                rows.append(row)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
