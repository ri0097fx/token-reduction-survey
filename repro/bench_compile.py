#!/usr/bin/env python3
"""Usage: VTR_WORK=/path/to/work python3 repro/bench_compile.py [--device cuda] [--out repro/bench_compile.csv]"""
import argparse
import csv
import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "official"))
import run_official as ro

SETTINGS = [("none", ""), ("tome", "13"), ("evit", "0.7"), ("diffrate", "2.9")]


def timed(forward, x, passes=300, warm=20):
    with torch.no_grad():
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
    return 1000 * times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_compile.csv"))
    ap.add_argument("--methods", default="", help="comma-separated subset of none,tome,evit,diffrate (one process per"
                    " method avoids the class patches of one released code reaching another)")
    args = ap.parse_args()
    rows = []
    wanted = set(args.methods.split(",")) if args.methods else None
    for method, setting in SETTINGS:
        if wanted and method not in wanted:
            continue
        a = types.SimpleNamespace(method=method, setting=setting, model="deit_small_patch16_224",
                                  weights=os.path.join(ro.SCRATCH, "weights"), repos=os.path.join(ro.SCRATCH, "official"))
        try:
            model, forward, label, _ = ro.ADAPTERS[method](a)
            model.eval().to(args.device)
            x = torch.randn(1, 3, 224, 224, device=args.device)
            with torch.no_grad():
                ref = forward(x)
            row = {"method": method, "setting": label, "eager_ms": round(timed(forward, x), 3)}
        except Exception as e:
            row = {"method": method, "setting": setting, "eager_ms": ("failed: " + str(e).splitlines()[0])[:120]}
            print(row, flush=True)
            rows.append(row)
            continue
        for mode in ("default", "reduce-overhead"):
            try:
                torch._dynamo.reset()
                compiled = torch.compile(forward, mode=mode, dynamic=False)
                with torch.no_grad():
                    out = compiled(x)
                same = bool(torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3))
                row[mode + "_ms"] = round(timed(compiled, x), 3)
                row[mode + "_same_output"] = same
            except Exception as e:
                row[mode + "_ms"] = ""
                row[mode + "_same_output"] = ("failed: " + str(e).splitlines()[0])[:120]
        row["note"] = "DeiT-S fp32, batch 1; torch %s; %s" % (torch.__version__, torch.cuda.get_device_name(0))
        print(row, flush=True)
        rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
