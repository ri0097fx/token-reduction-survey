#!/usr/bin/env python3
"""Usage: VTR_WORK=/path/to/work python3 repro/run_schedule_grid.py VAL_DIR [--device cuda] [--models S,T,B] [--plan]"""
import argparse
import csv
import datetime
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "repro", "official"))
from run_official import schedule_counts

MODELS = {"S": ("deit_small_patch16_224", 384), "T": ("deit_tiny_patch16_224", 192), "B": ("deit_base_patch16_224", 768)}
N0, DEPTH, PATCHES = 197, 12, 196
TARGETS = (0.80, 0.72, 0.65, 0.58, 0.50, 0.44)

SCHEDULES = {
    "from 1": range(1, 13), "from 2": range(2, 13), "from 3": range(3, 13), "from 4": range(4, 13),
    "from 5": range(5, 13), "from 7": range(7, 13),
    "4,7,10": (4, 7, 10), "4,6,8,10,12": (4, 6, 8, 10, 12), "1,4,7,10": (1, 4, 7, 10),
}
SUBSET = ("from 1", "from 4", "from 7", "4,7,10", "1,4,7,10")


def gmacs(blocks, amount, method, d):
    keep, fuse = method != "tome", method == "evit"
    plan = schedule_counts(list(blocks), amount, fuse, N0, DEPTH, keep=keep)
    n, total = N0, PATCHES * 768 * d + d * 1000
    for v in plan:
        if keep:
            n_out = n if v is None else v + (2 if fuse else 1)
        else:
            n_out = n - v
        total += 4 * n * d * d + 2 * n * n * d + 8 * n_out * d * d
        n = n_out
    return total / 1e9


def pick(blocks, unit, method, d):
    base = gmacs(blocks, ("n", 0), method, d)
    if unit == "n":
        cands = [("n", r) for r in range(1, 160)]
    else:
        cands = [("rho", round(0.30 + 0.005 * i, 3)) for i in range(140)]
    table = [(a, gmacs(blocks, a, method, d) / base) for a in cands]
    out, used = [], set()
    for t in TARGETS:
        a, f = min(table, key=lambda x: abs(x[1] - t))
        if abs(f - t) <= 0.02 and a not in used:
            used.add(a)
            out.append((a, f))
    return out


def setting_of(method, blocks, amount):
    loc = ",".join(str(b) for b in blocks)
    kind, v = amount
    if method == "tome":
        return f"{v}@{loc}" if kind == "n" else f"rho={v}@{loc}"
    return f"n={v}@{loc}" if kind == "n" else f"{v}@{loc}"


def label_of(method, setting):
    v, loc = setting.split("@")
    if method == "tome":
        return (f"rho={float(v[4:])}" if v.startswith("rho=") else f"r={int(v)}") + f" at blocks {loc}"
    tag = ", fuse_token" if method == "evit" else ", no fused token"
    return (f"n={int(v[2:])}" if v.startswith("n=") else f"keep={float(v)}") + f" at blocks {loc}{tag}"


def planned_runs(models, methods=("tome", "evit")):
    runs = []
    for m in models:
        name, d = MODELS[m]
        for sched, blocks in SCHEDULES.items():
            if m != "S" and sched not in SUBSET:
                continue
            for method in methods:
                for unit in ("n", "rho"):
                    for amount, frac in pick(blocks, unit, method, d):
                        runs.append((name, method, setting_of(method, blocks, amount), sched, unit, frac))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default="S,T,B")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", default="tome,evit")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    runs = planned_runs(args.models.split(","), args.methods.split(","))
    out = os.path.join(REPO, "repro", "results_imagenet_schedule_grid.csv")
    log = os.path.join(REPO, "repro", "logs", "imagenet_schedule_grid")
    if args.plan:
        for r in runs:
            print("%-24s %-5s %-28s %-12s %-3s %.3f" % r)
        print(len(runs), "runs")
        return
    work = os.environ["VTR_WORK"]
    py = os.path.join(work, "venv_timm04", "bin", "python")
    os.makedirs(log, exist_ok=True)
    done = set()
    if os.path.exists(out):
        for row in csv.DictReader(open(out)):
            done.add((row["note"].split(";")[0].strip() if "deit_" in row["note"] else "deit_small_patch16_224",
                      row["method"].split()[0], row["setting"]))
    prog = open(os.path.join(log, "progress.txt"), "a")
    for name, method, setting, sched, unit, frac in runs:
        tag = "%s_%s_%s" % (name, method, setting.replace("@", "_at_").replace(",", "_").replace("=", ""))
        cmd = [py, os.path.join(REPO, "repro", "official", "run_official.py"), "--model", name, "--method", method,
               "--setting", setting, "--data", args.data, "--device", args.device, "--batch", str(args.batch),
               "--workers", str(args.workers), "--out", out]
        if (name, method, label_of(method, setting)) in done:
            continue
        with open(os.path.join(log, tag + ".log"), "w") as fh:
            stamp = datetime.datetime.now().strftime("%F %T")
            prog.write("%s start %s (%s, %s, target %.2f)\n" % (stamp, tag, sched, unit, frac)); prog.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, VTR_WORK=work))
            stamp = datetime.datetime.now().strftime("%F %T")
            prog.write("%s %s %s\n" % (stamp, "done " if rc == 0 else "FAILED", tag)); prog.flush()
    prog.write("%s ALL DONE\n" % datetime.datetime.now().strftime("%F %T"))


if __name__ == "__main__":
    main()
