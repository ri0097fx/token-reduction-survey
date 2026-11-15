#!/usr/bin/env python3
"""Usage: VTR_WORK=/path/to/work python3 repro/run_schedule_methods.py VAL_DIR [--device cuda] [--plan]"""
import argparse
import csv
import datetime
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "repro"))
sys.path.insert(0, os.path.join(REPO, "repro", "official"))
from run_official import schedule_counts
from run_schedule_grid import N0, DEPTH, PATCHES, TARGETS

D = 384
SCHEDULES = [("Const from 1", "n", range(1, 13)), ("Ratio from 1", "rho", range(1, 13)),
             ("Const from 4", "n", range(4, 13)), ("Ratio 4/7/10", "rho", (4, 7, 10))]

METHODS = {"mctf": ("venv_timm04", "official", False), "ppt": ("venv_timm04", "official", False),
           "atc": ("venv_atc", "official", False), "reprshift": ("venv_base", "official", True),
           "topk": ("venv_base", "own", False)}


def gmacs(blocks, amount, after):
    plan = schedule_counts(list(blocks), amount, False, N0, DEPTH)
    n, total = N0, PATCHES * 768 * D + D * 1000
    for v in plan:
        n_out = n - v
        total += 4 * n * D * D + 2 * n * n * D + 8 * (n if after else n_out) * D * D
        n = n_out
    return total / 1e9


def pick(blocks, unit, after):
    base = gmacs(blocks, ("n", 0), after)
    cands = [("n", r) for r in range(1, 160)] if unit == "n" else [("rho", round(0.30 + 0.005 * i, 3)) for i in range(140)]
    table = [(a, gmacs(blocks, a, after) / base) for a in cands]
    out, used = [], set()
    for t in TARGETS:
        a, f = min(table, key=lambda x: abs(x[1] - t))
        if abs(f - t) <= 0.02 and a not in used:
            used.add(a)
            out.append((a, f))
    return out


def official_setting(amount, blocks):
    loc = ",".join(str(b) for b in blocks)
    return (f"rho={amount[1]}@{loc}" if amount[0] == "rho" else f"{amount[1]}@{loc}")


def own_setting(amount, blocks):
    locs = "/".join(str(b - 1) for b in blocks)
    return f"topk:n_remove={amount[1]},locs={locs}" if amount[0] == "n" else f"topk:keep_rate={amount[1]},locs={locs}"


def planned_runs():
    runs = []
    for method, (env, runner, after) in METHODS.items():
        for name, unit, blocks in SCHEDULES:
            for amount, frac in pick(blocks, unit, after):
                setting = own_setting(amount, blocks) if runner == "own" else official_setting(amount, blocks)
                runs.append((method, env, runner, setting, name, frac))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    runs = planned_runs()
    out = os.path.join(REPO, "repro", "results_imagenet_schedule_methods.csv")
    log = os.path.join(REPO, "repro", "logs", "imagenet_schedule_methods")
    if args.plan:
        for r in runs:
            print("%-10s %-11s %-9s %-55s %-13s %.3f" % r)
        print(len(runs), "runs")
        return
    work = os.environ["VTR_WORK"]
    os.makedirs(log, exist_ok=True)
    done = set()
    if os.path.exists(out):
        for row in csv.DictReader(open(out)):
            done.add((row["method"].split()[0], row["setting"]))
    prog = open(os.path.join(log, "progress.txt"), "a")
    for method, env, runner, setting, name, frac in runs:
        py = os.path.join(work, env, "bin", "python")
        tag = "%s_%s" % (method, setting.replace("@", "_at_").replace(",", "_").replace("=", "").replace(":", "_").replace("/", "-"))
        if runner == "official":
            cmd = [py, os.path.join(REPO, "repro", "official", "run_official.py"), "--model", "deit_small_patch16_224",
                   "--method", method, "--setting", setting, "--data", args.data, "--device", args.device,
                   "--batch", str(args.batch), "--workers", str(args.workers), "--out", out]
            cwd = REPO
        else:
            cmd = [py, "evaluate.py", "--model", "deit_small_patch16_224.fb_in1k", "--data", args.data, "--device",
                   args.device, "--batch", str(args.batch), "--workers", str(args.workers), "--out", out,
                   "--setting", setting]
            cwd = os.path.join(REPO, "repro")

        if os.path.exists(os.path.join(log, tag + ".done")):
            continue
        with open(os.path.join(log, tag + ".log"), "w") as fh:
            stamp = datetime.datetime.now().strftime("%F %T")
            prog.write("%s start %s (%s, target %.2f)\n" % (stamp, tag, name, frac)); prog.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=dict(os.environ, VTR_WORK=work))
            stamp = datetime.datetime.now().strftime("%F %T")
            prog.write("%s %s %s\n" % (stamp, "done " if rc == 0 else "FAILED", tag)); prog.flush()
            if rc == 0:
                open(os.path.join(log, tag + ".done"), "w").close()
    prog.write("%s ALL DONE\n" % datetime.datetime.now().strftime("%F %T"))


if __name__ == "__main__":
    main()
