#!/usr/bin/env python3
"""Usage: python3 -I tools/build_overview_plots.py"""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as fs
from families import ROOT, family

OUT = os.path.join(ROOT, "paper", "figures")
WORK = os.environ.get("VTR_WORK", os.path.join(os.path.dirname(ROOT), "work"))


TL_FAMILIES = ["Pruning", "Halting", "Merging and clustering", "Hybrid", "Resampling and resolution", "Other"]
TL_COLOURS = {
    "Pruning": fs.SETTING["training-free"][:2],
    "Halting": fs.METHOD["PiToMe"][:2],
    "Merging and clustering": fs.SETTING["fine-tuned or light"][:2],
    "Hybrid": fs.METHOD["DiffRate"][:2],
    "Resampling and resolution": fs.METHOD["EViT"][:2],
    "Other": fs.METHOD["ToMe"][:2],
}
YEARS = list(range(2020, 2027))
PARTIAL_YEAR = 2026


def tl_family(operation):
    first = operation.replace(",", ";").split(";")[0].strip()
    if first == "halt":
        return "Halting"
    return family(operation)


def timeline():
    counts = {f: [0] * len(YEARS) for f in TL_FAMILIES}
    skipped = 0
    for r in csv.DictReader(open(os.path.join(ROOT, "docs", "methods_table.csv"), encoding="utf-8")):
        if r["operation"].replace(",", ";").split(";")[0].strip() == "none":
            skipped += 1
            continue
        counts[tl_family(r["operation"])][YEARS.index(int(r["year"]))] += 1
    total = np.sum([counts[f] for f in TL_FAMILIES], axis=0)
    print("timeline: %d methods (%d analysis rows skipped)" % (total.sum(), skipped))
    for f in TL_FAMILIES:
        print("  %-26s %s" % (f, counts[f]))

    fig, ax = plt.subplots(figsize=(fs.TEXTWIDTH_IN, 2.2))
    x = np.arange(len(YEARS))
    bottom = np.zeros(len(YEARS))
    for f in TL_FAMILIES:
        edge, face = TL_COLOURS[f]
        v = np.array(counts[f])
        ax.bar(x, v, 0.62, bottom=bottom, color=face, edgecolor=edge, linewidth=0.7, label=f, zorder=2)
        bottom += v
    for xi, t in zip(x, total):
        ax.text(xi, t + 0.8, str(t), ha="center", va="bottom", fontsize=7.5, color=fs.INK)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) + ("\n(partial)" if y == PARTIAL_YEAR else "") for y in YEARS])
    ax.set_ylabel("methods")
    ax.set_ylim(0, max(total) * 1.18)
    ax.set_xlim(-0.6, len(YEARS) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.4, handleheight=1.0,
              columnspacing=1.6, borderaxespad=0.3)
    fig.tight_layout(pad=0.2)
    fs.save(fig, os.path.join(OUT, "timeline"))
    plt.close(fig)


N0 = 197
BLOCKS = 12
TOME_R = 13
UNREDUCED_TOP1 = 79.86


D = 384
PATCH_EMBED, HEAD = 196 * 768 * D, D * 1000


def fixed_keep(rate, at):
    n, out = N0, []
    for b in range(1, BLOCKS + 1):
        out.append(n)
        if b in at:
            n = 1 + math.ceil(rate * (n - 1))
    return out + [n]


def tome(r_of_block):
    n, out = N0, []
    for b in range(1, BLOCKS + 1):
        out.append(n)
        r = min(r_of_block(b), (n - 1) // 2)
        n -= r
    return out + [n]


def diffrate(setting="2.9"):
    d = json.load(open(os.path.join(WORK, "official", "DiffRate", "compression_rate.json")))["ViT-S-DeiT"][setting]
    return [N0] + json.loads(d["merge_kept_num"])


def cumulative_gmacs(n):
    total, out = PATCH_EMBED, []
    for i in range(BLOCKS):
        total += 4 * n[i] * D * D + 2 * n[i] ** 2 * D + 8 * n[i + 1] * D * D
        out.append(total / 1e9)
    out[-1] += HEAD / 1e9
    return out


def measured(method, setting, model="deit_small"):
    path = os.path.join(ROOT, "repro", "results_imagenet_curve_official.csv")
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        other = "deit_tiny" in r["note"] or "deit_base" in r["note"]
        if r["method"] == method and r["setting"] == setting and (model != "deit_small") == other:
            return float(r["top1"]), float(r["gmacs"])
    return None


def fixed_number(r, at):
    return tome(lambda b: r if b in at else 0)


def at_cost(target, at, unit):
    full = cumulative_gmacs([N0] * (BLOCKS + 1))[-1]
    if unit == "number":
        cands = [(r, fixed_number(r, at)) for r in range(1, 160)]
    else:
        cands = [(round(0.3 + 0.005 * k, 3), fixed_keep(round(0.3 + 0.005 * k, 3), at)) for k in range(140)]
    return min(cands, key=lambda c: abs(cumulative_gmacs(c[1])[-1] / full - target))


def grid_top1(method, unit, at, level=65.0):
    from build_schedule_grid import load, interp
    base, runs = load()
    pts = runs.get(("S", method, "n" if unit == "number" else "rho", ",".join(str(b) for b in at)))
    return interp(pts, level) if pts else None


def schedules():
    target = 0.65
    every = tuple(range(1, BLOCKS + 1))
    panels = [
        ("(a) Const and Ratio", [
            ("Const from 1 ($r$ = %d)", "number", every, "from 1"),
            ("Ratio from 1 ($\\rho$ = %.3g)", "fraction", every, "from 1")]),
        ("(b) From block 1, 4 or 7", [
            ("Const from 1 ($r$ = %d)", "number", tuple(range(1, 13)), "from 1"),
            ("Const from 4 ($r$ = %d)", "number", tuple(range(4, 13)), "from 4"),
            ("Const from 7 ($r$ = %d)", "number", tuple(range(7, 13)), "from 7")]),
        ("(c) Three sets of blocks", [
            ("Const from 4 ($r$ = %d)", "number", tuple(range(4, 13)), "from 4"),
            ("Const 4/6/8/10/12 ($r$ = %d)", "number", (4, 6, 8, 10, 12), "4/6/8/10/12"),
            ("Const 4/7/10 ($r$ = %d)", "number", (4, 7, 10), "4/7/10")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH_IN, 3.3), sharey=True)
    blocks = np.arange(1, BLOCKS + 1)
    edges = np.arange(0.5, BLOCKS + 1.5)
    for ax, (title, lines) in zip(axes, panels):
        ax.axhline(N0, color=fs.REF, lw=1.0, ls=(0, (4, 2)), zorder=1)
        ends = []
        for label, unit, at, key in lines:
            c, face, mk = fs.GEOMETRIC if (unit == "fraction" and title.startswith("(a)")) else fs.SCHEDULE[key]
            ls = fs.UNIT_STYLE["Const" if unit == "number" else "Ratio"]
            amount, n = at_cost(target, at, unit)
            cost = cumulative_gmacs(n)[-1]
            print("schedules: %-45s cost %.3f GMACs, counts %s" % (label % amount, cost, n))
            ax.step(edges, n[:BLOCKS] + [n[BLOCKS - 1]], where="post", color=c, lw=1.4, ls=ls, zorder=3)
            ax.plot(blocks, n[:BLOCKS], ls="none", marker=mk, ms=4.2, mfc=face, mec=c, mew=0.9, zorder=4)
            acc = [grid_top1(m, unit, at) for m in ("topk", "tome")]
            name = label.split(" (")[0]
            text = name + "  " + " / ".join("--" if a is None else "%.1f" % a for a in acc)
            print("schedules:   top-1 Top-K / ToMe at 65%%: %s" % text)
            ax.plot([], [], color=c, lw=1.4, ls=ls, marker=mk, ms=4.2, mfc=face, mec=c, mew=0.9, label=text)
            ends.append((n[BLOCKS - 1], c))

        ends.sort()
        ys = [e[0] for e in ends]
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + 14)
        for (v, c), y in zip(ends, ys):
            ax.text(BLOCKS + 0.65, y, str(v), ha="left", va="center", fontsize=7, color=c)
        ax.set_title(title, loc="left")
        ax.set_xlim(0.5, BLOCKS + 1.9)
        ax.set_ylim(0, 215)
        ax.set_xticks([1, 4, 7, 10, 12])
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_xlabel("block")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, handlelength=2.0, labelspacing=0.3,
                  borderaxespad=0.0, title="top-1 (%): Top-K / ToMe", title_fontsize=7, alignment="left")
    axes[0].set_ylabel("tokens entering the block")
    axes[2].text(12.4, N0 + 4, "unreduced", ha="right", va="bottom", fontsize=7, color=fs.REF)
    axes[2].text(BLOCKS + 0.65, N0, "197", ha="left", va="center", fontsize=7, color=fs.REF,
                 bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fs.save(fig, os.path.join(OUT, "schedules"))
    plt.close(fig)


if __name__ == "__main__":
    fs.apply()
    timeline()
    schedules()
