#!/usr/bin/env python3
"""Usage: python3 -I tools/build_throughput_plot.py"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as fs
import build_resources as br

ROOT = br.ROOT


COLOR = {"topk": fs.METHOD["Top-K"][:2], "evit": fs.METHOD["EViT"][:2], "tome": fs.METHOD["ToMe"][:2],
         "mctf": fs.METHOD["MCTF"][:2], "pitome": fs.METHOD["PiToMe"][:2], "diffrate": fs.METHOD["DiffRate"][:2],
         "atc": fs.METHOD["ATC"][:2], "ppt": fs.METHOD["PPT"][:2], "reprshift": fs.METHOD["Zero-TPrune"][:2],
         "gtp": fs.METHOD["eTPS"][:2], "papr": fs.METHOD["ToFu"][:2], "ats": fs.METHOD["BAT"][:2]}
STYLE = {"Pruning": (0, (3.5, 1.8)), "Merging and clustering": "-", "Hybrid": (0, (1, 1.3))}
MARK = {"topk": "D", "evit": "v", "reprshift": "o", "gtp": "s", "ats": "P", "papr": "^",
        "tome": "s", "mctf": "X", "pitome": "*", "atc": ">", "diffrate": "h", "ppt": "p"}

OFFSET = {"topk": (-1, (7, -5)), "evit": (-1, (7, 4)), "reprshift": (-1, (-9, -2)), "gtp": (-1, (-7, -2)),
          "ats": (0, (-6, 0)), "papr": (-1, (-7, 0)), "tome": (-1, (7, 0)), "mctf": (0, (2, -10)),
          "pitome": (-1, (7, 0)), "diffrate": (-1, (7, 0)), "ppt": (0, (0, 9)), "atc": (-1, (-7, 0))}
LEFT, RIGHT = (550, 1150), (1750, 8200)


def load_curve():
    import csv
    import statistics
    path = os.path.join(br.REPRO, "resources_gpu_throughput_curve.csv")
    if not os.path.exists(path):
        return {}
    runs = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        runs.setdefault((r["method"].replace(" (official code)", ""), r["setting"]), []).append(float(r["img_per_s"]))
    out = {}
    for r in csv.DictReader(open(os.path.join(br.REPRO, "results_imagenet_curve_official.csv"), newline="", encoding="utf-8")):
        note = r.get("note", "")
        if "deit_tiny" in note or "deit_base" in note:
            continue
        k = (r["method"].replace(" (official code)", ""), r["setting"])
        if k in runs:
            out.setdefault(k[0], []).append((float(r["gmacs"]), statistics.median(runs[k]), float(r["top1"])))
    return {m: sorted(v, reverse=True) for m, v in out.items()}


def main():
    fs.apply()
    thr = br.repeat_median(br.read("resources_gpu_throughput.csv"))
    acc = br.read("results_imagenet_official.csv")
    base = acc[("none", "no reduction")]
    base_t, base_a = float(thr[("none", "no reduction")]["img_per_s"]), float(base["top1"])
    curve = load_curve()
    fig, (axl, ax) = plt.subplots(1, 2, sharey=True, figsize=(fs.TEXTWIDTH_IN, 3.6),
                                  gridspec_kw=dict(width_ratios=[1, 6], wspace=0.04))
    for a in (axl, ax):
        a.axhline(base_a, color=fs.REF, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.axvline(base_t, color=fs.REF, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.plot([base_t], [base_a], marker="*", ms=15, color=fs.INK, mec="white", mew=0.7, ls="none", zorder=6)
    ax.text(RIGHT[1] - 60, base_a + 0.08, "unreduced DeiT-S", ha="right", va="bottom", fontsize=7.5, color=fs.MUTED)
    handles = {}
    for mid, label, key, family in br.METHODS:
        pts = sorted(((float(acc[(mid, s)]["gmacs"]), float(thr[(mid, s)]["img_per_s"]), float(acc[(mid, s)]["top1"]))
                      for (m, s) in thr if m == mid and (m, s) in acc), reverse=True)
        edge, face = COLOR[mid]
        handles.setdefault(family, plt.Line2D([], [], color=fs.INK, lw=1.3, ls=STYLE[family]))
        if mid in curve:
            pts = curve[mid]
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        for a in (axl, ax):
            a.plot(xs, ys, color=edge, lw=1.4, ls=STYLE[family], marker=MARK[mid], ms=7.0 if MARK[mid] == "*" else 5.0,
                   markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.9, zorder=3)
        i, off = OFFSET[mid]
        tx = axl if xs[i] < LEFT[1] else ax
        leader = dict(arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.6, shrinkA=0, shrinkB=3)) if False else {}
        tx.annotate(label, (xs[i], ys[i]), xytext=off, textcoords="offset points", fontsize=7, color=edge,
                    ha="left" if off[0] > 0 else ("right" if off[0] < 0 else "center"), va="center", **leader)
    axl.set_xlim(*LEFT)
    ax.set_xlim(*RIGHT)
    for a, ticks in ((axl, [600, 1000]), (ax, [2000, 3000, 4000, 5000, 6000, 7000, 8000])):
        a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        a.set_xticklabels(["{:,}".format(t) for t in ticks])
    axl.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.set_ylim(71.8, 80.4)
    fig.supxlabel("images per second on the GPU at batch size 256", fontsize=matplotlib.rcParams["axes.labelsize"], y=0.012)
    axl.set_ylabel("ImageNet-1k top-1 (%)")

    fig.legend([handles[f] for f in br.FAMILY_ORDER], br.FAMILY_ORDER, loc="lower left", bbox_to_anchor=(0.0, 0.0),
               bbox_transform=axl.transAxes, handlelength=2.4, frameon=True, framealpha=1.0, facecolor=fig.get_facecolor(),
               edgecolor="none", borderpad=0.3)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.99)

    axl.get_gridspec().update(wspace=0.0)
    Path = matplotlib.path.Path
    gap = Path([(-0.55, -0.12), (0.55, -0.12), (0.55, 0.12), (-0.55, 0.12), (-0.55, -0.12)])
    marks = Path([(-1.05, -1), (-0.05, 1), (0.05, -1), (1.05, 1)], [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])
    for mk, kw in ((gap, dict(mew=0, mfc=fig.get_facecolor(), zorder=6)), (marks, dict(mew=0.9, mec=fs.INK, mfc="none", zorder=7))):

        ax.plot([0.0], [0], transform=ax.transAxes, marker=mk, ms=10, ls="none", clip_on=False, **kw)
    fs.save(fig, os.path.join(ROOT, "paper", "figures", "acc_throughput"))
    for mid, *_ in br.METHODS:
        print(mid, [(round(float(thr[(mid, s)]["img_per_s"])), acc[(mid, s)]["top1"]) for (m, s) in thr if m == mid and (m, s) in acc])


if __name__ == "__main__":
    main()
