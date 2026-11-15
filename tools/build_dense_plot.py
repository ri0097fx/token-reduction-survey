#!/usr/bin/env python3
"""Usage: python3 -I tools/build_dense_plot.py"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPRO = os.path.join(ROOT, "repro")

LOOK = {"Top-K": (fs.METHOD["Top-K"], "-"), "ToMe": (fs.METHOD["ToMe"], "-"), "ALGM": (fs.METHOD["MCTF"], "-"),
        "Expedite": (fs.METHOD["DiffRate"], "-"), "ToMe for SD, attention": (fs.METHOD["EViT"], "-"),
        "ToMe for SD, attention and MLP": (fs.METHOD["eTPS"], (0, (3.5, 1.8)))}


def rows(name):
    return list(csv.DictReader(open(os.path.join(REPRO, name), newline="", encoding="utf-8")))


def panel(ax, title, base, curves, ylim, ylabel=None):
    ax.axhline(base, color=fs.REF, lw=0.9, ls=(0, (4, 2)), zorder=1)
    for label, pts in curves:
        (edge, face, marker), ls = LOOK[label]
        pts = sorted(pts, reverse=True)
        ax.plot([100 * p[0] for p in pts], [p[1] for p in pts], color=edge, lw=1.4, ls=ls, marker=marker,
                ms=7.0 if marker == "*" else 5.0, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.9,
                zorder=3, label=label, clip_on=True)
    ax.set_title(title, loc="left")
    ax.set_xlim(30, 102)
    ax.set_ylim(*ylim)
    ax.set_xticks([40, 60, 80, 100])
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    fs.apply()
    seg = rows("results_ade20k_segmenter.csv")
    yol = rows("results_coco_yolos.csv")
    vit = rows("results_coco_vitdet.csv")
    fig, axes = plt.subplots(2, 3, figsize=(fs.TEXTWIDTH_IN, 4.3))
    names = {"algm": "ALGM", "tome": "ToMe", "topk": "Top-K", "expedite": "Expedite"}
    for ax, (key, title, ylim) in zip(axes[0], (("s", "Segmenter ViT-S", (36, 47.2)), ("b", "Segmenter ViT-B", (39.5, 50.7)),
                                                ("l", "Segmenter ViT-L", (42, 53.2)))):
        sub = [r for r in seg if r["model"] == "segmenter_vit_" + key]
        base = next(float(r["miou"]) for r in sub if r["method"] == "none")
        curves = []
        for m in ("topk", "tome", "algm") + (("expedite",) if key == "l" else ()):
            curves.append((names[m], [(1.0, base)] + [(float(r["macs_frac"]), float(r["miou"])) for r in sub if r["method"] == m]))
        panel(ax, title, base, curves, ylim, "ADE20K mIoU (%)" if key == "s" else None)
    base = next(float(r["ap"]) for r in yol if r["method"] == "none")
    panel(axes[1][0], "YOLOS-S", base, [(names[m], [(1.0, base)] + [(float(r["macs_frac"]), float(r["ap"])) for r in yol if r["method"] == m])
                                       for m in ("topk", "tome")], (25.5, 36.8), "COCO box AP")
    base = next(float(r["box_ap"]) for r in vit if r["scope"] == "none")
    panel(axes[1][1], "ViTDet ViT-B", base,
          [(lab, [(1.0, base)] + [(float(r["macs_frac"]), float(r["box_ap"])) for r in vit if r["scope"] == sc])
           for sc, lab in (("attn", "ToMe for SD, attention"), ("attn_mlp", "ToMe for SD, attention and MLP"))], (44, 52.6))
    axes[1][1].set_xlim(55, 102)
    axes[1][1].set_xticks([60, 70, 80, 90, 100])

    axes[1][2].axis("off")
    handles = {}
    for ax in list(axes[0]) + list(axes[1][:2]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = ["Top-K", "ToMe", "ALGM", "Expedite", "ToMe for SD, attention", "ToMe for SD, attention and MLP"]
    short = {"ToMe for SD, attention": "ToMe for SD (attn.)", "ToMe for SD, attention and MLP": "ToMe for SD (attn. + MLP)"}
    axes[1][2].legend([handles[o] for o in order], [short.get(o, o) for o in order], loc="center", bbox_to_anchor=(0.5, 0.5), handlelength=2.4)
    fig.supxlabel("MACs of the blocks (% of the unreduced model)", fontsize=matplotlib.rcParams["axes.labelsize"], y=0.035)
    fig.tight_layout(w_pad=0.9, h_pad=1.0, rect=(0, 0.02, 1, 1))
    fs.save(fig, os.path.join(ROOT, "paper", "figures", "dense_curve"))


if __name__ == "__main__":
    main()
