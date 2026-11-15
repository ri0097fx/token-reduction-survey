#!/usr/bin/env python3
"""Usage: python3 -I tools/build_dense_grid.py IMAGE_DIR PRED_DIR"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as fs
from families import ROOT

SEG = [("none", "unreduced"), ("topk_0.7", "Top-K\n69%"), ("tome_40", "ToMe\n71%"), ("algm_0.92", "ALGM\n69%")]
DET = [("none", "unreduced"), ("topk_0.6", "Top-K\n51%"), ("tome_250", "ToMe\n49%")]
SIDE = 384


def view(im, w=SIDE, h=SIDE):
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR if im.mode == "RGB" else Image.NEAREST)
    left, top = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((left, top, left + w, top + h)), s, left, top


def palette():
    return np.array(json.load(open(os.path.join(ROOT, "repro", "dense", "ade20k_palette.json")))) / 255.0


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


IMAGES = ["000000234660", "000000331075", "000000440507", "000000343934", "000000211069", "000000403353"]
DW, DH = 512, 384


def main():
    img_dir, pred_dir = sys.argv[1], sys.argv[2]
    names = IMAGES
    seg = {k: np.load(os.path.join(pred_dir, "seg_%s.npz" % k)) for k, _ in SEG}
    det = {k: json.load(open(os.path.join(pred_dir, "det_%s.json" % k))) for k, _ in DET}
    pal = palette()
    fs.apply()
    W = fs.TEXTWIDTH_IN
    left, gap, mid = 0.13, 0.004, 0.02
    cw_s = cw_d = (1 - left - gap * (len(names) - 1)) / len(names)
    ch_s = ch_d = cw_d * W * DH / DW
    H = (len(SEG) + 1) * ch_s + len(DET) * ch_d + (len(SEG) + len(DET) - 2) * gap * W + 2 * mid * W
    fig = plt.figure(figsize=(W, H))
    y = 1.0
    seg_span, det_span = [], []
    y -= ch_s / H
    for c, n in enumerate(names):
        ax = fig.add_axes([left + c * (cw_s + gap), y, cw_s, ch_s / H])
        ax.imshow(np.asarray(view(Image.open(os.path.join(img_dir, n + ".jpg")).convert("RGB"), DW, DH)[0]))
        ax.axis("off")
    fig.text(left - 0.012, y + ch_s / H / 2, "input", ha="right", va="center", fontsize=7.5)
    y -= mid * W / H
    for r, (k, label) in enumerate(SEG):
        y -= ch_s / H
        for c, n in enumerate(names):
            ax = fig.add_axes([left + c * (cw_s + gap), y, cw_s, ch_s / H])
            rgb, _, _, _ = view(Image.open(os.path.join(img_dir, n + ".jpg")).convert("RGB"), DW, DH)
            pm, _, _, _ = view(Image.fromarray(seg[k][n]), DW, DH)
            ax.imshow(0.5 * np.asarray(rgb).astype(float) / 255 + 0.5 * pal[np.asarray(pm)])
            ax.axis("off")
        fig.text(left - 0.012, y + ch_s / H / 2, label, ha="right", va="center", fontsize=7.5)
        seg_span.append((y, y + ch_s / H))
        y -= gap * W / H
    y -= (mid - gap) * W / H
    for r, (k, label) in enumerate(DET):
        y -= ch_d / H
        for c, n in enumerate(names):
            ax = fig.add_axes([left + c * (cw_d + gap), y, cw_d, ch_d / H])
            rgb, s, l0, t0 = view(Image.open(os.path.join(img_dir, n + ".jpg")).convert("RGB"), DW, DH)
            ax.imshow(np.asarray(rgb))
            for i, (x0, y0, x1, y1, score, name) in enumerate(det[k][n]):
                x0, x1 = x0 * s - l0, x1 * s - l0
                y0, y1 = y0 * s - t0, y1 * s - t0
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=COLORS[i % len(COLORS)], lw=1.1, clip_on=True))
                ax.text(max(x0, 3), max(y0, 3), "%s: %.2f" % (name, score), fontsize=4.4, color="black", ha="left", va="top", clip_on=True,
                        bbox=dict(facecolor="yellow", alpha=0.5, edgecolor="none", pad=0.6))
            ax.set_xlim(0, DW)
            ax.set_ylim(DH, 0)
            ax.axis("off")
        fig.text(left - 0.012, y + ch_d / H / 2, label, ha="right", va="center", fontsize=7.5)
        det_span.append((y, y + ch_d / H))
        y -= gap * W / H
    fig.text(0.012, (seg_span[0][1] + seg_span[-1][0]) / 2, "Segmenter ViT-B", rotation=90, ha="center", va="center", fontsize=8, fontweight="bold")
    fig.text(0.012, (det_span[0][1] + det_span[-1][0]) / 2, "YOLOS-S", rotation=90, ha="center", va="center", fontsize=8, fontweight="bold")
    fs.save(fig, os.path.join(ROOT, "paper", "figures", "dense_grid"))


if __name__ == "__main__":
    main()
