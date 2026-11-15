#!/usr/bin/env python3
"""Usage: python3 -I tools/build_visual_grid.py IMAGE_DIR GROUP_DIR"""
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

ROWS = [
    ("topk", "Top-K\nprune"),
    ("papr", "PaPr\ninput selection"),
    ("tome", "ToMe\nmerge"),
    ("mctf", "MCTF\nmerge"),
    ("pitome", "PiToMe\nmerge"),
    ("diffrate", "DiffRate\nprune + merge"),
]
P = 16
FUSED = np.array([0.42, 0.33, 0.67])


def crop(path):
    im = Image.open(path).convert("RGB")
    s = 248 / min(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    left, top = (im.width - 224) // 2, (im.height - 224) // 2
    return np.asarray(im.crop((left, top, left + 224, top + 224))).astype(float) / 255


def render(img, g):
    out = img.copy()
    grid = g.reshape(14, 14)
    pix = np.kron(grid, np.ones((P, P), dtype=int))
    for k in np.unique(grid):
        m = pix == k
        if k == -1:
            out[m] = 0.25 * img[m] + 0.75
        elif k == -2:
            out[m] = 0.3 * img[m] + 0.7 * FUSED
        elif (grid == k).sum() > 1:
            out[m] = img[m].mean(0)

    edge = np.zeros(pix.shape, bool)
    edge[:-1] |= pix[:-1] != pix[1:]
    edge[:, :-1] |= pix[:, :-1] != pix[:, 1:]
    out[edge] = 0.15
    return out


def main():
    img_dir, grp_dir = sys.argv[1], sys.argv[2]
    data = {k: np.load(os.path.join(grp_dir, k + ".npz")) for k, _ in ROWS}
    names = list(data["tome"]["names"])
    imgs = [crop(os.path.join(img_dir, n)) for n in names]
    fs.apply()
    n_r, n_c = len(ROWS) + 1, len(names)
    fig, axes = plt.subplots(n_r, n_c, figsize=(fs.TEXTWIDTH_IN, fs.TEXTWIDTH_IN * n_r / n_c * 0.93))
    for c, img in enumerate(imgs):
        axes[0, c].imshow(img)
    axes[0, 0].set_ylabel("input", rotation=0, ha="right", va="center", fontsize=7.5)
    for r, (key, label) in enumerate(ROWS, start=1):
        d = data[key]
        for c, img in enumerate(imgs):
            g = d["groups"][c]
            axes[r, c].imshow(render(img, g))
            n = int(d["counts"][c])
            axes[r, c].text(220, 220, str(n), ha="right", va="bottom", fontsize=6.5, color="white",
                            bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7.5)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.13, right=1.0, top=1.0, bottom=0.0, wspace=0.03, hspace=0.03)
    fs.save(fig, os.path.join(ROOT, "paper", "figures", "visual_grid"))


if __name__ == "__main__":
    main()
