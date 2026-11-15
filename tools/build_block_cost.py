#!/usr/bin/env python3
"""Usage: python3 -I tools/build_block_cost.py"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, N0, BLOCKS, KEEP, AT = 384, 197, 12, 0.7, (4, 7, 10)
PARTS = [
    ("QKV, proj.", S.METHOD["ToMe"][0], S.METHOD["ToMe"][1]),
    ("attn. matrix", S.METHOD["ToFu"][0], S.METHOD["ToFu"][1]),
    ("MLP", S.METHOD["MCTF"][0], S.METHOD["MCTF"][1]),
]


def counts():
    n, rows = N0, []
    for b in range(1, BLOCKS + 1):
        n_out = 1 + math.ceil(KEEP * (n - 1)) if b in AT else n
        rows.append((n, n_out))
        n = n_out
    return rows


def main():
    S.apply()
    rows = counts()
    lin = [4 * n * D * D / 1e6 for n, _ in rows]
    att = [2 * n * n * D / 1e6 for n, _ in rows]
    mlp = [8 * m * D * D / 1e6 for _, m in rows]
    full = (4 * N0 * D * D + 2 * N0 * N0 * D + 8 * N0 * D * D) / 1e6
    total = sum(lin) + sum(att) + sum(mlp)
    saved = BLOCKS * full - total
    share = [(BLOCKS * 4 * N0 * D * D / 1e6 - sum(lin)) / saved, (BLOCKS * 2 * N0 * N0 * D / 1e6 - sum(att)) / saved,
             (BLOCKS * 8 * N0 * D * D / 1e6 - sum(mlp)) / saved]
    print("blocks: %.2f -> %.2f GMACs; saving by part: QKV, proj. %.0f%%, attn. matrix %.0f%%, MLP %.0f%%"
          % (BLOCKS * full / 1e3, total / 1e3, 100 * share[0], 100 * share[1], 100 * share[2]))

    fig, ax = plt.subplots(figsize=(0.66 * S.TEXTWIDTH_IN, 2.45))
    x = list(range(1, BLOCKS + 1))
    bottom = [0.0] * BLOCKS
    for (label, edge, face), vals in zip(PARTS, (lin, att, mlp)):
        ax.bar(x, vals, 0.7, bottom=bottom, color=face, edgecolor=edge, linewidth=0.5, label=label, zorder=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.axhline(full, color=S.REF, linewidth=1.1, linestyle=(0, (4, 2)), zorder=1)
    ax.text(BLOCKS + 0.55, full + 7, "unreduced block", ha="right", va="bottom", fontsize=7.5, color=S.MUTED)
    for b in AT:
        ax.text(b, bottom[b - 1] + 7, "%d" % rows[b - 1][1], ha="center", va="bottom", fontsize=7.5, color=S.INK)
    ax.set_xlim(0.4, BLOCKS + 0.6)
    ax.set_ylim(0, 430)
    ax.set_xticks(x)
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_xlabel("block")
    ax.set_ylabel("MMACs per block")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1.4, handleheight=1.0, labelspacing=0.35,
              borderaxespad=0.0, fontsize=7.5)
    fig.tight_layout(pad=0.2)
    S.save(fig, os.path.join(ROOT, "paper", "figures", "block_cost"))


if __name__ == "__main__":
    main()
