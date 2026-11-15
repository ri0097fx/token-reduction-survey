import os

import matplotlib
from matplotlib import font_manager

TEXTWIDTH_IN = 455.24 / 72.27
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

INK = "#3f4a56"
MUTED = "#6b7682"
GRID = "#d5dae0"
REF = "#5c6878"
GHOST = "#cdd2d8"


METHOD = {
    "ToMe":        ("#4a5260", "#b7babf", "s"),
    "ToFu":        ("#cc8530", "#ebceac", "^"),
    "MCTF":        ("#3a76bd", "#b0c8e5", "X"),
    "PiToMe":      ("#b4548c", "#e1bbd1", "*"),
    "DiffRate":    ("#3c965a", "#b1d5bd", "h"),
    "eTPS":        ("#2f978a", "#acd5d0", "P"),
    "EViT":        ("#6c54aa", "#c4bbdd", "v"),
    "Zero-TPrune": ("#c2463d", "#e7b5b1", "o"),

    "Top-K":       ("#8a6d3b", "#d9c9ad", "D"),
    "BAT":         ("#6a8f2f", "#cfe0b0", "<"),
    "ATC":         ("#9c3f8f", "#dcb8d6", ">"),
    "PPT":         ("#c9a227", "#efe2a8", "p"),
}


SCHEDULE = {
    "from 1":      ("#4a5260", "#b7babf", "s"),
    "from 2":      ("#cc8530", "#ebceac", "^"),
    "from 3":      ("#3a76bd", "#b0c8e5", "X"),
    "from 4":      ("#b4548c", "#e1bbd1", "*"),
    "from 5":      ("#3c965a", "#b1d5bd", "h"),
    "from 7":      ("#2f978a", "#acd5d0", "P"),
    "4/7/10":      ("#6c54aa", "#c4bbdd", "v"),
    "4/6/8/10/12": ("#c2463d", "#e7b5b1", "o"),
    "1/4/7/10":    ("#8a6d3b", "#d9c9ad", "D"),
}
UNIT_STYLE = {"Const": "-", "Ratio": (0, (3.5, 2))}
GEOMETRIC = ("#cc8530", "#ebceac", "o")

SETTING = {
    "training-free":        ("#5a84b8", "#c9daf0", "o"),
    "fine-tuned or light":  ("#d97a12", "#f5c45a", "s"),
    "trained from scratch": ("#3f9585", "#bfe3dc", "^"),
}


def apply():
    fm = font_manager.fontManager
    fm.ttflist = [f for f in fm.ttflist if not (f.name == "LINE Seed JP" and not f.fname.startswith(FONT_DIR))]
    for f in ("Regular", "Bold"):
        fm.addfont(os.path.join(FONT_DIR, "LINESeedJP-%s.ttf" % f))
    fam = "LINE Seed JP"
    matplotlib.rcParams.update({
        "font.family": fam, "font.size": 8, "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.cal": fam, "mathtext.it": fam,
        "mathtext.bf": fam + ":bold",
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlepad": 7, "axes.labelsize": 8,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": (0, (2, 2)),
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "lines.linewidth": 1.6, "lines.markersize": 5.5, "lines.markeredgewidth": 0.9,
        "legend.frameon": False, "legend.fontsize": 8, "legend.handlelength": 2.2, "legend.columnspacing": 1.6,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def mkw(name, own=False):
    c, face, m = METHOD[name]
    return dict(color=c, marker=m, markerfacecolor=face, markeredgecolor=c, lw=1.7,
                ms=8.0 if m == "*" else 6.0, label=name + ("*" if own else ""))


def save(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path + ".png", bbox_inches="tight", pad_inches=0.02)
