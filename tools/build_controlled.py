#!/usr/bin/env python3
"""Usage: python3 -I tools/build_controlled.py [imagenet|imagenette|imagenetv2]   (default: imagenet, the dataset of the paper)"""
import re
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import family_of_key, grouped
from venues import year_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPRO = os.path.join(ROOT, "repro")
MODELS = ["T", "S", "B"]


def same(mid, s65, s50):
    return {"65": {m: (mid, s65) for m in MODELS}, "50": {m: (mid, s50) for m in MODELS}}


ROWS = [
    ("DiffRate", "chen2023diffrate", False,
     {"65": {"T": ("diffrate", "0.8"), "S": ("diffrate", "2.9"), "B": ("diffrate", "11.5")},
      "50": {"T": ("diffrate", "0.6"), "S": ("diffrate", "2.3"), "B": ("diffrate", "8.7")}}),
    ("MCTF", "lee2024mctf", False, same("mctf", "r=13", "r=19")),
    ("ToMe", "bolya2023tome", False, same("tome", "r=11", "r=16")),
    ("ToFu", "kim2024tofu", True, same("tofu", "r=11", "r=16")),
    ("PiToMe", "tran2024pitome", False, same("pitome", "ratio=0.925", "ratio=0.875")),
    ("GTP-ViT", "xu2024gtpvit", False, same("gtp", "P=11", "P=16")),
    ("PPT", "wu2023ppt", False, {"65": {m: ("ppt", "r=50") for m in MODELS}, "50": {}}),
    ("ATC", "haurum2024atc", False, same("atc", "keep=0.7", "keep=0.5")),
    ("Top-K", "topk", False, same("topk", "keep=0.7", "keep=0.5")),
    ("EViT", "liang2022evit", False, same("evit", "keep=0.7", "keep=0.5")),
    ("Representation Shift", "choi2025reprshift", False,
     same("reprshift", "drop 0.2 after blocks 1/4/7", "drop 0.31 after blocks 1/4/7")),
    ("PaPr", "mahmud2024papr", False, same("papr", "z=0.6", "z=0.45")),
    ("eTPS", "wei2023tps", True, same("etps", "keep_rate=0.7", "keep_rate=0.5")),
    ("Zero-TPrune", "wang2024zerotprune", True,
     same("zero_tprune", "retention=1/0.8/0.8/0.8/1", "retention=1/0.63/0.63/0.63/1")),
    ("BAT", "long2023beyond", True, same("long_bat", "keep_rate=0.7", "keep_rate=0.5")),
    ("Token Pooling", "marin2023tokenpooling", True,
     same("token_pooling", "ks=196/195/193/188/169/140/121/110/73/38/7/0;prop=1",
          "ks=196/194/187/163/118/85/58/47/20/12/2/0;prop=1")),
]
ROWS.sort(key=lambda r: 0 if r[1] == "topk" else year_of(r[1]))


SEARCHED = {"chen2023diffrate", "marin2023tokenpooling"}
MAIN = ["bolya2023tome", "kim2024tofu", "lee2024mctf", "tran2024pitome", "chen2023diffrate", "wei2023tps",
        "topk", "wang2024zerotprune"]


def model_of(note):
    if "deit_tiny" in note:
        return "T"
    if "deit_base" in note:
        return "B"
    return "S"


def load(dataset="imagenet"):
    res = {}
    files = [("results_%s_official.csv" % dataset, True), ("results_%s_official_tb.csv" % dataset, True),
             ("results_%s.csv" % dataset, False), ("results_%s_tb.csv" % dataset, False)]
    for name, official in files:
        path = os.path.join(REPRO, name)
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            mid = r["method"].replace(" (official code)", "")
            setting = r["setting"]
            if official:

                setting = setting.split(",")[0].strip()
                if mid == "diffrate":
                    setting = r["setting"]
                if mid == "none":
                    setting = ""
            res[(model_of(r.get("note", "")), mid, setting, official)] = (float(r["top1"]), float(r["gmacs"]))
    return res


def bold_best(lines, ncols, higher=True):
    cols = [[] for _ in range(ncols)]
    for i, (_, line) in enumerate(lines):
        cells = line.rstrip(" \\\\").split(" & ")[1:]
        for j, c in enumerate(cells):
            try:
                cols[j].append((float(c), i))
            except ValueError:
                pass
    best = {}
    for j, vals in enumerate(cols):
        ranked = sorted({v for v, _ in vals}, reverse=higher)[:2]
        for v, i in vals:
            if v in ranked:
                best[(i, j)] = "\\textbf{%s}" if v == ranked[0] else "\\underline{%s}"
    out = []
    for i, (key, line) in enumerate(lines):
        parts = line.rstrip(" \\\\").split(" & ")
        parts = [parts[0]] + [best[(i, j)] % c if (i, j) in best else c for j, c in enumerate(parts[1:])]
        out.append((key, " & ".join(parts) + " \\\\"))
    return out


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "imagenet"
    res = load(dataset)
    base = {m: res.get((m, "none", "", True)) or res.get((m, "none", "", False)) for m in MODELS}
    out = []
    out.append("\\begin{small}\n\\setlength{\\tabcolsep}{8pt}\n\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l" + "r" * 6 + "@{}}\n\\toprule\n")
    out.append(" & \\multicolumn{3}{c}{$\\approx$65\\% of the GMACs} & \\multicolumn{3}{c}{$\\approx$50\\% of the GMACs} \\\\\n")
    out.append("\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n")
    out.append("Method & DeiT-T & DeiT-S & DeiT-B & DeiT-T & DeiT-S & DeiT-B \\\\\n\\midrule\n")
    out.append("Unreduced & " + " & ".join("%.2f" % base[m][0] for m in MODELS) + " & "
               + " & ".join("%.2f" % base[m][0] for m in MODELS) + " \\\\\n\\midrule\n")
    missing = []
    fam = family_of_key()
    fam["topk"] = "Pruning"
    lines = []
    for label, key, own, spec in ROWS:
        cells = []
        for lvl in ("65", "50"):
            for m in MODELS:
                if m not in spec.get(lvl, {}):
                    cells.append("--")
                    continue
                mid, setting = spec[lvl][m]
                v = res.get((m, mid, setting, not own))
                if v is None:
                    missing.append((label, lvl, m))
                    cells.append("?")
                else:
                    cells.append("%.2f" % v[0])
        name = label + ("$^\\ast$" if own else "") + ("$^\\ddagger$" if key in SEARCHED else "") + ("~\\cite{%s}" % key if key != "topk" else "")
        lines.append((key, name + " & " + " & ".join(cells) + " \\\\"))
    lines = bold_best(lines, 6)
    out.append(grouped(lines, lambda k: fam[k], 7) + "\n")
    out.append("\\bottomrule\n\\end{tabular*}\n\\end{small}\n")
    with open(os.path.join(ROOT, "paper", "tables", "controlled.tex"), "w") as fh:
        fh.write("% generated by tools/build_controlled.py\n" + "".join(out))

    for lvl in ("65", "50"):
        ratios = []
        for label, key, own, spec in ROWS:
            for m in MODELS:
                if m in spec.get(lvl, {}):
                    mid, setting = spec[lvl][m]
                    v = res.get((m, mid, setting, not own))
                    if v:
                        ratios.append(v[1] / base[m][1])
        print("level", lvl, "GMACs ratio %.3f to %.3f" % (min(ratios), max(ratios)))
    print("missing", missing)
    main_table(res, base, fam)
    plot_curves()

    schedule_table()


def main_table(res, base, fam):
    lines = []
    for label, key, own, spec in ROWS:
        if key not in MAIN:
            continue
        cells = []
        for m in MODELS:
            mid, setting = spec["50"][m]
            v = res.get((m, mid, setting, not own))
            cells.append("%.1f" % v[0] if v else "?")
        name = label + ("$^\\ast$" if own else "") + ("$^\\ddagger$" if key in SEARCHED else "") + ("~\\cite{%s}" % key if key != "topk" else "")
        lines.append((key, name + " & " + " & ".join(cells) + " \\\\"))
    lines = bold_best(lines, 3)
    out = ["% generated by tools/build_controlled.py\n\\begin{small}\n\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n",
           "Method & DeiT-T & DeiT-S & DeiT-B \\\\\n\\midrule\n",
           "Unreduced & " + " & ".join("%.1f" % base[m][0] for m in MODELS) + " \\\\\n\\midrule\n",
           grouped(lines, lambda k: fam[k], 4) + "\n", "\\bottomrule\n\\end{tabular}\n\\end{small}\n"]
    with open(os.path.join(ROOT, "paper", "tables", "controlled_main.tex"), "w") as fh:
        fh.write("".join(out))


def load_curves():
    curves = {}
    for name, official in (("results_imagenet_curve_official.csv", True), ("results_imagenet_curve.csv", False)):
        path = os.path.join(REPRO, name)
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            mid = r["method"].replace(" (official code)", "")
            curves.setdefault(mid, []).append((float(r["gmacs"]), float(r["top1"])))
    return curves


def plot_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as S
    S.apply()
    curves = load_curves()
    if "none" not in curves:
        print("no curve results yet")
        return
    base_g, base_a = curves["none"][0]
    mid_of = {key: spec["65"]["S"][0] for _, key, _, spec in ROWS}
    name_of = {key: (label, own) for label, key, own, _ in ROWS}
    panels = [("(a) Merging in every block", ["bolya2023tome", "kim2024tofu", "lee2024mctf",
                                                      "tran2024pitome", "chen2023diffrate"]),
              ("(b) Selection by an attention score at a few blocks", ["topk", "wei2023tps",
                                                                 "wang2024zerotprune"])]
    ylo, yhi = 69.0, 80.4
    lines = {}
    for key in MAIN:
        pts = sorted(curves.get(mid_of[key], []))
        if pts:
            lines[key] = ([100 * g / base_g for g, _ in pts], [a for _, a in pts])
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTWIDTH_IN, 2.9), sharey=True)
    handles = []
    for ax, (title, keys) in zip(axes, panels):
        ax.axhline(base_a, color=S.REF, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
        ax.text(39, base_a + 0.12, "unreduced DeiT-S", ha="left", va="bottom", fontsize=7.5, color=S.MUTED)
        for key, (x, y) in lines.items():
            if key not in keys:
                ax.plot(x, y, color="#e1e4e8", linewidth=1.0, zorder=2)
        for z, key in enumerate(reversed(keys)):
            if key not in lines:
                continue
            x, y = lines[key]
            kw = S.mkw(*name_of[key])
            kw.update(lw=1.2, ms=6.0 if kw["marker"] == "*" else 4.4, markeredgewidth=0.8)
            h, = ax.plot(x, y, zorder=3 + z, **kw)
            handles.append((MAIN.index(key), h))
            if y[0] < ylo:
                k = next(i for i, v in enumerate(y) if v >= ylo)
                xb = x[k] + (x[k - 1] - x[k]) * (y[k] - ylo) / (y[k] - y[k - 1])
                ax.annotate("%s: %.1f%% at %.0f%%" % (h.get_label(), y[0], x[0]), (xb, ylo + 0.05),
                            xytext=(xb + 4, ylo + 0.45), fontsize=7.5, color=S.INK, ha="left", va="center",
                            arrowprops=dict(arrowstyle="-", color=S.MUTED, linewidth=0.6, shrinkA=1, shrinkB=2))
        ax.set_title(title, loc="left")
        ax.set_xlim(38, 101)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("GMACs (% of the unreduced model)")
    axes[0].set_ylabel("ImageNet-1k top-1 (%)")
    by_key = dict(handles)
    for ax, (_, keys), lift in zip(axes, panels, (0.0, 0.1)):
        ax.legend(handles=[by_key[MAIN.index(k)] for k in keys if MAIN.index(k) in by_key], loc="lower right",
                  bbox_to_anchor=(1.0, lift), fontsize=7.5, handlelength=1.8, labelspacing=0.3, borderaxespad=0.3)
    fig.tight_layout(w_pad=1.5)
    S.save(fig, os.path.join(ROOT, "paper", "figures", "controlled_curve"))


def plot_schedule():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as S
    S.apply()

    def rows(name):
        if isinstance(name, tuple):
            return [r for n in name for r in rows(n)]
        path = os.path.join(REPRO, name)
        return list(csv.DictReader(open(path))) if os.path.exists(path) else []


    OWN = ("results_imagenet_curve_official.csv", "results_imagenet_curve_tb_official.csv",
           "results_imagenet_official.csv")

    base = {model_of(r["note"]): (float(r["gmacs"]), float(r["top1"]))
            for r in rows("results_imagenet_official.csv") if r["method"].startswith("none")}

    series = [
        ("ToMe, every block", "ToMe", "-", True, (OWN, lambda r: r["method"].startswith("tome"))),
        ("ToMe, blocks 4/7/10", "ToMe", (0, (3.5, 2)), False, ("results_imagenet_schedule.csv",
                                                              lambda r: r["method"].startswith("tome"))),
        ("EViT, blocks 4/7/10", "EViT", "-", True, (OWN, lambda r: r["method"].startswith("evit"))),
        ("EViT, blocks 4 to 12", "EViT", (0, (1, 1.2)), False, ("results_imagenet_schedule.csv",
                                                              lambda r: "blocks 4,5," in r["setting"])),
        ("EViT, every block", "EViT", (0, (3.5, 2)), False, ("results_imagenet_schedule.csv",
                                                            lambda r: "blocks 1,2," in r["setting"])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(S.TEXTWIDTH_IN, 2.9))
    for ax, m, title in zip(axes, MODELS, ("DeiT-T", "DeiT-S", "DeiT-B")):
        if m not in base:
            continue
        g0, a0 = base[m]
        ax.axhline(a0, color=S.REF, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
        for label, meth, ls, filled, (name, keep) in series:
            pts = sorted((100 * float(r["gmacs"]) / g0, float(r["top1"])) for r in rows(name)
                         if keep(r) and model_of(r["note"]) == m)
            if not pts:
                continue
            c, face, mk = S.METHOD[meth]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=1.2, ls=ls, marker=mk, ms=4.0,
                    mfc=face if filled else "white", mec=c, markeredgewidth=0.8, label=label, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlim(38, 101)
        ax.set_ylim(a0 - 8.5, a0 + 0.6)
        ax.set_xlabel("GMACs (% of unreduced)")
    axes[0].set_ylabel("ImageNet-1k top-1 (%)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, handlelength=2.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.08, 1, 1))
    S.save(fig, os.path.join(ROOT, "paper", "figures", "schedule_ablation"))

def schedule_table():
    def rows(names):
        out = []
        for n in names:
            path = os.path.join(REPRO, n)
            if os.path.exists(path):
                out += list(csv.DictReader(open(path)))
        return out
    own = rows(("results_imagenet_curve_official.csv", "results_imagenet_curve_tb_official.csv",
                "results_imagenet_official.csv"))
    exch = rows(("results_imagenet_schedule.csv",))
    base = {model_of(r["note"]): (float(r["gmacs"]), float(r["top1"])) for r in own if r["method"].startswith("none")}
    series = [
        ("ToMe", "every block (own)", own, lambda r: r["method"].startswith("tome")),
        ("ToMe", "blocks 4, 7, 10", exch, lambda r: r["method"].startswith("tome")),
        ("EViT", "blocks 4, 7, 10 (own)", own, lambda r: r["method"].startswith("evit")),
        ("EViT", "blocks 4 to 12", exch, lambda r: "blocks 4,5," in r["setting"]),
        ("EViT", "every block", exch, lambda r: "blocks 1,2," in r["setting"]),
    ]

    def interp(pts, x):
        pts = sorted(set(pts))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

        near = min(pts, key=lambda p: abs(p[0] - x))
        return near[1] if abs(near[0] - x) <= 0.5 else None

    out = ["% generated by tools/build_controlled.py\n\\begin{small}\n\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n",
           " & & \\multicolumn{3}{c}{65\\% of the GMACs} & \\multicolumn{3}{c}{50\\% of the GMACs} \\\\\n",
           "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}\n",
           "Method & Schedule & DeiT-T & DeiT-S & DeiT-B & DeiT-T & DeiT-S & DeiT-B \\\\\n\\midrule\n",
           "Unreduced & & " + " & ".join("%.1f" % base[m][1] for m in MODELS) + " & "
           + " & ".join("%.1f" % base[m][1] for m in MODELS) + " \\\\\n\\midrule\n"]
    prev = None
    for meth, sched, src, keep in series:
        cells = []
        for lvl in (65.0, 50.0):
            for m in MODELS:
                pts = [(100 * float(r["gmacs"]) / base[m][0], float(r["top1"])) for r in src
                       if keep(r) and model_of(r["note"]) == m]
                v = interp(pts, lvl)
                cells.append("--" if v is None else "%.1f" % v)
        if prev is not None and prev != meth:
            out.append("\\midrule\n")
        out.append((meth if meth != prev else "") + " & " + sched + " & " + " & ".join(cells) + " \\\\\n")
        prev = meth
    out.append("\\bottomrule\n\\end{tabular}\n\\end{small}\n")
    with open(os.path.join(ROOT, "paper", "tables", "schedule.tex"), "w") as fh:
        fh.write("".join(out))


if __name__ == "__main__":
    main()
