#!/usr/bin/env python3
"""Usage: python3 -I tools/build_finetune.py"""
import csv
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = ("65", "50")

ROWS = [
    ("Top-K", None, "free", {"65": (("topk", "keep=0.7"), "keep=0.7, no fused token"), "50": (("topk", "keep=0.5"), "keep=0.5, no fused token")}),
    ("EViT", "liang2022evit", "free", {"65": (("evit", "keep=0.7"), "keep=0.7, fuse_token"), "50": (("evit", "keep=0.5"), "keep=0.5, fuse_token")}),
    ("ToMe", "bolya2023tome", "free", {"65": (("tome", "r=11"), "r=11"), "50": (("tome", "r=16"), "r=16")}),
    ("DiffRate", "chen2023diffrate", "free", {"65": (("diffrate", "2.9"), "2.9"), "50": (("diffrate", "2.3"), "2.3")}),
    ("MCTF", "lee2024mctf", "free", {"65": (("mctf", "r=13"), "r=13, blocks 2-12, one-step-ahead"), "50": (("mctf", "r=19"), "r=19, blocks 2-12, one-step-ahead")}),
    ("Representation Shift", "choi2025reprshift", "free",
     {"65": (("reprshift", "drop 0.2 after blocks 1/4/7"), "drop 0.2 after blocks 1/4/7, MLP shift (L2)"),
      "50": (("reprshift", "drop 0.31 after blocks 1/4/7"), "drop 0.31 after blocks 1/4/7, MLP shift (L2)")}),
    ("DynamicViT", "rao2021dynamicvit", "learned", {"65": (None, "keep=0.7, full"), "50": (None, "keep=0.5, full")}),
    ("\\quad with distillation", None, "learned", {"65": (None, "keep=0.7, full, distill"), "50": (None, "keep=0.5, full, distill")}),
    ("\\quad backbone frozen", None, "learned", {"65": (None, "keep=0.7, predictors only"), "50": (None, "keep=0.5, predictors only")}),
    ("\\quad frozen, with distillation", None, "learned", {"65": (None, "keep=0.7, predictors only, distill"), "50": (None, "keep=0.5, predictors only, distill")}),
    ("dTPS", "wei2023tps", "learned", {"65": (None, "keep=0.7, full"), "50": (None, "keep=0.5, full")}),
    ("\\quad backbone frozen", None, "learned", {"65": (None, "keep=0.7, predictors only"), "50": (None, "keep=0.5, predictors only")}),
]
METHOD_ID = {"Top-K": "topk", "EViT": "evit", "ToMe": "tome", "DiffRate": "diffrate", "MCTF": "mctf",
             "Representation Shift": "reprshift", "DynamicViT": "dynamicvit", "dTPS": "dtps"}
LEARNED_GMACS = {"65": 2.878, "50": 2.147}


def tuned():
    out = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "repro", "results_finetune.csv"))):
        if r["aug"] != "deit":
            continue
        k = (r["method"], r["setting"])
        if k not in out or int(r["epoch"]) > out[k][0]:
            out[k] = (int(r["epoch"]), float(r["top1"]))
    return {k: v[1] for k, v in out.items()}


def mark(values, fmt="%.2f"):
    printed = [fmt % v if v is not None else None for v in values]
    distinct = sorted({float(p) for p in printed if p is not None}, reverse=True)
    out = []
    for p in printed:
        if p is None:
            out.append("--")
        elif distinct and float(p) == distinct[0]:
            out.append("\\textbf{%s}" % p)
        elif len(distinct) > 1 and float(p) == distinct[1]:
            out.append("\\underline{%s}" % p)
        else:
            out.append(p)
    return out


def main():
    sys.argv = sys.argv[:1]
    g = runpy.run_path(os.path.join(ROOT, "tools", "build_controlled.py"), run_name="__not_main__")
    res = g["load"]("imagenet")
    ft = tuned()
    base_free = res[("S", "none", "", True)][0]
    base_tuned = ft[("none", "no reduction")]
    cells = []
    current = None
    for label, key, group, lv in ROWS:
        row = []
        for level in LEVELS:
            fk, setting = lv[level]
            mid = METHOD_ID[label if not label.startswith("\\quad") else current]
            if not label.startswith("\\quad"):
                current = label
            if fk is not None:
                top1, gm = res[("S",) + fk + (True,)]
            else:
                top1, gm = None, LEARNED_GMACS[level]
            row += [gm, top1, ft[(mid, setting)]]
        cells.append(row)
    free65 = mark([c[1] for c in cells]); tuned65 = mark([c[2] for c in cells])
    free50 = mark([c[4] for c in cells]); tuned50 = mark([c[5] for c in cells])
    lines = ["\\begin{small}", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{$\\approx$65\\% of the GMACs} & \\multicolumn{3}{c}{$\\approx$50\\% of the GMACs} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
             "Method & GMACs & Free & Tuned & GMACs & Free & Tuned \\\\", "\\midrule",
             "Unreduced & 4.60 & %.2f & %.2f & 4.60 & %.2f & %.2f \\\\" % (base_free, base_tuned, base_free, base_tuned), "\\midrule"]
    last_group = "free"
    for i, (label, key, group, lv) in enumerate(ROWS):
        if group != last_group:
            lines.append("\\midrule")
            last_group = group
        name = label + ("~\\cite{%s}" % key if key else "")
        c = cells[i]
        lines.append("%s & %.2f & %s & %s & %.2f & %s & %s \\\\" % (name, c[0], free65[i], tuned65[i], c[3], free50[i], tuned50[i]))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{small}"]
    out = os.path.join(ROOT, "paper", "tables", "finetune.tex")
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote", out)
    for i, (label, _, _, _) in enumerate(ROWS):
        print("%-32s %s" % (label, cells[i]))


if __name__ == "__main__":
    main()
