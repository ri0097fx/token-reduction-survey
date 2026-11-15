#!/usr/bin/env python3
"""Usage: python3 -I tools/build_method_table.py"""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import family, grouped
import paper_scope
from venues import venue_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["key", "name", "year", "target", "operation", "criterion", "schedule", "training", "backbones",
        "main_result", "deits_top1", "deits_gflops", "deits_throughput", "latency_reported", "batch1",
        "code", "notes", "evidence", "source"]
ALLOWED = {
    "target": {"classification", "dense", "video", "vlm", "generation", "nlp", "other"},
    "operation": {"prune", "fuse", "merge", "cluster", "halt", "resample", "resolution", "hybrid", "reactivate", "other", "none", "n/a"},
    "criterion": {"cls-attention", "learned-predictor", "similarity", "norm", "gradient", "diversity", "query-aware",
                  "frequency", "random", "none", "other", "n/a"},
    "schedule": {"fixed", "learned-rate", "threshold", "halting", "budget", "input-adaptive", "other", "none", "n/a"},
    "training": {"training-free", "light", "fine-tune", "scratch", "mixed", "none", "n/a"},
}
R = ">{\\raggedright\\arraybackslash}p{%s}"

COLSPEC = "@{\\extracolsep{\\fill}}" + R % "2.65cm" + "llllllc@{}"


def display_names():
    bib = open(os.path.join(ROOT, "paper", "references.bib"), encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^@\w+\{([^,]+),(.*?)^\}", bib, re.M | re.S):
        a = re.search(r"author\s*=\s*\{(.*?)\},", m.group(2), re.S)
        if not a:
            continue
        names = [x.strip() for x in a.group(1).split(" and ")]
        fam = [re.sub(r"[{}\\]", "", n.split(",")[0]).strip() for n in names]
        if len(fam) == 1:
            out[m.group(1)] = fam[0]
        elif len(fam) == 2:
            out[m.group(1)] = "%s and %s" % (fam[0], fam[1])
        else:
            out[m.group(1)] = "%s et al." % fam[0]
    return out


def tex_escape(s):
    s = s.replace("\\", "\\textbackslash{}")
    for c in "&%$#_{}":
        s = s.replace(c, "\\" + c)
    s = s.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
    s = s.replace("√r", "$\\sqrt{r}$")
    return s


NAME_TEX = {"pan2021iared2": "IA-RED$^2$", "cordonnier2021dps": "DPS", "tang2022psvit": "Patch Slimming",
            "long2023beyond": "BAT",
            "rao2023dss": "DynamicViT (journal)", "haurum2022sinkhorn": "MSHViT", "zhang2023segvitv2": "SegViT v2",
            "mao2025pruneandmerge": "PM-ViT", "ishibashi2025pruningrate": "GAS + VPA", "zhou2023tokensparsemedseg": "STP + MTA", "yue2021progsampling": "PS-ViT (sampling)"}

SHORT_VALUE = {"classification": "cls", "generation": "gen",
               "resample": "resamp.", "resolution": "res.", "reactivate": "react.",
               "cls-attention": "CA", "learned-predictor": "pred.", "similarity": "sim.", "gradient": "grad.",
               "diversity": "div.", "frequency": "freq.", "query-aware": "query", "random": "rand.",
               "learned-rate": "learned", "threshold": "thresh.", "halting": "halt", "input-adaptive": "input",
               "training-free": "free"}


def cell(value):
    vals = [v.strip() for v in re.split(r"[;,]", value) if v.strip()]
    return ", ".join("--" if v == "?" else tex_escape(SHORT_VALUE.get(v, v)) for v in vals)


def main():
    rows = []
    problems = []
    for f in sorted(glob.glob(os.path.join(ROOT, "docs", "methods", "*_filled.csv"))):
        with open(f, newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            missing = [c for c in COLS if c not in (r.fieldnames or [])]
            if missing:
                problems.append("%s: missing columns %s" % (os.path.basename(f), missing))
                continue
            for row in r:
                row = {c: (row.get(c) or "").strip() for c in COLS}
                for col, allowed in ALLOWED.items():
                    vals = [v.strip() for v in re.split(r"[;,]", row[col]) if v.strip()]
                    bad = [v for v in vals if v not in allowed and v != "?"]
                    if bad:
                        problems.append("%s/%s: %s has %s" % (os.path.basename(f), row["key"], col, bad))
                rows.append(row)
    keys = [r["key"] for r in rows]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        problems.append("duplicate keys: %s" % dups)
    excl_path = os.path.join(ROOT, "docs", "excluded_methods.txt")
    excluded = {l.strip() for l in open(excl_path, encoding="utf-8") if l.strip() and not l.startswith("#")}
    rows = [r for r in rows if r["key"] not in excluded]
    rows.sort(key=lambda r: (r["year"], r["key"]))
    out = os.path.join(ROOT, "docs", "methods_table.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    os.makedirs(os.path.join(ROOT, "paper", "tables"), exist_ok=True)
    methods = [r for r in rows if r["operation"] not in ("none", "n/a")]

    names = display_names()

    def table_rows(sel):
        lines = []
        for r in sel:
            name = NAME_TEX.get(r["key"]) or tex_escape(r["name"] or names.get(r["key"], r["key"]))
            lines.append((r, " & ".join([
                "%s~\\cite{%s}" % (name, r["key"]), venue_of(r["key"]), cell(r["target"]),
                cell(r["operation"]), cell(r["criterion"]), cell(r["schedule"]),
                cell(r["training"]), "yes" if r["code"] not in ("", "none", "?") else "no"]) + " \\\\"))

        return grouped(lines, lambda r: family(r["operation"]), 8)

    head = ("Method & Venue & Target & Operation & Criterion & Schedule & Training & Code \\\\")
    with open(os.path.join(ROOT, "paper", "tables", "methods_full.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by tools/build_method_table.py, do not edit by hand\n")
        fh.write("\\begin{footnotesize}\n\\setlength{\\tabcolsep}{2.5pt}\n\\setlength{\\LTleft}{0pt}\n\\setlength{\\LTright}{0pt}\n\\begin{longtable}{%s}\n" % COLSPEC)
        fh.write("\\caption{The token reduction methods of this survey on the five axes of Section~\\ref{sec:taxonomy}.}\\label{tab:methods-full}\\\\\n")
        fh.write("\\toprule\n" + head + "\n\\midrule\n\\endfirsthead\n\\caption[]{(continued)}\\\\\n\\toprule\n" + head + "\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n")
        fh.write(table_rows(methods) + "\n\\end{longtable}\n\\end{footnotesize}\n")
    summary = os.path.join(ROOT, "paper", "tables", "methods_summary.tex")
    if os.path.exists(summary):
        os.remove(summary)
    print("rows:", len(rows), "methods in the appendix:", len(methods))
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
