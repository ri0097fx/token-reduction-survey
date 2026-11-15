import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILIES = ["Pruning", "Merging and clustering", "Hybrid", "Resampling and resolution", "Other"]
_OP = {"prune": "Pruning", "fuse": "Pruning", "halt": "Pruning",
       "merge": "Merging and clustering", "cluster": "Merging and clustering",
       "hybrid": "Hybrid", "reactivate": "Hybrid",
       "resample": "Resampling and resolution", "resolution": "Resampling and resolution"}


def family(operation):
    first = operation.replace(",", ";").split(";")[0].strip()
    return _OP.get(first, "Other")


def family_of_key():
    rows = csv.DictReader(open(os.path.join(ROOT, "docs", "methods_table.csv"), encoding="utf-8"))
    return {r["key"]: family(r["operation"]) for r in rows}


def grouped(items, fam, ncols, order=None):
    out = []
    for f in (order or FAMILIES):
        sel = [row for it, row in items if fam(it) == f]
        if not sel:
            continue
        if out:
            out.append("\\midrule")
        out.append("\\multicolumn{%d}{@{}l}{\\textit{%s}} \\\\*" % (ncols, f))
        out.extend(sel)
    return "\n".join(out)
