#!/usr/bin/env python3
"""Usage: python3 -I tools/build_results.py"""
import csv
import os
from decimal import Decimal, ROUND_HALF_UP
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import family, grouped
import paper_scope
from venues import venue_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ANNOT = {
    "rao2021dynamicvit": ("fine-tune 30 ep (distill.)", "RTX 3090, b32", 1337.7),
    "rao2023dss": ("fine-tune 30 ep", "RTX 3090, b128", None),
    "pan2021iared2": ("fine-tune 3x30 ep, RL (REINFORCE)", "V100", 930),
    "wang2021dvt": ("scratch", "", None),
    "liang2022evit": ("scratch 300 ep", "A100, b128", 2923),
    "xu2022evovit": ("scratch 300 ep", "V100, b256", 940),
    "yin2022avit": ("fine-tune 100 ep, own DeiT-S baseline 78.9%", "RTX 2080, b64", 800),
    "tang2022psvit": ("fine-tune per block and the whole model", "V100", 940),
    "fayyaz2022ats": ("fine-tune 30 ep", "Quadro RTX 6000", 1010),
    "kong2022spvit": ("fine-tune 60 ep (distill.)", "", None),
    "zong2022sit": ("fine-tune 125 ep (distill.)", "V100", 1637),
    "li2022sait": ("scratch 300 ep (distill.)", "RTX 8000, b64", None),
    "bolya2023tome": ("scratch 300 ep", "V100, fp32", 930),
    "chen2023cfvit": ("scratch", "A100, b1024", 2601),
    "wei2023tps": ("fine-tune 100 ep (distill.)", "", None),
    "long2023beyond": ("scratch 300 ep", "V100, b256", 943),
    "chang2023stvit": ("scratch 300 ep", "V100, b128", 1408),
    "lin2023supervit": ("scratch", "A100, b512", 2461),
    "dong2023heatvit": ("fine-tune 270 ep (progressive), 8-bit", "ZCU102 FPGA", None),
    "bonnaerens2023ltmp": ("fine-tune 1 ep, only the thresholds", "", None),
    "chen2023diffrate": ("training-free, with rates learned in 3 ep", "A100, fp16, b1024", 5039),
    "wu2023ppt": ("fine-tune 30 ep", "V100, b256", 993),
    "havtorn2023msvit": ("fine-tune 20 ep (gate)", "", None),
    "marin2023tokenpooling": ("fine-tune, own DeiT-S baseline 81.2%", "", None),
    "xu2024gtpvit": ("training-free", "A6000, b128", 1265.3),
    "wang2024zerotprune": ("training-free", "A100, b512", 1505.9),
    "mahmud2024papr": ("trained, $z=0.55$", "RTX A5000", 960),
    "lee2024mctf": ("fine-tune 30 ep", "", None),
    "tran2024pitome": ("training-free", "", None),
    "lee2024dtem": ("fine-tune 30 ep (end-to-end)", "RTX 3090,\\newline fp32, b128", 1390),
    "liu2024revisiting": ("fine-tune 30 ep", "A100, b512", 1524),
    "peng2025votemix": ("training-free", "", None),
    "lee2025atm": ("training-free", "", None),
    "zeng2025tokentransforming": ("training-free", "V100", 974),
    "lee2025freqaware": ("fine-tune 30 ep (distill.)", "RTX 4090, fp32", 2803),
    "pan2021hvt": ("scratch 300 ep (HVT-S-1)", "", None),
    "wang2022vtclfc": ("fine-tune 150 ep (distill.), with channel pruning", "V100, b256", 987.9),
    "yu2022miaformer": ("fine-tune 200+70 ep, RL (A2C), also heads and blocks, own baseline 79.9%", "", None),
    "liu2023asvit": ("fine-tune 30 ep (distill.)", "RTX 2080 Ti, b64", 770),
    "xu2023idlevit": ("fine-tune 30 ep (distill.)", "V100, b128", 2476.8),
    "zhang2023mgvit": ("scratch 300 ep", "RTX 3090", 1341),
    "hu2024lfvit": ("scratch 350 ep", "A100, b1024", 2601),
    "liu2024metr": ("fine-tune 30 ep (on EViT)", "RTX 4090, b512", 2360),
    "zhang2024star": ("fine-tune 120 ep (distill.)", "A100, fp32, b128", 1995.6),
    "lu2025rl4evit": ("RL policy only (MAPPO), own baseline 79.9%", "A6000", 1265.3),
    "mao2025pruneandmerge": ("fine-tune 60 ep (self-distill.)", "V100, b256", 1153),
    "huo2026mame": ("training-free", "A100, fp16, b1024", 5039),
    "li2026prance": ("500 + 30 ep (distill.), RL (PPO), channel pruning, own baseline 79.9%", "", None),
    "schlesinger2026spot": ("fine-tune 80 ep (distill.)", "", None),
    "xie2026sadtm": ("training-free", "", None),
    "zhang2026tofe": ("fine-tune 30 ep (self-distill.), own baseline 79.9%", "RTX 4090, b128", 2383.9),
    "choi2025reprshift": ("training-free (flash attn.)", "RTX A6000, b512", 3002),
    "wang2026cait": ("fine-tune 300 ep (distill.), with channel pruning", "RTX 3090, b256", None),
    "kong2023trilevel": ("scratch (sparse training, keep 0.9)", "", None),
    "yoa2025imagepiece": ("scratch 300 ep, conv stem", "RTX 3090", 2531.1),
}

EXCLUDE_FIG = {"marin2023tokenpooling", "xu2022evovit", "li2026prance", "ronen2023mixedres", "chen2021svite",
               "choi2026rapid", "haurum2024atc", "dong2023heatvit", "wang2022vtclfc", "wang2026cait"}
BASELINE = (79.8, 4.6)
RL_KEYS = {"pan2021iared2"}
INPUT_KEYS = {"mahmud2024papr"}
GRAPH_KEYS = {"wang2024zerotprune", "xu2024gtpvit"}
SUMMARY_KEYS = {"chang2023stvit", "zong2022sit"}
LOWER_BOUND = {"pan2021iared2"}
SHORT = {"rao2023dss": "DynamicViT-J", "lee2025freqaware": "Freq.-aware", "mao2025pruneandmerge": "PM-ViT", "long2023beyond": "BAT", "tang2022psvit": "Patch Slimming",
         "zeng2025tokentransforming": "Token Transforming", "pan2021iared2": "IA-RED$^2$"}


def group(setting):
    if setting.startswith("training-free"):
        return "training-free"
    if setting.startswith("scratch"):
        return "trained from scratch"
    return "fine-tuned or light"


def place_labels(ax, pts, avoid=(), cluster=None, fontsize=6.5, manual=None, split_x=None):
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tr, inv = ax.transData, ax.transData.inverted()
    bb = ax.get_window_extent()
    P = [tuple(tr.transform((x, y))) for x, y, _ in pts]
    pt = fig.dpi / 72.0
    pad, r = 1.2 * pt, 5.0 * pt
    boxes = [(px - r, py - r, px + r, py + r) for px, py in P]
    boxes += [tuple(a.get_window_extent(rend).extents) for a in avoid]

    def size(name):
        t = ax.text(0, 0, name, fontsize=fontsize)
        e = t.get_window_extent(rend)
        t.remove()
        return e.width + 2 * pad, e.height + 2 * pad

    def free(b):
        if b[0] < bb.x0 or b[2] > bb.x1 or b[1] < bb.y0 or b[3] > bb.y1:
            return False
        return all(b[2] <= o[0] or b[0] >= o[2] or b[3] <= o[1] or b[1] >= o[3] for o in boxes)

    def crosses(p, b, own):
        q = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for o in boxes:
            if o == own:
                continue
            t0, t1 = 0.0, 1.0
            for d, lo, hi, s0 in ((q[0] - p[0], o[0], o[2], p[0]), (q[1] - p[1], o[1], o[3], p[1])):
                if abs(d) < 1e-9:
                    if s0 < lo or s0 > hi:
                        t0, t1 = 1.0, 0.0
                    continue
                a, c = sorted(((lo - s0) / d, (hi - s0) / d))
                t0, t1 = max(t0, a), min(t1, c)
            if t0 < t1:
                return True
        return False

    def put(i, b, leader, relpos=(0.5, 0.5)):
        boxes.append(b)
        lx, ly = inv.transform(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        ax.annotate(pts[i][2], pts[i][:2], xytext=(lx, ly), textcoords="data", fontsize=fontsize, ha="center",
                    va="center", color="#3f4a56", zorder=4, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
                    arrowprops=dict(arrowstyle="-", color="#a9b1ba", linewidth=0.6, shrinkA=0, shrinkB=2.5, relpos=relpos)
                    if leader else None)


    placed = set()
    for name, (lx, ly, leader, *ha) in (manual or {}).items():
        i = next(k for k, p in enumerate(pts) if p[2] == name)
        w, h = size(name)
        cx, cy = tr.transform((lx, ly))
        if ha == ["right"]:
            cx -= w / 2
        put(i, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), leader, (1, 0.5) if ha == ["right"] else (0.5, 0.5))
        placed.add(i)

    dense = [] if cluster is None else [i for i, (x, y, _) in enumerate(pts)
                                        if cluster[0] <= x <= cluster[1] and cluster[2] <= y <= cluster[3] and i not in placed]
    if dense:
        xs = sorted(P[i][0] for i in dense)

        cuts = [(a + b) / 2 for a, b in zip(xs, xs[1:]) if b - a > 1]
        split = min(cuts, key=lambda c: abs(2 * sum(x < c for x in xs) - len(xs))) if cuts else xs[0] - 1
        if split_x is not None:
            split = tr.transform((split_x, 0))[0]
        x_lo, x_hi = xs[0] - 14 * pt, xs[-1] + 14 * pt
        for side in ("left", "right"):
            sel = sorted([i for i in dense if (P[i][0] < split) == (side == "left")], key=lambda i: (-P[i][1], -P[i][0] if side == "left" else P[i][0]))
            if not sel:
                continue
            sz = [size(pts[i][2]) for i in sel]

            ys = []
            for k, i in enumerate(sel):
                y = P[i][1]
                if ys and ys[-1] - y < sz[k][1]:
                    y = ys[-1] - sz[k][1]
                ys.append(y)
            shift = (sum(P[i][1] for i in sel) - sum(ys)) / len(sel)
            ys = [y + shift for y in ys]


            step = -2 * pt if side == "left" else 2 * pt

            clear = [b if i in dense else (b[0] - 12 * pt, b[1] - 3 * pt, b[2] + 12 * pt, b[3] + 3 * pt)
                     for i, b in enumerate(boxes[:len(P)])]
            for move in range(200):
                xs0 = [(x_lo - sz[k][0] if side == "left" else x_hi) + move * step for k in range(len(sel))]
                if not any(xs0[k] < o[2] and xs0[k] + sz[k][0] > o[0] and ys[k] - sz[k][1] / 2 < o[3]
                           and ys[k] + sz[k][1] / 2 > o[1] for k in range(len(sel)) for o in clear):
                    break
            for k, i in enumerate(sel):
                w, h = sz[k]
                x0 = max(xs0[k], bb.x0 + 2) if side == "left" else min(xs0[k], bb.x1 - 2 - w)
                put(i, (x0, ys[k] - h / 2, x0 + w, ys[k] + h / 2), True)
    for i in sorted(set(range(len(pts))) - set(dense) - placed, key=lambda i: -P[i][1]):
        w, h = size(pts[i][2])
        px, py = P[i]
        cands = [(px + r + 1, py - h / 2), (px - r - 1 - w, py - h / 2),
                 (px - w / 2, py + r + 1), (px - w / 2, py - r - 1 - h)]
        for k in range(1, 8):
            for dy in (k * h * 0.6, -k * h * 0.6):
                cands += [(px + r + 1, py + dy - h / 2), (px - r - 1 - w, py + dy - h / 2)]
        for k in range(1, 8):
            cands += [(px - w / 2, py + r + 1 + k * h * 0.6), (px - w / 2, py - r - 1 - h - k * h * 0.6)]
        def cost(b):
            c = sum(max(0, min(b[2], o[2]) - max(b[0], o[0])) * max(0, min(b[3], o[3]) - max(b[1], o[1]))
                    for o in boxes)
            return c + 10 * (max(0, bb.x0 - b[0]) + max(0, b[2] - bb.x1)) * h
        boxes_c = [(x0, y0, x0 + w, y0 + h) for x0, y0 in cands]
        own = (px - r, py - r, px + r, py + r)
        k = next((k for k, b in enumerate(boxes_c) if free(b) and (k < 4 or not crosses((px, py), b, own))), None)
        if k is None:
            k = min(range(len(boxes_c)), key=lambda k: cost(boxes_c[k]) + (w * h if k >= 4 and crosses((px, py), boxes_c[k], own) else 0))
        put(i, boxes_c[k], k >= 4)


def tex(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def main():
    rows = {r["key"]: r for r in csv.DictReader(open(os.path.join(ROOT, "docs", "methods_table.csv"), encoding="utf-8"))}
    names = {}
    import re
    bib = open(os.path.join(ROOT, "paper", "references.bib"), encoding="utf-8").read()
    for m in re.finditer(r"^@\w+\{([^,]+),(.*?)^\}", bib, re.M | re.S):
        a = re.search(r"author\s*=\s*\{(.*?)\},", m.group(2), re.S)
        fam = [re.sub(r"[{}\\]", "", n.split(",")[0]).strip() for n in a.group(1).split(" and ")] if a else []
        names[m.group(1)] = fam[0] + (" et al." if len(fam) > 2 else (" and " + fam[1] if len(fam) == 2 else ""))
    cited = paper_scope.cited_keys()
    sel = []
    for k, (setting, gpu, base) in ANNOT.items():
        if k not in rows or k not in cited:
            continue
        r = rows[k]
        top1, gf, thr = r["deits_top1"], r["deits_gflops"], r["deits_throughput"]
        if not top1 or top1 == "?":
            continue
        sel.append((r["year"], k, SHORT.get(k, r["name"] or names[k]), setting, top1, gf, thr, gpu, base))
    sel.sort(key=lambda t: (t[0], t[1]))
    os.makedirs(os.path.join(ROOT, "paper", "tables"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "paper", "figures"), exist_ok=True)


    def chapter(k):
        op = rows[k]["operation"].replace(",", ";").split(";")[0].strip()
        if op in ("halt", "resolution") or "halting" in rows[k]["schedule"] or k in INPUT_KEYS:
            return "input"
        return "pruning" if family(rows[k]["operation"]) in ("Pruning", "Other") else "merging"

    def first(k, axis):
        return rows[k][axis].replace(",", ";").split(";")[0].strip()

    def fam_pruning(k):
        if k in RL_KEYS or first(k, "criterion") == "learned-predictor":
            return "Learned selectors"
        if first(k, "criterion") == "cls-attention" and k not in GRAPH_KEYS:
            return "Class attention"
        return "Other scores"

    def fam_merging(k):
        if k in SUMMARY_KEYS or first(k, "operation") == "resample":
            return "Learned summaries"
        return {"merge": "Matching", "cluster": "Clustering and pooling"}.get(first(k, "operation"), "Hybrid")

    def fam_input(k):
        if k in INPUT_KEYS:
            return "Selection at the input"
        return "Resolution" if first(k, "operation") == "resolution" else "Halting"

    groups = {"pruning": (fam_pruning, ["Learned selectors", "Class attention", "Other scores"]),
              "merging": (fam_merging, ["Matching", "Clustering and pooling", "Learned summaries", "Hybrid"]),
              "input": (fam_input, ["Halting", "Selection at the input", "Resolution"])}
    lines = {part: [] for part in groups}
    for year, k, name, setting, top1, gf, thr, gpu, base in sel:
        gf_s = "%.1f" % float(gf.split(" ")[0]) if gf and gf != "?" else "--"
        thr_s = "%d" % round(float(thr)) if thr and thr != "?" else "--"
        sp_s = "%.2f$\\times$" % (float(thr) / base) if base and thr_s != "--" else "--"
        if k in LOWER_BOUND and thr_s != "--":
            thr_s, sp_s = "$\\geq$" + thr_s, "$\\geq$" + sp_s
        top1_s = str(Decimal(top1).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        lines[chapter(k)].append((k, " & ".join([tex(name) + "~\\cite{%s}" % k, venue_of(k), tex(setting),
                                                 top1_s, gf_s, thr_s, sp_s, tex(gpu) or "--"]) + " \\\\"))
    for old in ("adaptive",):
        path = os.path.join(ROOT, "paper", "tables", "deits_results_%s.tex" % old)
        if os.path.exists(path):
            os.remove(path)
    for part, ls in lines.items():
        fam, order = groups[part]
        with open(os.path.join(ROOT, "paper", "tables", "deits_results_%s.tex" % part), "w", encoding="utf-8") as fh:
            fh.write("% generated by tools/build_results.py, do not edit by hand\n")
            fh.write("\\begin{footnotesize}\n\\setlength{\\tabcolsep}{3pt}\n"
                     "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{2.9cm}lY"
                     "rrrr>{\\raggedright\\arraybackslash}p{2.1cm}@{}}\n\\toprule\n")
            fh.write("Method & Venue & Setting & Top-1 (\\%) & GFLOPs & img/s & Speed-up & GPU, batch \\\\\n\\midrule\n")
            fh.write("DeiT-S~\\cite{touvron2021deit} & ICML 2021 & unreduced & 79.8 & 4.6 & -- & -- & -- \\\\\n\\midrule\n")
            fh.write(grouped(ls, fam, 8, order=order) + "\n\\bottomrule\n\\end{tabularx}\n\\end{footnotesize}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_style as S
    S.apply()

    groups = ["training-free", "fine-tuned or light", "trained from scratch"]
    pts = []
    for year, k, name, setting, top1, gf, thr, gpu, base in sel:
        if k in EXCLUDE_FIG or not gf or gf == "?":
            continue
        pts.append((float(gf.split(" ")[0]), float(top1), SHORT.get(k, name), group(setting)))

    fig, axes = plt.subplots(1, 3, figsize=(S.TEXTWIDTH_IN, 3.5), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.9, 1]))
    for ax, g in zip(axes, groups):
        sub = [p for p in pts if p[3] == g]
        edge, face, marker = S.SETTING[g]
        ax.axhline(BASELINE[0], color=S.REF, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
        ax.scatter([p[0] for p in sub], [p[1] for p in sub], marker=marker, s=46, color=face, edgecolor=edge,
                   linewidth=1.0, zorder=3)
        ax.set_title("%s (%d)" % (g.capitalize(), len(sub)), loc="left")
        ax.set_xlim(0.5, 5.15 if g == "fine-tuned or light" else 5.0)
        ax.set_ylim(77.5, 80.6)
        ax.set_xticks([1, 2, 3, 4, 5] if g == "fine-tuned or light" else [1, 2, 3, 4])
        ax.set_xlabel("GFLOPs (reported)")
    axes[0].set_ylabel("ImageNet-1k top-1 (%)")
    base_text = axes[0].text(1.05, BASELINE[0] + 0.04, "unreduced DeiT-S", ha="left", va="bottom",
                             fontsize=7, color=S.MUTED)
    fig.tight_layout(w_pad=0.8)
    for ax, g in zip(axes, groups):
        same = {}
        for x, y, n, gg in pts:
            if gg == g:
                same.setdefault((x, y), []).append(n)
        place_labels(ax, [(x, y, ", ".join(ns)) for (x, y), ns in same.items()], [base_text] if ax is axes[0] else [],
                     cluster={"fine-tuned or light": (2.25, 3.15, 79.05, 79.95),
                              "trained from scratch": (1.6, 2.1, 79.7, 79.9)}.get(g),
                     manual={"fine-tuned or light": {"MIA-Former": (4.55, 80.18, True),
                                                     "Freq.-aware": (4.66, 80.01, True),


                                                     **{n: (x[0] if x else 2.1, y, True, "right") for n, y, *x in (
                                                         ("DTEM", 80.08), ("PPT", 79.98), ("ATS", 79.88), ("PM-ViT", 79.71),
                                                         ("Patch Slimming, STAR", 79.55, 2.6), ("SiT", 79.42), ("SPViT", 79.29),
                                                         ("DynamicViT", 79.2), ("PaPr", 79.1), ("IA-RED$^2$", 79.0))},
                                                     "MCTF": (2.6, 80.24, False)},

                             "trained from scratch": {"CF-ViT": (1.4, 80.06, True, "right"), "LF-ViT": (1.4, 79.56, True, "right"),
                                                      "STViT": (4.3, 79.88, True), "MG-ViT": (4.25, 79.73, True)}}.get(g),
                     split_x={"fine-tuned or light": 2.95}.get(g))
    S.save(fig, os.path.join(ROOT, "paper", "figures", "deits_acc_flops"))
    print("table rows:", len(sel), "figure points:", len(pts))


if __name__ == "__main__":
    main()
