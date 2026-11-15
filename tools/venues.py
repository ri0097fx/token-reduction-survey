#!/usr/bin/env python3
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNALS = {
    "Transactions on Machine Learning Research": "TMLR",
    "IEEE Transactions on Image Processing": "TIP",
    "IEEE Transactions on Multimedia": "TMM",
    "IEEE Transactions on Neural Networks and Learning Systems": "TNNLS",
    "IEEE Transactions on Circuits and Systems for Video Technology": "TCSVT",
    "IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems": "TCAD",
    "IEEE Transactions on Computers": "IEEE TC",
    "IEEE Internet of Things Journal": "IEEE IoT J.",
    "IEEE Open Journal of the Communications Society": "IEEE OJ-COMS",
    "International Journal of Computer Vision": "IJCV",
    "Pattern Recognition": "Pattern Recognit.",
    "Expert Systems with Applications": "Expert Syst. Appl.",
    "Remote Sensing": "Remote Sens.",
    "Journal of Electronic Imaging": "J. Electron. Imaging",
    "Journal of Parallel and Distributed Computing": "J. Parallel Distrib. Comput.",
    "Automation in Construction": "Autom. Constr.",
    "Neurocomputing": "Neurocomput.",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence": "TPAMI",
    "Applied Intelligence": "Appl. Intell.",
    "Visual Intelligence": "Vis. Intell.",
    "Journal of Information, Technology and Policy": "JITP",
}


def _entries():
    bib = open(os.path.join(ROOT, "paper", "references.bib"), encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^@(\w+)\{([^,]+),(.*?)^\}", bib, re.M | re.S):
        out[m.group(2)] = (m.group(1).lower(), m.group(3))
    return out


_ENTRIES = None


def year_of(key):
    global _ENTRIES
    if _ENTRIES is None:
        _ENTRIES = _entries()
    m = re.search(r"year\s*=\s*\{(\d{4})\}", _ENTRIES[key][1])
    return int(m.group(1)) if m else 0


def venue_of(key):
    global _ENTRIES
    if _ENTRIES is None:
        _ENTRIES = _entries()
    kind, body = _ENTRIES[key]
    year = re.search(r"year\s*=\s*\{(\d{4})\}", body)
    year = year.group(1) if year else ""
    b = re.search(r"booktitle\s*=\s*\{(.*?)\},", body, re.S)
    j = re.search(r"journal\s*=\s*\{(.*?)\},", body, re.S)
    if b:
        t = " ".join(b.group(1).split())
        acr = re.search(r"\(([A-Z][A-Za-z-]*(?: [A-Z][A-Za-z-]*)*)(?: (\d{4}))?\)", t)
        if acr:
            name = acr.group(1)
            if acr.group(2):
                return name + " " + acr.group(2)
            if "Findings" in t:
                name += " Findings"
        elif "ECCV" in t:
            name = "ECCV"
        elif re.search(r"Findings of .*EMNLP", t):
            name = "EMNLP Findings"
        else:
            name = t
    elif j:
        t = " ".join(j.group(1).split())
        name = JOURNALS.get(t, t)
    elif re.search(r"eprint|arxiv", body, re.I):
        name = "arXiv"
    else:
        return "?"
    return (name + " " + year).strip()


if __name__ == "__main__":
    import sys
    for k in sys.argv[1:]:
        print(k, venue_of(k))
