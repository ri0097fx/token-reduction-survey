#!/usr/bin/env python3
"""Usage: python3 -I tools/citation_counts.py OUT_CSV"""
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "vit-token-reduction-survey/0.1 (citation counts)"}


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(2 ** attempt)
    return None


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def main():
    out = sys.argv[1]
    bib = open(os.path.join(ROOT, "paper", "references.bib"), encoding="utf-8").read()
    info = {}
    for m in re.finditer(r"^@\w+\{([^,]+),(.*?)^\}", bib, re.M | re.S):
        body = m.group(2)
        t = re.search(r"title\s*=\s*\{(.*?)\},\s*\n", body, re.S)
        d = re.search(r"doi\s*=\s*\{(.*?)\}", body)
        info[m.group(1)] = (re.sub(r"[{}\\$^]", "", t.group(1)) if t else "", d.group(1) if d else "")
    rows = list(csv.DictReader(open(os.path.join(ROOT, "docs", "methods_table.csv"), encoding="utf-8")))
    res = []
    for r in rows:
        k = r["key"]
        title, doi = info.get(k, ("", ""))
        counts = []
        if doi:
            w = get("https://api.openalex.org/works/doi:%s?select=display_name,cited_by_count" % doi)
            if w:
                counts.append(("doi", w.get("cited_by_count", 0), w.get("display_name")))
        q = urllib.parse.quote(title.replace(",", " ").replace(":", " "))
        s = get("https://api.openalex.org/works?search=%s&per-page=8&select=display_name,cited_by_count,publication_year" % q)
        for w in (s or {}).get("results", []):
            if difflib.SequenceMatcher(None, norm(w.get("display_name")), norm(title)).ratio() >= 0.9:
                counts.append(("title", w.get("cited_by_count", 0), w.get("display_name")))
        best = max([c[1] for c in counts], default=None)
        res.append({"key": k, "name": r["name"], "year": r["year"], "target": r["target"], "citations": best,
                    "records": "; ".join("%s:%s" % (c[0], c[1]) for c in counts)})
        print(k, best, flush=True)
        time.sleep(0.3)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)


if __name__ == "__main__":
    main()
