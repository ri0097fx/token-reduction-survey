import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cited_keys():
    with open(os.path.join(ROOT, "docs", "cited_keys.txt"), encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}
