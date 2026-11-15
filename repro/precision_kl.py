import csv
import os

import numpy as np

LGT = os.path.join(os.environ["VTR_WORK"], "logits")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_imagenet_precision_kl.csv")


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


rows = []
for key in ["none", "topk_0.7", "evit_0.7", "tome_r13", "diffrate_2.9"]:
    if not os.path.exists(os.path.join(LGT, "fp32_" + key + ".npy")):
        continue
    ref = log_softmax(np.load(os.path.join(LGT, "fp32_" + key + ".npy")))
    p = np.exp(ref)
    top2 = np.sort(p, 1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    margin, ties = np.median(gap), int((gap < 0.01).sum())
    for d in ["fp16", "bf16", "fp16m32", "bf16m32"]:
        f = os.path.join(LGT, f"{d}_{key}.npy")
        if not os.path.exists(f):
            continue
        q = log_softmax(np.load(f))
        kl = (p * (ref - q)).sum(1)
        flip = ref.argmax(1) != q.argmax(1)
        rows.append({"model": key, "dtype": d, "flips": int(flip.sum()),
                     "kl_mean": "%.3e" % kl.mean(), "kl_median": "%.3e" % np.median(kl),
                     "kl_p95": "%.3e" % np.percentile(kl, 95), "kl_max": "%.3e" % kl.max(),
                     "fp32_margin_median": "%.3f" % margin,
                     "fp32_near_ties": ties, "flip_margin_median": "%.4f" % np.median(gap[flip])})
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(open(OUT).read())
