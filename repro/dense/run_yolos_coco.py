#!/usr/bin/env python3
"""Usage: VTR_WORK=... HF_HOME=$VTR_WORK/hf_cache venv_hf/bin/python -I repro/dense/run_yolos_coco.py --runs ..."""
import argparse, contextlib, csv, io, json, math, os, sys, time
import numpy as np, torch
from PIL import Image

WORK = os.environ["VTR_WORK"]
sys.path.insert(0, os.path.join(WORK, "official", "ToMe"))
from tome.merge import bipartite_soft_matching, merge_wavg, merge_source
from transformers import YolosForObjectDetection, AutoImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_DIR = os.path.join(os.environ["VTR_DATASETS"], "coco")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "repro", "results_coco_yolos.csv")
LOCS = (3, 6, 9)
NDET = 100


class Reduced:
    def __init__(self, model, method, amount):
        self.m, self.method, self.amount = model, method, amount
        self.D = model.config.hidden_size
        self.macs = 0.0

    def cost(self, n_att, n_mlp):
        D = self.D
        self.macs += 4 * n_att * D * D + 2 * n_att * n_att * D + 8 * n_mlp * D * D

    def attn(self, layer, x, size):
        a = layer.attention.attention
        B, N, C = x.shape
        H = a.num_attention_heads
        hd = C // H
        q = a.query(x).view(B, N, H, hd).transpose(1, 2)
        k = a.key(x).view(B, N, H, hd).transpose(1, 2)
        v = a.value(x).view(B, N, H, hd).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) * a.scaling
        if size is not None:
            logits = logits + size.log()[:, None, None, :, 0]
        p = logits.softmax(-1)
        o = layer.attention.output.dense((p @ v).transpose(1, 2).reshape(B, N, C))
        return o, k.mean(1), p

    @torch.no_grad()
    def __call__(self, pixel_values):
        vit = self.m.vit
        enc = vit.encoder
        H, W = pixel_values.shape[-2:]
        x = vit.embeddings(pixel_values)
        mid = enc.interpolation(enc.mid_position_embeddings, (H, W))
        B, N0, D = x.shape
        P = N0 - 1 - NDET
        size = torch.ones(B, N0, 1, device=x.device, dtype=x.dtype) if self.method == "tome" else None
        keep = torch.arange(P, device=x.device).expand(B, P)
        source = None
        for i, layer in enumerate(enc.layer):
            n_att = x.shape[1]
            o, metric, p = self.attn(layer, layer.layernorm_before(x), size)
            x = x + o
            n = x.shape[1] - 1 - NDET
            if self.method == "topk" and i in LOCS:
                k = math.ceil(self.amount * n)
                score = p[:, :, -NDET:, 1:1 + n].mean((1, 2))
                idx = score.argsort(-1, descending=True)[:, :k].sort(-1).values
                patches = x[:, 1:1 + n].gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
                x = torch.cat([x[:, :1], patches, x[:, -NDET:]], 1)
                keep = keep.gather(1, idx)
            elif self.method == "tome":
                r = min(int(self.amount), n // 2)
                if r > 0:
                    merge, _ = bipartite_soft_matching(metric[:, 1:1 + n], r, class_token=False, distill_token=False)
                    xp, sp = x[:, 1:1 + n], size[:, 1:1 + n]
                    source = merge_source(merge, xp, source)
                    xp, sp = merge_wavg(merge, xp, sp)
                    x = torch.cat([x[:, :1], xp, x[:, -NDET:]], 1)
                    size = torch.cat([size[:, :1], sp, size[:, -NDET:]], 1)
            x = layer.output(layer.intermediate(layer.layernorm_after(x)), x)
            self.cost(n_att, x.shape[1])
            if i < len(enc.layer) - 1:
                pe = mid[i].expand(B, -1, -1)
                pp = pe[:, 1:1 + P]
                if self.method == "topk":
                    pp = pp.gather(1, keep.unsqueeze(-1).expand(-1, -1, D))
                elif self.method == "tome" and source is not None:
                    pp = (source @ pp) / source.sum(-1, keepdim=True)
                x = x + torch.cat([pe[:, :1], pp, pe[:, -NDET:]], 1)
        x = vit.layernorm(x)[:, -NDET:]
        return self.m.class_labels_classifier(x), self.m.bbox_predictor(x).sigmoid()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="none"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small").to(dev).eval()
    proc = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(os.path.join(COCO_DIR, "annotations", "instances_val2017.json"))
    ids = sorted(gt.getImgIds())
    if args.limit:
        ids = ids[: args.limit]
    if args.check:
        im = Image.open(os.path.join(COCO_DIR, "val2017", gt.loadImgs(ids[0])[0]["file_name"])).convert("RGB")
        pv = proc(images=im, return_tensors="pt")["pixel_values"].to(dev)
        with torch.no_grad():
            ref = model(pixel_values=pv)
            lg, bx = Reduced(model, "tome", 0)(pv)
            lg2, bx2 = Reduced(model, "topk", 1.0)(pv)
        print("check max |diff| logits/boxes:", (ref.logits - lg).abs().max().item(), (ref.pred_boxes - bx).abs().max().item(),
              (ref.logits - lg2).abs().max().item(), flush=True)
        return
    done = set()
    if os.path.exists(OUT) and not args.limit:
        done = {r["run"] for r in csv.DictReader(open(OUT))}
    base = None
    for spec in args.runs.split(","):
        if spec in done:
            continue
        method = spec.split(":")[0]
        amount = float(spec.split(":")[1]) if ":" in spec else 0.0
        red = Reduced(model, "tome" if method == "none" else method, amount)
        unred = Reduced(model, "tome", 0)
        dets, t0 = [], time.time()
        for img_id in ids:
            info = gt.loadImgs(img_id)[0]
            im = Image.open(os.path.join(COCO_DIR, "val2017", info["file_name"])).convert("RGB")
            pv = proc(images=im, return_tensors="pt")["pixel_values"].to(dev)
            n0 = pv.shape[-2] * pv.shape[-1] // 256 + 1 + NDET
            for _ in range(12):
                unred.cost(n0, n0)
            logits, boxes = red(pv)

            class O:
                pass
            o = O(); o.logits, o.pred_boxes = logits, boxes
            res = proc.post_process_object_detection(o, threshold=0.0, target_sizes=[(info["height"], info["width"])])[0]
            for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
                dets.append({"image_id": img_id, "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dets)
            E = COCOeval(gt, dt, "bbox"); E.params.imgIds = ids
            E.evaluate(); E.accumulate(); E.summarize()
        st = E.stats
        row = {"model": "yolos_small", "run": spec, "method": method, "amount": amount,
               "macs_frac": round(red.macs / unred.macs, 3), "ap": round(100 * float(st[0]), 2), "ap50": round(100 * float(st[1]), 2),
               "ap75": round(100 * float(st[2]), 2), "ap_s": round(100 * float(st[3]), 2), "ap_m": round(100 * float(st[4]), 2), "ap_l": round(100 * float(st[5]), 2),
               "images": len(ids), "seconds": round(time.time() - t0, 1)}
        print(row, flush=True)
        if not args.limit:
            new = not os.path.exists(OUT)
            with open(OUT, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)


if __name__ == "__main__":
    main()
