#!/usr/bin/env python3
"""Usage: HF_HOME=$VTR_WORK/hf_cache venv_hf/bin/python -I repro/dense/vis_yolos.py IMAGE_DIR OUT_DIR [--runs none,topk:0.6,tome:250]"""
import argparse, json, os, sys
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_yolos_coco as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir"); ap.add_argument("out_dir")
    ap.add_argument("--runs", default="none,topk:0.6,tome:250"); ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    model = R.YolosForObjectDetection.from_pretrained("hustvl/yolos-small").to(dev).eval()
    proc = R.AutoImageProcessor.from_pretrained("hustvl/yolos-small")
    os.makedirs(args.out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(args.image_dir) if f.endswith(".jpg"))
    for spec in args.runs.split(","):
        method = spec.split(":")[0]
        amount = float(spec.split(":")[1]) if ":" in spec else 0.0
        red = R.Reduced(model, "tome" if method == "none" else method, amount)
        out = {}
        for n in names:
            im = Image.open(os.path.join(args.image_dir, n)).convert("RGB")
            pv = proc(images=im, return_tensors="pt")["pixel_values"].to(dev)
            logits, boxes = red(pv)

            class O:
                pass
            o = O(); o.logits, o.pred_boxes = logits, boxes
            res = proc.post_process_object_detection(o, threshold=args.thr, target_sizes=[(im.height, im.width)])[0]
            out[n[:-4]] = [[round(v, 1) for v in b] + [round(s, 3), model.config.id2label[l]]
                           for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist())]
        json.dump(out, open(os.path.join(args.out_dir, "det_%s.json" % spec.replace(":", "_")), "w"))
        print(spec, {k: len(v) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
