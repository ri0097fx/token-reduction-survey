#!/usr/bin/env python3
"""Usage: venv_timm04/bin/python -I repro/dense/vis_segmenter.py IMAGE_DIR OUT_DIR [--model b] [--runs none,topk:0.7,tome:40,algm:0.92]"""
import argparse, os, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_segmenter_ade as R


@torch.no_grad()
def predict(fwd, path, dev, crop, stride, wbs=4):
    im = R.mmseg_resize(Image.open(path).convert("RGB"), crop)
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float().to(dev)
    x = (x / 255.0 - 0.5) / 0.5
    win = R.sliding_window(x, False, crop, stride)
    cr = torch.stack(win.pop("crop"))[:, 0]
    seg = torch.zeros((len(cr), 150, crop, crop), device=dev)
    for i in range(0, len(cr), wbs):
        seg[i:i + wbs] = fwd(cr[i:i + wbs])
    win["seg_maps"] = seg
    logit = R.merge_windows(win, crop, (im.size[1], im.size[0]))
    return logit.argmax(0).cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir"); ap.add_argument("out_dir")
    ap.add_argument("--model", default="b"); ap.add_argument("--runs", default="none,topk:0.7,tome:40,algm:0.92")
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(args.image_dir) if f.endswith(".jpg"))
    for spec in args.runs.split(","):
        model, D = R.build(args.model)
        model = model.to(dev).eval()
        method = spec.split(":")[0]
        amount = float(spec.split(":")[1]) if ":" in spec else 0.0
        if method == "none":
            fwd = model
        elif method == "algm":
            R.algm.patch.algm_segmenter_patch(model.cpu(), list(R.ALGM_LAYERS[args.model]), trace_source=True)
            model.encoder.window_size = (2, 2)
            model.encoder.threshold = amount
            fwd = model = model.to(dev).eval()
        else:
            fwd = R.Reduced(model, method, amount, None).eval()
        out = {n[:-4]: predict(fwd, os.path.join(args.image_dir, n), dev, *R.CROP[args.model]) for n in names}
        np.savez_compressed(os.path.join(args.out_dir, "seg_%s.npz" % spec.replace(":", "_")), **out)
        print(spec, {k: v.shape for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
