#!/usr/bin/env python3
"""Usage: DETECTRON2_DATASETS=$VTR_DATASETS python repro/dense/run_vitdet_coco.py --runs ..."""
import argparse, collections, csv, importlib.util, io, os, pickle, time, types
import numpy as np, torch

WORK = os.environ["VTR_WORK"]
D2 = os.environ["DETECTRON2_REPO"]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "repro", "results_coco_vitdet.csv")
CKPT = os.path.join(WORK, "weights", "vitdet_b", "model_final_61ccd1.pkl")
_spec = importlib.util.spec_from_file_location("tomesd_merge", os.path.join(WORK, "venv_hf", "lib", "python3.13", "site-packages", "tomesd", "merge.py"))
tomesd_merge = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tomesd_merge)

from detectron2.config import LazyConfig, instantiate
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


class SafeUnpickler(pickle.Unpickler):
    OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
          ("numpy", "dtype"), ("collections", "OrderedDict"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError("blocked %s.%s" % (module, name))


def load_weights(model):
    with open(CKPT, "rb") as fh:
        ck = SafeUnpickler(fh, encoding="latin1").load()
    sd = {k: torch.as_tensor(np.asarray(v)) for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])


def rel_term(q, table, pos):
    G = (table.shape[0] + 1) // 2
    qt = q @ table.t()
    BH, n, _ = qt.shape; B = pos.shape[0]
    idx = (pos[:, :, None] - pos[:, None, :] + G - 1).repeat_interleave(BH // B, 0)
    return qt.gather(2, idx)


def merged_attention(attn, x, ys, xs):
    B, n, C = x.shape; H = attn.num_heads
    qkv = attn.qkv(x).reshape(B, n, 3, H, -1).permute(2, 0, 3, 1, 4).reshape(3, B * H, n, -1)
    q, k, v = qkv.unbind(0)
    w = (q * attn.scale) @ k.transpose(-2, -1)
    if attn.use_rel_pos:
        w = w + rel_term(q, attn.rel_pos_h, ys) + rel_term(q, attn.rel_pos_w, xs)
    w = w.softmax(-1)
    return attn.proj((w @ v).reshape(B, H, n, -1).permute(0, 2, 1, 3).reshape(B, n, C))


def patch(backbone, ratio, scope):
    for blk in backbone.blocks:
        if blk.window_size > 0:
            continue

        def forward(self, x, ratio=ratio, scope=scope):
            B, Hh, Ww, C = x.shape
            N = Hh * Ww
            res = x
            h = self.norm1(x).reshape(B, N, C)
            m, u = tomesd_merge.bipartite_soft_matching_random2d(h, Ww, Hh, 2, 2, int(N * ratio), no_rand=True)
            grid = torch.stack(torch.meshgrid(torch.arange(Hh, device=h.device), torch.arange(Ww, device=h.device), indexing="ij"), -1)
            coords = m(grid.reshape(1, N, 2).expand(B, -1, -1).to(h.dtype), mode="mean").round().long()
            a = merged_attention(self.attn, m(h), coords[..., 0], coords[..., 1])
            if scope == "attn_mlp":
                xm = m(res.reshape(B, N, C)) + a
                xm = xm + self.mlp(self.norm2(xm))
                return u(xm).reshape(B, Hh, Ww, C)
            x = res + u(a).reshape(B, Hh, Ww, C)
            return x + self.mlp(self.norm2(x))
        blk.forward = types.MethodType(forward, blk)


def macs_frac(backbone, ratio, scope, N=4096, Npad=4900, w=196):
    D = backbone.blocks[0].attn.qkv.in_features
    full = red = 0.0
    for blk in backbone.blocks:
        if blk.window_size > 0:
            c = 4 * Npad * D * D + 2 * Npad * w * D + 8 * N * D * D
            full += c; red += c
        else:
            n = N - int(N * ratio)
            full += 12 * N * D * D + 2 * N * N * D
            red += 4 * n * D * D + 2 * n * n * D + 8 * (n if scope == "attn_mlp" else N) * D * D
    return red / full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="none"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg = LazyConfig.load(os.path.join(D2, "projects", "ViTDet", "configs", "COCO", "mask_rcnn_vitdet_b_100ep.py"))
    if args.limit:
        cfg.dataloader.test.dataset.names = "coco_2017_val"
    done = set()
    if os.path.exists(OUT) and not args.limit:
        done = {r["run"] for r in csv.DictReader(open(OUT))}
    for spec in args.runs.split(","):
        if spec in done:
            continue
        scope, ratio = (spec.split(":") + ["0"])[:2]; ratio = float(ratio)
        model = instantiate(cfg.model).cuda().eval()
        load_weights(model)
        bb = model.backbone.net
        if args.check:
            ref = instantiate(cfg.model).cuda().eval(); load_weights(ref)
            patch(bb, 0.0, "attn")
            x = torch.randn(1, 3, 1024, 1024, device="cuda")
            with torch.no_grad():
                d = (bb(x)["last_feat"] - ref.backbone.net(x)["last_feat"]).abs().max().item()
            print("check: max |diff| of the backbone output with r = 0: %.2e" % d, flush=True)
            return
        if scope != "none":
            patch(bb, ratio, scope)
        loader = instantiate(cfg.dataloader.test)
        if args.limit:
            import itertools

            class Limited:
                def __init__(self, it, n): self.it, self.n = it, n
                def __iter__(self): return itertools.islice(iter(self.it), self.n)
                def __len__(self): return self.n
            loader = Limited(loader, args.limit)
        ev = COCOEvaluator("coco_2017_val", output_dir=None)
        t0 = time.time()
        with torch.no_grad():
            res = inference_on_dataset(model, loader, ev)
        row = {"model": "vitdet_b_mask_rcnn", "run": spec, "scope": scope, "ratio": ratio,
               "macs_frac": round(macs_frac(bb, ratio, scope), 3),
               "box_ap": round(res["bbox"]["AP"], 2), "box_ap50": round(res["bbox"]["AP50"], 2), "box_ap_s": round(res["bbox"]["APs"], 2),
               "box_ap_l": round(res["bbox"]["APl"], 2), "mask_ap": round(res["segm"]["AP"], 2),
               "images": args.limit or 5000, "seconds": round(time.time() - t0, 1)}
        print(row, flush=True)
        if not args.limit:
            new = not os.path.exists(OUT)
            with open(OUT, "a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    wr.writeheader()
                wr.writerow(row)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
