#!/usr/bin/env python3
"""Usage: VTR_WORK=... venv_timm04/bin/python -I repro/dense/run_segmenter_ade.py --model s [--limit N] --runs ..."""
import argparse, csv, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

WORK = os.environ["VTR_WORK"]
sys.path[:0] = [os.path.join(WORK, "official", "ALGM"), os.path.join(WORK, "official", "ALGM", "algm"), os.path.join(WORK, "official", "ToMe")]
from segm.model.vit import VisionTransformer
from segm.model.decoder import MaskTransformer
from segm.model.segmenter import Segmenter
from segm.model.utils import sliding_window, merge_windows, resize_pos_embed
from segm.model.blocks import Block
import algm.patch
from tome.merge import bipartite_soft_matching, merge_wavg, merge_source
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


EXP = os.path.join(WORK, "official", "Expedit-Segmenter", "mmseg", "ops")
exp_cluster = _load("exp_cluster", os.path.join(EXP, "cluster.py"))
exp_recon = _load("exp_recon", os.path.join(EXP, "reconstruction.py"))

ADE = os.path.join(os.environ["VTR_DATASETS"], "ADEChallengeData2016", "ADEChallengeData2016")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "repro", "results_ade20k_segmenter.csv")
CKPT = {"s": ("segmenter_mmseg_s", "segmenter_vit-s_mask_8x1_512x512_160k_ade20k_20220105_151706-511bb103.pth", 384, 6),
        "b": ("segmenter_mmseg_b", "segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth", 768, 12),
        "l": ("segmenter_mmseg_l", "segmenter_vit-l_mask_8x1_512x512_160k_ade20k_20220105_162750-7ef345be.pth", 1024, 16)}
DEPTH = {"s": 12, "b": 12, "l": 24}
CROP = {"s": (512, 480), "b": (512, 480), "l": (640, 608)}
ALGM_LAYERS = {"s": [1, 5], "b": [1, 5], "l": [1, 7]}
LOCS = {12: (3, 6, 9), 24: (7, 13, 19)}


def build(size):
    d, f, D, H = CKPT[size]
    c = CROP[size][0]
    enc = VisionTransformer((c, c), 16, DEPTH[size], D, 4 * D, H, 1000, dropout=0.0, drop_path_rate=0.0)
    dec = MaskTransformer(150, 16, D, 2, H, D, 4 * D, 0.0, 0.0)
    model = Segmenter(enc, dec, n_cls=150)
    sd = torch.load(os.path.join(WORK, "weights", d, f), map_location="cpu", weights_only=True)["state_dict"]
    out = {}
    for k, v in sd.items():
        k2 = k.replace("backbone.", "encoder.").replace("decode_head.", "decoder.")
        k2 = k2.replace("encoder.layers.", "encoder.blocks.").replace("decoder.layers.", "decoder.blocks.")
        k2 = k2.replace(".ln1.", ".norm1.").replace(".ln2.", ".norm2.").replace("encoder.norm1.", "encoder.norm.")
        k2 = k2.replace("attn.attn.in_proj_", "attn.qkv.").replace("attn.attn.out_proj.", "attn.proj.")
        k2 = k2.replace("ffn.layers.0.0.", "mlp.fc1.").replace("ffn.layers.1.", "mlp.fc2.")
        k2 = k2.replace("patch_embed.projection.", "patch_embed.proj.").replace("decoder.dec_proj.", "decoder.proj_dec.")
        if k2 == "decoder.patch_proj.weight":
            k2, v = "decoder.proj_patch", v.t().contiguous()
        if k2 == "decoder.classes_proj.weight":
            k2, v = "decoder.proj_classes", v.t().contiguous()
        out[k2] = v
    missing, unexpected = model.load_state_dict(out, strict=False)
    assert set(missing) <= {"encoder.head.weight", "encoder.head.bias"} and not unexpected, (missing, unexpected)
    for m in enc.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.eps = 1e-6
    return model, D


class Counter:
    def __init__(self, model, D):
        self.D, self.macs, self.hooks = D, 0.0, []
        for b in list(model.encoder.blocks) + list(model.decoder.blocks):
            self.hooks.append(b.attn.qkv.register_forward_pre_hook(self.att))
            self.hooks.append(b.mlp.register_forward_pre_hook(self.mlp))

    def att(self, mod, inp):
        B, N, _ = inp[0].shape
        self.macs += B * (4 * N * self.D ** 2 + 2 * N * N * self.D)

    def mlp(self, mod, inp):
        B, N, _ = inp[0].shape
        self.macs += B * 8 * N * self.D ** 2


def attn_size(blk, x, size):
    a = blk.attn
    B, N, C = x.shape
    qkv = a.qkv(x).reshape(B, N, 3, a.heads, C // a.heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    logits = (q @ k.transpose(-2, -1)) * a.scale
    if size is not None:
        logits = logits + size.log()[:, None, None, :, 0]
    p = logits.softmax(-1)
    return a.proj((p @ v).transpose(1, 2).reshape(B, N, C)), k.mean(1), p


class Reduced(torch.nn.Module):
    def __init__(self, model, method, amount, loc=None):
        super().__init__()
        self.m, self.method, self.amount = model, method, amount
        if method == "expedite":
            self.loc = loc
            self.clu = exp_cluster.TokenClusteringBlock(num_spixels=int(amount), n_iters=5, temperture=0.02, window_size=5)
            self.rec = exp_recon.TokenReconstructionBlock(cfg=dict(k=20, temperture=0.02))
        self.n_cls, self.patch_size = model.n_cls, model.patch_size

    def encode(self, im):
        enc = self.m.encoder
        B, _, H, W = im.shape
        x = enc.patch_embed(im)
        x = torch.cat((enc.cls_token.expand(B, -1, -1), x), 1)
        pos = enc.pos_embed
        if x.shape[1] != pos.shape[1]:
            pos = resize_pos_embed(pos, enc.patch_embed.grid_size, (H // 16, W // 16), 1)
        x = x + pos
        N0, D = x.shape[1], x.shape[2]
        size = source = None
        full = keep = None
        if self.method == "topk":
            full = x.clone()
            keep = torch.arange(N0, device=x.device).expand(B, N0).clone()
        for l, blk in enumerate(enc.blocks):
            if self.method in ("tome", "tomeu"):
                o, metric, _ = attn_size(blk, blk.norm1(x), size)
                x = x + o
                r = min(int(self.amount), (x.shape[1] - 1) // 2)
                if r > 0:
                    merge, _ = bipartite_soft_matching(metric, r, class_token=True, distill_token=False)
                    source = merge_source(merge, x, source)
                    x, size = merge_wavg(merge, x, size)
            elif self.method == "expedite":
                x = blk(x)
                if l == self.loc:
                    state = self.rec.derive_unpooler()
                    state.update_state(feat_before_pooling=x[:, 1:])
                    g = int(math.sqrt(x.shape[1] - 1))
                    xc, _ = self.clu(x[:, 1:].transpose(1, 2).reshape(B, D, g, g))
                    state.update_state(feat_after_pooling=xc)
                    x = torch.cat([x[:, :1], xc], 1)
                continue
            else:
                o, _, p = attn_size(blk, blk.norm1(x), None)
                x = x + o
                if l in LOCS[len(enc.blocks)]:
                    n_keep = math.ceil(self.amount * (x.shape[1] - 1))
                    cls_attn = p[:, :, 0, 1:].mean(1)
                    idx = cls_attn.argsort(-1, descending=True)[:, :n_keep].sort(-1).values + 1
                    idx = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), idx], 1)
                    full = full.scatter(1, keep.unsqueeze(-1).expand(-1, -1, D), x)
                    x, keep = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)), keep.gather(1, idx)
            x = x + blk.mlp(blk.norm2(x))
        if self.method == "topk":
            x = full.scatter(1, keep.unsqueeze(-1).expand(-1, -1, D), x)
        x = enc.norm(x)
        if self.method == "expedite" and x.shape[1] != N0:
            out, _ = state.call(x[:, 1:])
            x = torch.cat([x[:, :1], out], 1)
        if self.method == "tomeu" and source is not None:
            x = x.gather(1, source.argmax(1).unsqueeze(-1).expand(-1, -1, D))
            source = None
        return x, source

    def forward(self, im):
        H, W = im.shape[2], im.shape[3]
        x, source = self.encode(im)
        masks = self.m.decoder(x[:, 1:], (H, W), source)
        return F.interpolate(masks, size=(H, W), mode="bilinear")


def mmseg_resize(im, short=512):
    w, h = im.size
    s = min(2048 / max(w, h), short / min(w, h))
    return im.resize((int(w * s + 0.5), int(h * s + 0.5)), Image.BILINEAR)


@torch.no_grad()
def evaluate(fwd, counter, imgs, dev, crop=512, stride=480, wbs=4):
    inter = np.zeros(150); union = np.zeros(150); crops = 0
    for name in imgs:
        im = Image.open(os.path.join(ADE, "images", "validation", name)).convert("RGB")
        gt = np.array(Image.open(os.path.join(ADE, "annotations", "validation", name.replace(".jpg", ".png")))).astype(np.int64) - 1
        x = torch.from_numpy(np.array(mmseg_resize(im, crop))).permute(2, 0, 1)[None].float().to(dev)
        x = (x / 255.0 - 0.5) / 0.5
        win = sliding_window(x, False, crop, stride)
        cr = torch.stack(win.pop("crop"))[:, 0]
        seg = torch.zeros((len(cr), 150, crop, crop), device=dev)
        for i in range(0, len(cr), wbs):
            seg[i:i + wbs] = fwd(cr[i:i + wbs])
        crops += len(cr)
        win["seg_maps"] = seg
        logit = merge_windows(win, crop, gt.shape)
        pred = logit.argmax(0).cpu().numpy()
        valid = gt >= 0
        p, g = pred[valid], gt[valid]
        hit = np.bincount(g[p == g], minlength=150)[:150]
        inter += hit
        union += (np.bincount(p, minlength=150) + np.bincount(g, minlength=150))[:150] - hit
    miou = 100 * np.nanmean(inter / np.where(union > 0, union, np.nan))
    return float(miou), counter.macs / crops, crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="s"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--runs", default="none"); ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    imgs = sorted(os.listdir(os.path.join(ADE, "images", "validation")))
    if args.limit:
        imgs = imgs[: args.limit]
    done = set()
    if os.path.exists(OUT) and not args.limit:
        done = {(r["model"], r["run"]) for r in csv.DictReader(open(OUT))}
    base_macs = None
    for spec in args.runs.split(","):
        model, D = build(args.model)
        model = model.to(dev).eval()
        method = spec.split(":")[0]
        amount = float(spec.split(":")[1]) if ":" in spec else 0.0
        if method == "none":
            fwd = model
        elif method == "algm":
            algm.patch.algm_segmenter_patch(model.cpu(), list(ALGM_LAYERS[args.model]), trace_source=True)
            model.encoder.window_size = (2, 2)
            model.encoder.threshold = amount
            model = model.to(dev).eval()
            fwd = model
        else:
            loc = int(spec.split(":")[2]) if method == "expedite" else None
            fwd = Reduced(model, method, amount, loc).eval()
        if args.check:
            ref, _ = build(args.model)
            ref = ref.to(dev).eval()
            xc = torch.randn(2, 3, 512, 512, device=dev)
            with torch.no_grad():
                print("check", spec, "max |diff| vs released forward:", (ref(xc) - Reduced(model, method, 1.0 if method == "topk" else 0).eval()(xc)).abs().max().item(), flush=True)
            return
        counter = Counter(model, D)
        t0 = time.time()
        miou, macs, crops = evaluate(fwd, counter, imgs, dev, *CROP[args.model])
        if method == "none":
            base_macs = macs
        if base_macs is None:
            n = (CROP[args.model][0] // 16) ** 2
            base_macs = DEPTH[args.model] * (12 * (n + 1) * D ** 2 + 2 * (n + 1) ** 2 * D) + 2 * (12 * (n + 150) * D ** 2 + 2 * (n + 150) ** 2 * D)
        row = {"model": "segmenter_vit_" + args.model, "run": spec, "method": method, "amount": amount,
               "macs_frac": round(macs / base_macs, 3), "gmacs_per_crop": round(macs / 1e9, 2), "miou": round(miou, 2),
               "images": len(imgs), "crops": crops, "seconds": round(time.time() - t0, 1)}
        print(row, flush=True)
        if not args.limit and (row["model"], spec) not in done:
            new = not os.path.exists(OUT)
            with open(OUT, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)


if __name__ == "__main__":
    main()
