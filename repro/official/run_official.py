#!/usr/bin/env python3
"""Usage: python repro/official/run_official.py --method diffrate --setting 2.9 --data DIR [--limit N]"""
import argparse
import ast
import csv
import json
import math
import os
import sys
import time
import types

import torch
import timm
from timm.data import create_transform
from torchvision.datasets import ImageFolder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


SCRATCH = os.environ["VTR_WORK"]


MAE_MODELS = {"mae_vit_base_patch16": ("vit_base_patch16", "mae_finetuned_vit_base.pth"),
              "mae_vit_large_patch16": ("vit_large_patch16", "mae_finetuned_vit_large.pth"),
              "mae_vit_huge_patch14": ("vit_huge_patch14", "mae_finetuned_vit_huge.pth")}


def mae_model(args):
    add_path(os.path.join(args.repos, "PaPr", "ViT", "MAE"))
    import models_vit
    fn, ckpt = MAE_MODELS[args.model]
    model = getattr(models_vit, fn)(num_classes=1000, global_pool=True)
    state = torch.load(os.path.join(args.weights, "mae", ckpt), map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    return model


def timm_deit(args):
    if args.model in MAE_MODELS:
        return mae_model(args)
    model = timm.create_model(args.model, pretrained=False)
    model.load_state_dict(load_state(args))
    return model


def load_state(args):
    return torch.load(os.path.join(args.weights, args.model + ".pth"), map_location="cpu")


def load_into(model, args, allowed_missing=()):
    res = model.load_state_dict(load_state(args), strict=False)
    missing = [k for k in res.missing_keys if not any(a in k for a in allowed_missing)]
    if missing or res.unexpected_keys:
        raise RuntimeError(f"weights do not fit: missing {missing}, unexpected {res.unexpected_keys}")
    return res.missing_keys


def parse_at(setting):
    if "@" not in setting:
        return setting, []
    v, locs = setting.split("@")
    blocks = []
    for part in locs.split(","):
        a, _, b = part.partition("-")
        blocks += list(range(int(a), int(b or a) + 1))
    return v, blocks


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


N0_TOKENS = 197


def schedule_counts(blocks, amount, fuse, n0=N0_TOKENS, depth=12, keep=None):
    keep = fuse if keep is None else keep
    kind, v = amount
    out, n = [], n0
    for l in range(1, depth + 1):
        p = n - 1
        if l not in blocks or p < 2:
            out.append(0 if not keep else None)
            continue
        if keep:
            left = p - int(v) if kind == "n" else math.ceil(v * p)
            left = max(1, min(left, p))
            out.append(left if left < p else None)
            n = left + (2 if fuse else 1) if left < p else n
        else:
            r = int(v) if kind == "n" else int((1 - v) * p)
            r = max(0, min(r, p // 2))
            out.append(r)
            n -= r
    return out


DIMS = {"deit_tiny_patch16_224": dict(embed_dim=192, num_heads=3),
        "deit_small_patch16_224": dict(embed_dim=384, num_heads=6),
        "deit_base_patch16_224": dict(embed_dim=768, num_heads=12)}


def adapter_diffrate(args):
    add_path(os.path.join(args.repos, "DiffRate"))
    import DiffRate
    model = timm_deit(args)
    DiffRate.patch.deit(model, prune_granularity=1, merge_granularity=1)
    rates = json.load(open(os.path.join(args.repos, "DiffRate", "compression_rate.json")))
    key = {"deit_small_patch16_224": "ViT-S-DeiT", "deit_tiny_patch16_224": "ViT-T-DeiT",
           "deit_base_patch16_224": "ViT-B-DeiT"}[args.model]
    d = rates[key][args.setting]
    model.set_kept_num(eval(d["prune_kept_num"]), eval(d["merge_kept_num"]))
    return model, lambda x: model(x, return_flop=False), args.setting, ""


def adapter_tome(args):
    add_path(os.path.join(args.repos, "ToMe"))
    import tome
    model = timm_deit(args)
    setting, flag = (args.setting or "13"), ""


    if setting.rsplit(",", 1)[-1].strip() in ("noprop", "prop", "metric32"):
        setting, flag = setting.rsplit(",", 1)[0], setting.rsplit(",", 1)[1].strip()
    mae = args.model in MAE_MODELS

    prop = (flag == "prop") if mae else (flag != "noprop")
    if mae:
        tome.patch.mae(model, trace_source=False, prop_attn=prop)
    else:
        tome.patch.timm(model, trace_source=False, prop_attn=prop)
    if flag == "metric32":
        tpt = sys.modules["tome.patch.timm"]
        released = tpt.bipartite_soft_matching

        def matching32(metric, *a, **kw):
            with torch.autocast(device_type=metric.device.type, enabled=False):
                return released(metric.float(), *a, **kw)
        tpt.bipartite_soft_matching = matching32
    r, blocks = parse_at(setting)
    if r.startswith("rho="):
        rho = float(r[4:])
        model.r = schedule_counts(blocks, ("rho", rho), fuse=False, depth=len(model.blocks))
        return model, model, f"rho={rho} at blocks {','.join(map(str, blocks))}", "schedule grid"
    if blocks:
        model.r = [int(r) if i + 1 in blocks else 0 for i in range(len(model.blocks))]
        return model, model, f"r={int(r)} at blocks {','.join(map(str, blocks))}", "schedule ablation"
    model.r = int(r)
    label = f"r={model.r}" + ("" if prop else ", no proportional attention") + (", matching in fp32" if flag == "metric32" else "")
    if mae:
        label = f"r={model.r}, " + ("with" if prop else "without") + " proportional attention"
    return model, model, label, ""


def adapter_evit(args):
    return _evit(args, fuse=True)


def adapter_topk(args):
    return _evit(args, fuse=False)


def _evit(args, fuse):
    tag = ", fuse_token" if fuse else ", no fused token"
    add_path(os.path.join(args.repos, "evit"))
    import evit
    keep, blocks = parse_at(args.setting or "0.7")
    fn = evit.deit_base_patch16_shrink_base if args.model == "deit_base_patch16_224" \
        else evit.deit_small_patch16_shrink_base
    if keep.startswith("n="):

        n = int(keep[2:])
        tokens = schedule_counts(blocks, ("n", n), fuse=fuse, keep=True)
        model = fn(pretrained=False, base_keep_rate=1.0, drop_loc=(), fuse_token=fuse, **DIMS[args.model])
        load_into(model, args)
        return model, (lambda x: model(x, tokens=tokens)), \
            f"n={n} at blocks {','.join(map(str, blocks))}{tag}", "schedule grid"
    keep = float(keep)
    drop_loc = tuple(b - 1 for b in blocks) if blocks else (3, 6, 9)


    model = fn(pretrained=False, base_keep_rate=keep, drop_loc=drop_loc, fuse_token=fuse, **DIMS[args.model])
    load_into(model, args)
    if blocks:
        return model, model, f"keep={keep} at blocks {','.join(map(str, blocks))}{tag}", "schedule ablation"
    return model, model, f"keep={keep}{tag}", "no fine-tuning"


def adapter_dynamicvit(args):
    if args.model != "deit_small_patch16_224":
        raise ValueError("the released DynamicViT checkpoint used here is DeiT-S/0.7")
    add_path(os.path.join(args.repos, "DynamicViT"))
    sys.modules.setdefault("torch._six", types.SimpleNamespace(inf=math.inf))
    from models.dyvit import VisionTransformerDiffPruning
    r = float(args.setting or "0.7")
    model = VisionTransformerDiffPruning(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                         qkv_bias=True, pruning_loc=[3, 6, 9], token_ratio=[r, r ** 2, r ** 3])
    ckpt = torch.load(os.path.join(args.weights, "dynamicvit", "dynamicvit_deit-s_0.7.pth"), map_location="cpu",
                      weights_only=True)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    return model, model, f"keep={r}, released checkpoint", "fine-tuned (released)"


def adapter_ats(args):
    import yaml
    root = os.path.join(args.repos, "ATS")
    add_path(root)


    for name, sub in [("libs", "libs"), ("libs.models", "libs/models")]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(root, sub)]
            sys.modules[name] = m

    class Registry:
        def __init__(self, name):
            self.name = name

        def register(self, obj=None):
            return obj if obj is not None else (lambda o: o)

    for name in ["fvcore", "fvcore.common"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    reg = types.ModuleType("fvcore.common.registry")
    reg.Registry = Registry
    sys.modules["fvcore.common.registry"] = reg
    from libs.models.transformers.vit import ViT

    raw = yaml.safe_load(open(os.path.join(root, "configs", "ImageNet", "DeiT_S.yaml")))["VIT"]

    def decode(v):
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return v
        return v

    cfg = types.SimpleNamespace(VIT=types.SimpleNamespace(**{k: decode(v) for k, v in raw.items()}))
    label = "DeiT_S.yaml (blocks 4-12, K=197)"
    if args.setting:
        k = int(args.setting)
        cfg.VIT.NUM_TOKENS = [k if i in cfg.VIT.ATS_BLOCKS else n for i, n in enumerate(cfg.VIT.NUM_TOKENS)]
        label = f"DeiT_S.yaml with K={k} (blocks 4-12)"
    model = ViT(cfg)
    missing = load_into(model, args, allowed_missing=("out_zero_mask",))
    note = f"not loaded: {len(missing)} out_zero_mask buffers (zeros by design)"
    return model, lambda x: model(x)[0], label, note


def adapter_ppt(args):
    add_path(os.path.join(args.repos, "PPT"))
    from ppt_deit import apply_patch
    model = timm_deit(args)
    v, blocks = parse_at(args.setting or "50")
    if blocks:
        amount = ("rho", float(v[4:])) if v.startswith("rho=") else ("n", int(v))
        apply_patch(model, trace_source=False, prop_attn=True, pp_loc_list=[b - 1 for b in blocks], threshold=7e-5)
        removed = schedule_counts(blocks, amount, fuse=False)
        model.r = [removed[b - 1] for b in blocks]
        label = (f"rho={amount[1]}" if amount[0] == "rho" else f"r={amount[1]}") + \
            f" at blocks {','.join(map(str, blocks))}, tau=7e-5"
        return model, model, label, "schedule grid"
    apply_patch(model, trace_source=False, prop_attn=True, pp_loc_list=[3, 6, 9], threshold=7e-5)
    model.r = int(v)
    return model, model, f"r={model.r}, blocks 4/7/10, tau=7e-5", ""


def adapter_pitome(args):
    root = os.path.join(args.repos, "PiToMe")
    add_path(root)


    import algo
    for name, sub in [("algo.pitome", "algo/pitome"), ("algo.pitome.patch", "algo/pitome/patch")]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(root, sub)]
            sys.modules[name] = m
    from algo.pitome.patch.deit import apply_patch
    model = timm_deit(args)
    apply_patch(model)
    model.ratio = float(args.setting or 0.925)
    return model, lambda x: model(x, return_flop=False), f"ratio={model.ratio}", ""


def adapter_mctf(args):
    root = os.path.join(args.repos, "MCTF")
    add_path(root)
    import models.deit as mdeit
    v, blocks = parse_at(args.setting or "16")


    r = round(1.0 - float(v[4:]), 4) if v.startswith("rho=") else int(v)
    active = [b - 1 for b in blocks] if blocks else list(range(1, 12))
    fn = {"deit_tiny_patch16_224": mdeit.deit_tiny, "deit_small_patch16_224": mdeit.deit_small,
          "deit_base_patch16_224": mdeit.deit_base}[args.model]
    model = fn(pretrained=False, num_classes=1000, use_mctf=True,
                             mctf_type=[r, 0, 0, 1, 1, 1, 20, 40, 1, 1, 0], task_type=[1, 0, 0, 0],
                             activate_layer=active)
    load_into(model, args)
    model.cost_hook = "mctf"
    if blocks:
        label = (f"rho={float(v[4:])}" if v.startswith("rho=") else f"r={r}") + \
            f" at blocks {','.join(map(str, blocks))}, one-step-ahead"
        return model, lambda x: model(x, r_eval=r), label, "schedule grid"
    return model, lambda x: model(x, r_eval=r), f"r={r}, blocks 2-12, one-step-ahead", "no training"


def adapter_atc(args):
    add_path(os.path.join(args.repos, "ATC"))
    from ATC_setup.models.atc import AgglomerativeClusteringVisionTransformer
    from functools import partial
    keep, blocks = parse_at(args.setting or "0.7")
    if blocks:
        amount = ("rho", float(keep[4:])) if keep.startswith("rho=") else ("n", int(keep))
        removed = schedule_counts(blocks, amount, fuse=False)
        n, kept = N0_TOKENS, []
        for l in range(1, 13):
            if l in blocks:
                n -= removed[l - 1]
                kept.append(n - 1)
        margs = argparse.Namespace(reduction_ratio=kept, reduction_loc=[b - 1 for b in blocks], linkage="average",
                                   proportional_attn=True)
        label = (f"rho={amount[1]}" if amount[0] == "rho" else f"r={amount[1]}") + \
            f" at blocks {','.join(map(str, blocks))}, average linkage"
        note = "schedule grid"
    else:
        keep = float(keep)
        margs = argparse.Namespace(reduction_ratio=[keep], reduction_loc=[3, 6, 9], linkage="average",
                                   proportional_attn=True)
        label, note = f"keep={keep}, average linkage, blocks 4/7/10", "no fine-tuning"

    model = AgglomerativeClusteringVisionTransformer(
        patch_size=16, depth=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_classes=1000, args=margs, **DIMS[args.model])
    load_into(model, args)
    return model, model, label, note


def adapter_gtp(args):
    add_path(os.path.join(args.repos, "GTP-ViT"))
    import models_v3


    models_v3.open = lambda *a, **k: open(os.devnull, "a")
    parts = [float(v) for v in (args.setting or "11").split(",")]
    p = int(parts[0])
    theta = parts[1] if len(parts) > 1 else (0.7 if p <= 11 else 0.6)
    alpha = parts[2] if len(parts) > 2 else 0.3
    from functools import partial


    orig_to = torch.Tensor.to

    def to_cpu(self, *a, **k):
        if a and a[0] == "cuda":
            a = ("cpu",) + a[1:]
        return orig_to(self, *a, **k)

    if not args.device.startswith("cuda"):
        torch.Tensor.to = to_cpu
    try:

        model = models_v3.GraphPropagationTransformer(
            patch_size=16, depth=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_classes=1000, selection="MixedAttnMax", propagation="GraphProp", num_prop=p, num_neighbours=8,
            sparsity=theta, alpha=alpha, token_scale=True, graph_type="Mixed", **DIMS[args.model])
    finally:
        torch.Tensor.to = orig_to
    load_into(model, args)


    return model, model, f"P={p}, theta={theta}, alpha={alpha}, Mixed graph M=8, MixedAttnMax", ""


def adapter_reprshift(args):
    root = os.path.join(args.repos, "Representation-Shift")
    add_path(root)


    if "flash_attn.flash_attn_triton" not in sys.modules:
        fa = types.ModuleType("flash_attn")
        fat = types.ModuleType("flash_attn.flash_attn_triton")

        def flash_attn_func(*a, **k):
            raise RuntimeError("FlashAttention is not available on the CPU")

        fat.flash_attn_func = flash_attn_func
        fa.flash_attn_triton = fat
        sys.modules["flash_attn"] = fa
        sys.modules["flash_attn.flash_attn_triton"] = fat
    from models.deit import VisionTransformer
    from functools import partial
    v, blocks = parse_at(args.setting or "0.2")
    if blocks:

        amount = ("rho", float(v[4:])) if v.startswith("rho=") else ("n", int(v))
        drop_r = schedule_counts(blocks, amount, fuse=False)
        rho = None
    else:
        rho = float(v)
        drop_r = [rho if i in (0, 3, 6) else 0 for i in range(12)]


    model = VisionTransformer(patch_size=16, depth=12, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                              num_classes=1000, token_nums=[1, 0, 0], use_flash=True, drop_r=drop_r,
                              **DIMS[args.model])
    load_into(model, args)

    if rho is None:
        label = (f"rho={amount[1]}" if amount[0] == "rho" else f"r={amount[1]}") + \
            f" after blocks {','.join(map(str, blocks))}, MLP shift (L2)"
        return model, model, label, "schedule grid"
    return model, model, f"drop {rho:g} after blocks 1/4/7, MLP shift (L2)", ""


def adapter_papr(args):
    add_path(os.path.join(args.repos, "PaPr", "ViT", "AugReg"))
    import PaPr
    z = float(args.setting or 0.55)
    model = timm_deit(args)
    PaPr.create_class_token_ViT_model(model, proposal_model="mobileone_s0", z=z, input_size=224,
                                      proposal_weights=os.path.join(args.weights, "mobileone", "mobileone_s0.pth.tar"))


    return model, model, f"z={z}, MobileOne-S0 proposal", "GMACs include the MobileOne-S0 proposal"


def adapter_none(args):
    model = timm_deit(args)
    return model, model, "no reduction", ""


ADAPTERS = {"none": adapter_none, "diffrate": adapter_diffrate, "tome": adapter_tome, "evit": adapter_evit, "topk": adapter_topk, "dynamicvit": adapter_dynamicvit, "ats": adapter_ats,
            "ppt": adapter_ppt, "pitome": adapter_pitome, "mctf": adapter_mctf, "atc": adapter_atc,
            "gtp": adapter_gtp, "reprshift": adapter_reprshift, "papr": adapter_papr}


def add_cost_hooks(model, cost):
    d = model.embed_dim
    hidden = model.blocks[0].mlp.fc1.out_features

    def tokens(args, kwargs):
        x = args[0] if args else kwargs["x"]
        pol = kwargs.get("policy")
        if pol is not None:
            return pol.reshape(pol.shape[0], -1).sum(1).double()
        return torch.full((x.shape[0],), float(x.shape[1]), dtype=torch.float64)

    def hook_block(mod, args, kwargs):
        n = tokens(args, kwargs)
        cost["macs"] += (4 * n * d * d + 2 * n * n * d).mean().item()

    def hook_mctf_attn(mod, args, kwargs):


        n = float(args[0].shape[1])
        a = kwargs.get("_attn", False)
        if isinstance(a, bool) and not a:
            cost["macs"] += 4 * n * d * d + 2 * n * n * d
        else:
            cost["macs"] += 2 * n * d * d + n * n * d

    def hook_mlp(mod, args, kwargs):
        cost["macs"] += (2 * tokens(args, kwargs) * d * hidden).mean().item()

    for blk in model.blocks:
        if getattr(model, "cost_hook", "") == "mctf":
            blk.attn.register_forward_pre_hook(hook_mctf_attn, with_kwargs=True)
        else:
            blk.register_forward_pre_hook(hook_block, with_kwargs=True)
        blk.mlp.register_forward_pre_hook(hook_mlp, with_kwargs=True)

    def hook_conv(mod, inp, out):
        k = mod.weight[0].numel()
        cost["macs"] += out[0].numel() * k


    for m in (model.proposal.modules() if hasattr(model, "proposal") else []):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            m.register_forward_hook(hook_conv)
    p = model.patch_embed.patch_size
    p = p[0] if isinstance(p, (tuple, list)) else p
    return model.patch_embed.num_patches * 3 * p * p * d + d * model.head.out_features


def device_name(dev):
    return torch.cuda.get_device_name(dev) if dev.type == "cuda" else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--setting", default="")
    ap.add_argument("--model", default="deit_small_patch16_224")
    ap.add_argument("--weights", default=os.path.join(SCRATCH, "weights"))
    ap.add_argument("--repos", default=os.path.join(SCRATCH, "official"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--latency", type=int, default=0,
                    help="instead of accuracy, latency at batch size 1: median over this many images")
    ap.add_argument("--throughput", type=int, default=0,
                    help="instead of accuracy, images per second and peak GPU memory at this batch size")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (cuda:0, ...)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="autocast of the forward on the GPU (Section 9.9)")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True), to check run-to-run variation (Section 9.9)")
    ap.add_argument("--save-preds", default="", help="save the predicted class of every image to this .npy file")
    ap.add_argument("--save-logits", default="", help="save the fp32 logits of every image to this .npy file")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    model, forward, setting, note = ADAPTERS[args.method](args)
    model.eval().to(args.device)
    dev = torch.device(args.device)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)


    tf = create_transform(input_size=(3, 224, 224), interpolation="bicubic", crop_pct=0.875 if args.model in MAE_MODELS else 0.9,
                          mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    cost = {"macs": 0.0}

    const = 0 if (args.latency or args.throughput) else add_cost_hooks(model, cost)

    ds = ImageFolder(args.data, transform=tf)
    wnids = [l.strip() for l in open(os.path.join(HERE, "imagenet_wnids.txt"))]

    tmap = torch.tensor([int(c) if c.isdigit() else wnids.index(c) for c in ds.classes])
    if args.limit:
        g = torch.Generator().manual_seed(0)
        ds = torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=g)[: args.limit].tolist())
    if args.latency:
        imgs = [ds[i][0].unsqueeze(0).to(dev) for i in range(args.latency)]
        times = []
        with torch.no_grad():
            for img in imgs[:10]:
                forward(img)
            for _ in range(3):
                for img in imgs:
                    sync()
                    t0 = time.perf_counter()
                    forward(img)
                    sync()
                    times.append(time.perf_counter() - t0)
        times.sort()
        row = {"method": args.method + (" (official code)" if args.method != "none" else ""), "setting": setting,
               "ms_batch1": round(1000 * times[len(times) // 2], 2), "threads": args.threads,
               "runs": len(times), "note": "; ".join(s for s in [args.model if args.model != "deit_small_patch16_224" else "",
                                                               "timm " + timm.__version__, device_name(dev)] if s)}
        print(row, flush=True)
        if args.out:
            new = not os.path.exists(args.out)
            with open(args.out, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        return
    if args.throughput:

        batch = torch.stack([ds[i][0] for i in range(args.throughput)]).to(dev)
        with torch.no_grad():
            for _ in range(5):
                forward(batch)
            sync()
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            for _ in range(20):
                forward(batch)
            sync()
            elapsed = time.perf_counter() - t0
        row = {"method": args.method + (" (official code)" if args.method != "none" else ""), "setting": setting,
               "batch": args.throughput, "img_per_s": round(20 * args.throughput / elapsed, 1),
               "peak_mb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 20) if dev.type == "cuda" else "",
               "threads": args.threads if dev.type == "cpu" else "",
               "note": "; ".join(s for s in [args.model if args.model != "deit_small_patch16_224" else "",
                                             "timm " + timm.__version__, device_name(dev)] if s)}
        print(row, flush=True)
        if args.out:
            new = not os.path.exists(args.out)
            with open(args.out, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        return
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
    correct = total = 0
    macs = 0.0
    preds, logits = [], []
    dtype ={"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dtype, enabled=dtype is not None):
        for img, y in dl:
            cost["macs"] = 0.0
            out = forward(img.to(dev))
            pred = out.argmax(-1).cpu()
            preds.append(pred)
            if args.save_logits:
                logits.append(out.float().cpu())
            correct += (pred == tmap[y]).sum().item()
            total += y.numel()
            macs += (const + cost["macs"]) * y.numel()
    if args.save_preds:
        import numpy as np
        np.save(args.save_preds, torch.cat(preds).numpy())
    if args.save_logits:
        import numpy as np
        np.save(args.save_logits, torch.cat(logits).numpy())
    extra =[args.dtype if args.dtype != "fp32" else "", "deterministic" if args.deterministic else ""]
    row = {"method": args.method + " (official code)", "setting": setting, "top1": round(100.0 * correct / total, 2),
           "gmacs": round(macs / total / 1e9, 3), "images": total, "seconds": round(time.time() - t0, 1),
           "note": "; ".join(s for s in [args.model if args.model != "deit_small_patch16_224" else "",
                                         "timm " + timm.__version__, device_name(dev), note] + extra if s)}
    print(row, flush=True)
    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
