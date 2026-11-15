from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform, resolve_data_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokred.vit import Ctx, block_forward

WORK = os.environ.get("VTR_WORK", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "work"))
WEIGHTS = os.path.join(WORK, "weights", "backbones")
OFFICIAL = os.path.join(WORK, "official")


@dataclass
class Spec:
    name: str
    label: str
    net: nn.Module
    blocks: List[nn.Module]
    embed: Callable
    head: Callable
    transform: object
    embed_dim: int
    num_patches: int
    collect: List[int] = field(default_factory=list)
    attn_extras: Optional[Callable] = None
    note: str = ""
    extra_prefix: int = 0


def _size_weighted_mean(x, ctx):
    s = ctx.size[:, 1:]
    return (x[:, 1:] * s).sum(1) / s.sum(1)


def _timm_vit(model_name, label, note="", **kw):
    v = timm.create_model(model_name, pretrained=True, **kw).eval()
    tf = create_transform(**resolve_data_config({}, model=v))

    def embed(img):
        x = v.patch_embed(img)
        x = v._pos_embed(x)
        x = v.patch_drop(x)
        return v.norm_pre(x)

    def head(x, ctx, feats):
        x = v.norm(x)
        if v.global_pool == "avg":
            x = _size_weighted_mean(x, ctx)
            return v.head(v.head_drop(v.fc_norm(x)))
        return v.forward_head(x)

    return Spec(model_name, label, v, list(v.blocks), embed, head, tf, v.embed_dim, v.patch_embed.num_patches, note=note)


def _timm_vit_distilled(model_name, label, note=""):
    v = timm.create_model(model_name, pretrained=True).eval()
    tf = create_transform(**resolve_data_config({}, model=v))

    def embed(img):
        x = v.patch_embed(img)
        x = v._pos_embed(x)
        x = v.patch_drop(x)
        return v.norm_pre(x)

    def head(x, ctx, feats):
        x = v.norm(x)
        j = (ctx.pos == 1).float().argmax(1)
        x_dist = x[torch.arange(x.shape[0], device=x.device), j]
        return (v.head(x[:, 0]) + v.head_dist(x_dist)) / 2

    return Spec(model_name, label, v, list(v.blocks), embed, head, tf, v.embed_dim, v.patch_embed.num_patches, note=note,
                extra_prefix=1)


def _dino(model_name, label, head_file, n_last, avgpool):
    v = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    sd = torch.load(os.path.join(WEIGHTS, head_file), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    lin = nn.Linear(sd["module.linear.weight"].shape[1], 1000)
    lin.load_state_dict({"weight": sd["module.linear.weight"], "bias": sd["module.linear.bias"]})
    v.linear_head = lin.eval()
    tf = create_transform(**resolve_data_config({}, model=v))
    L = len(v.blocks)

    def embed(img):
        x = v.patch_embed(img)
        x = v._pos_embed(x)
        return v.norm_pre(x)

    def head(x, ctx, feats):
        x = v.norm(x)
        cls = torch.cat(feats + [x[:, 0]], -1) if n_last > 1 else x[:, 0]
        if avgpool:
            cls = torch.stack([cls, _size_weighted_mean(x, ctx)], -1).reshape(x.shape[0], -1)
        return v.linear_head(cls)

    return Spec(model_name, label, v, list(v.blocks), embed, head, tf, v.embed_dim, v.patch_embed.num_patches,
                collect=list(range(L - n_last, L - 1)), note="official linear head")


def _dinov2(model_name, label, head_file):
    v = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=224).eval()
    sd = torch.load(os.path.join(WEIGHTS, head_file), map_location="cpu", weights_only=False)
    lin = nn.Linear(sd["weight"].shape[1], 1000)
    lin.load_state_dict({"weight": sd["weight"], "bias": sd["bias"]})
    v.linear_head = lin.eval()
    tf = create_transform(input_size=224, crop_pct=224 / 256, interpolation="bicubic",
                          mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def embed(img):
        x = v.patch_embed(img)
        x = v._pos_embed(x)
        return v.norm_pre(x)

    def head(x, ctx, feats):
        x = v.norm(x)
        return v.linear_head(torch.cat([x[:, 0], _size_weighted_mean(x, ctx)], -1))

    return Spec(model_name, label, v, list(v.blocks), embed, head, tf, v.embed_dim, v.patch_embed.num_patches,
                note="official linear head, 224 px")


def _beit(model_name, label):
    v = timm.create_model(model_name, pretrained=True).eval()
    tf = create_transform(**resolve_data_config({}, model=v))

    def embed(img):
        x = v.patch_embed(img)
        x = torch.cat((v.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if v.pos_embed is not None:
            x = x + v.pos_embed
        return v.pos_drop(x)

    def attn_extras(i, ctx):
        attn = v.blocks[i].attn
        bias = attn._get_rel_pos_bias()
        if v.rel_pos_bias is not None:
            bias = bias + v.rel_pos_bias()
        pos = ctx.pos
        b = bias[0][:, pos[:, :, None], pos[:, None, :]]
        return {"bias": b.permute(1, 0, 2, 3)}

    def head(x, ctx, feats):
        x = v.norm(x)
        x = _size_weighted_mean(x, ctx) if v.global_pool == "avg" else x[:, 0]
        return v.head(v.head_drop(v.fc_norm(x)))

    return Spec(model_name, label, v, list(v.blocks), embed, head, tf, v.embed_dim, v.patch_embed.num_patches,
                attn_extras=attn_extras)


def _eva(model_name, label):
    v = timm.create_model(model_name, pretrained=True).eval()
    tf = create_transform(**resolve_data_config({}, model=v))
    state = {}

    def embed(img):
        x = v.patch_embed(img)
        x, rope = v._pos_embed(x)
        state["rope"] = rope
        return v.norm_pre(x)

    def attn_extras(i, ctx):
        rope = state["rope"]
        if rope.dim() == 3:
            rope = rope[0]
        npt = v.num_prefix_tokens
        idx = (ctx.pos[:, npt:] - npt).clamp_min(0)
        return {"rope": rope[idx][:, None]}

    def head(x, ctx, feats):
        x = v.norm(x)
        x = _size_weighted_mean(x, ctx) if v.global_pool == "avg" else x[:, 0]
        return v.head(v.head_drop(v.fc_norm(x)))

    return Spec(model_name, label, v, list(v.blocks), embed, head, tf, v.embed_dim, v.patch_embed.num_patches,
                attn_extras=attn_extras)


def _t2t(label="T2T-ViT-14"):
    sys.path.insert(0, os.path.join(OFFICIAL, "T2T-ViT"))
    from models.t2t_vit import T2T_ViT
    v = T2T_ViT(tokens_type="performer", embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.)
    sd = torch.load(os.path.join(WEIGHTS, "81.5_T2T_ViT_14.pth.tar"), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict_ema", sd.get("state_dict", sd))
    v.load_state_dict(sd)
    v.eval()
    tf = create_transform(input_size=224, crop_pct=0.9, interpolation="bicubic", mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def embed(img):
        x = v.tokens_to_token(img)
        x = torch.cat((v.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return v.pos_drop(x + v.pos_embed)

    def head(x, ctx, feats):
        return v.head(v.norm(x)[:, 0])

    return Spec("t2t_vit_14", label, v, list(v.blocks), embed, head, tf, 384, v.tokens_to_token.num_patches,
                note="official checkpoint 81.5")


def _lvvit(variant, label):
    sys.path.insert(0, os.path.join(OFFICIAL, "TokenLabeling"))
    from tlt.models.lvvit import LV_ViT
    if variant == "s":
        v = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3., p_emb="4_2", skip_lam=2., return_dense=True, mix_token=True)
        ck = "lvvit_s-26M-224-83.3.pth.tar"
    else:
        v = LV_ViT(patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3., p_emb="4_2", skip_lam=2., return_dense=True, mix_token=True)
        ck = "lvvit_m-56M-224-84.0.pth.tar"
    sd = torch.load(os.path.join(WEIGHTS, ck), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict_ema", sd.get("state_dict", sd))
    v.load_state_dict(sd)
    v.eval()
    tf = create_transform(input_size=224, crop_pct=0.9, interpolation="bicubic", mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def embed(img):
        x = v.forward_embeddings(img).flatten(2).transpose(1, 2)
        x = torch.cat((v.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return v.pos_drop(x + v.pos_embed)

    def head(x, ctx, feats):
        x = v.norm(x)
        return v.head(x[:, 0]) + 0.5 * v.aux_head(x[:, 1:]).max(1)[0]

    return Spec("lvvit_" + variant, label, v, list(v.blocks), embed, head, tf, v.embed_dim, v.pos_embed.shape[1] - 1,
                note="official checkpoint")


BACKBONES = {
    "deit_s": lambda: _timm_vit("deit_small_patch16_224.fb_in1k", "DeiT-S"),
    "deit_b": lambda: _timm_vit("deit_base_patch16_224.fb_in1k", "DeiT-B"),
    "deit3_s": lambda: _timm_vit("deit3_small_patch16_224.fb_in1k", "DeiT III-S"),
    "deit3_b": lambda: _timm_vit("deit3_base_patch16_224.fb_in1k", "DeiT III-B"),
    "deit3_s_384": lambda: _timm_vit("deit3_small_patch16_384.fb_in1k", "DeiT III-S (384)"),
    "augreg_s": lambda: _timm_vit("vit_small_patch16_224.augreg_in21k_ft_in1k", "AugReg ViT-S"),
    "augreg_b": lambda: _timm_vit("vit_base_patch16_224.augreg_in21k_ft_in1k", "AugReg ViT-B"),
    "vit_b_orig": lambda: _timm_vit("vit_base_patch16_224.orig_in21k_ft_in1k", "ViT-B/16 (original)"),
    "clip_b_openai": lambda: _timm_vit("vit_base_patch16_clip_224.openai_ft_in1k", "CLIP ViT-B (OpenAI, ft)"),
    "clip_b_laion": lambda: _timm_vit("vit_base_patch16_clip_224.laion2b_ft_in1k", "CLIP ViT-B (LAION-2B, ft)"),
    "beit_b": lambda: _beit("beit_base_patch16_224.in22k_ft_in22k_in1k", "BEiT-B"),
    "eva02_s": lambda: _eva("eva02_small_patch14_336.mim_in22k_ft_in1k", "EVA-02-S (336)"),
    "eva02_b": lambda: _eva("eva02_base_patch14_448.mim_in22k_ft_in1k", "EVA-02-B (448)"),
    "dino_s": lambda: _dino("vit_small_patch16_224.dino", "DINO ViT-S (linear)", "dino_deitsmall16_linearweights.pth", 4, False),
    "dino_b": lambda: _dino("vit_base_patch16_224.dino", "DINO ViT-B (linear)", "dino_vitbase16_linearweights.pth", 1, True),
    "dinov2_s": lambda: _dinov2("vit_small_patch14_dinov2.lvd142m", "DINOv2 ViT-S (linear)", "dinov2_vits14_linear_head.pth"),
    "dinov2_b": lambda: _dinov2("vit_base_patch14_dinov2.lvd142m", "DINOv2 ViT-B (linear)", "dinov2_vitb14_linear_head.pth"),
    "t2t_14": lambda: _t2t(),
    "lvvit_s": lambda: _lvvit("s", "LV-ViT-S"),
    "lvvit_m": lambda: _lvvit("m", "LV-ViT-M"),

    "deit_s_dist": lambda: _timm_vit_distilled("deit_small_distilled_patch16_224.fb_in1k", "DeiT-S (distilled)"),
    "deit_b_dist": lambda: _timm_vit_distilled("deit_base_distilled_patch16_224.fb_in1k", "DeiT-B (distilled)"),
    "autoformer_s": lambda: __import__("tokred.autoformer", fromlist=["spec"]).spec("small", "AutoFormer-S"),
    "autoformer_b": lambda: __import__("tokred.autoformer", fromlist=["spec"]).spec("base", "AutoFormer-B"),
    "nvit_s": lambda: __import__("tokred.nvit", fromlist=["spec"]).spec("s", "NViT-S"),
    "nvit_b": lambda: __import__("tokred.nvit", fromlist=["spec"]).spec("b", "NViT-B"),
}


class ReducedBackbone(nn.Module):

    def __init__(self, spec: Spec, reducer=None, prop_attn: bool = True):
        super().__init__()
        self.spec = spec
        self.net = spec.net
        self.reducer = reducer
        self.prop_attn = prop_attn
        self.last_ctx: Optional[Ctx] = None

    def forward(self, img):
        s = self.spec
        x = s.embed(img)
        B, N, _ = x.shape
        ctx = Ctx(size=torch.ones(B, N, 1, device=x.device, dtype=x.dtype), prop_attn=self.prop_attn,
                  pos=torch.arange(N, device=x.device).expand(B, N).contiguous())
        ctx.stash["depth"] = len(s.blocks)
        if s.extra_prefix:
            ctx.stash["extra_prefix"] = s.extra_prefix
        feats = []
        for i, blk in enumerate(s.blocks):
            extras = s.attn_extras(i, ctx) if s.attn_extras is not None else {}
            x = block_forward(blk, x, ctx, self.reducer, bias=extras.get("bias"), rope=extras.get("rope"), rel=extras.get("rel"))
            if i in s.collect:
                feats.append(s.net.norm(x)[:, 0])
        self.last_ctx = ctx
        return s.head(x, ctx, feats)


def block_macs_per_token(spec: Spec):
    blk = spec.blocks[0]
    lin = lambda mod: sum(m.in_features * m.out_features for m in mod.modules() if isinstance(m, nn.Linear))
    q, k, v = None, None, None
    h = blk.attn.num_heads
    hd = getattr(blk.attn, "head_dim", None) or spec.embed_dim // h
    return lin(blk.attn), lin(blk.mlp), h * hd


def macs_of_counts(spec: Spec, tokens_in, tokens_in_sq, tokens_mlp):
    a_lin, mlp, hhd = block_macs_per_token(spec)
    total = 0.0
    for n_in, n_sq, n_mlp in zip(tokens_in, tokens_in_sq, tokens_mlp):
        total += a_lin * n_in + 2 * n_sq * hhd + mlp * n_mlp
    return total
