from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Ctx:
    layer: int = 0
    attn: Optional[torch.Tensor] = None
    keys: Optional[torch.Tensor] = None
    attn_out: Optional[torch.Tensor] = None
    size: Optional[torch.Tensor] = None
    prop_attn: bool = True
    need_attn: bool = True
    tokens_in: List[float] = field(default_factory=list)
    tokens_in_sq: List[float] = field(default_factory=list)
    tokens_mlp: List[float] = field(default_factory=list)
    extra_macs: float = 0.0
    stash: dict = field(default_factory=dict)
    key_mask: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    source: Optional[torch.Tensor] = None


def _qkv(attn_mod: nn.Module, x: torch.Tensor):
    B, N, C = x.shape
    h = attn_mod.num_heads
    if getattr(attn_mod, "qkv", None) is not None:
        hd = getattr(attn_mod, "head_dim", None) or attn_mod.qkv.out_features // (3 * h)
        qb = getattr(attn_mod, "q_bias", None)
        if qb is not None:
            bias = torch.cat((qb, torch.zeros_like(qb), attn_mod.v_bias))
            if getattr(attn_mod, "qkv_bias_separate", False):
                qkv = attn_mod.qkv(x) + bias
            else:
                qkv = F.linear(x, attn_mod.qkv.weight, bias)
        else:
            qkv = attn_mod.qkv(x)
        q, k, v = qkv.reshape(B, N, 3, h, hd).permute(2, 0, 3, 1, 4).unbind(0)
    else:
        q = attn_mod.q_proj(x).reshape(B, N, h, -1).transpose(1, 2)
        k = attn_mod.k_proj(x).reshape(B, N, h, -1).transpose(1, 2)
        v = attn_mod.v_proj(x).reshape(B, N, h, -1).transpose(1, 2)
    qn, kn = getattr(attn_mod, "q_norm", None), getattr(attn_mod, "k_norm", None)
    if qn is not None:
        q = qn(q)
    if kn is not None:
        k = kn(k)
    return q, k, v


def attention_forward(attn_mod: nn.Module, x: torch.Tensor, ctx: Ctx, bias: Optional[torch.Tensor] = None,
                      rope: Optional[torch.Tensor] = None, rel=None) -> torch.Tensor:
    B, N, C = x.shape
    q, k, v = _qkv(attn_mod, x)
    if rope is not None:
        from timm.layers import apply_rot_embed_cat
        npt = getattr(attn_mod, "num_prefix_tokens", 1)
        half = getattr(attn_mod, "rotate_half", False)
        q = torch.cat([q[:, :, :npt], apply_rot_embed_cat(q[:, :, npt:], rope, half=half)], dim=2).type_as(v)
        k = torch.cat([k[:, :, :npt], apply_rot_embed_cat(k[:, :, npt:], rope, half=half)], dim=2).type_as(v)
    logits = (q * attn_mod.scale) @ k.transpose(-2, -1)
    if bias is not None:
        logits = logits + bias
    if rel is not None:
        logits = logits + rel.logits(q) * attn_mod.scale
    if ctx.prop_attn and ctx.size is not None:
        logits = logits + ctx.size.log()[:, None, None, :, 0]
    a = logits.softmax(dim=-1)
    if ctx.key_mask is not None:

        M = ctx.key_mask[:, None, None, :].expand(B, 1, N, N).clone()
        eye = torch.eye(N, device=x.device, dtype=torch.bool)[None, None]
        M = torch.where(eye, torch.ones_like(M), M)
        a = a * M
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    out = attn_mod.attn_drop(a) @ v
    if rel is not None:
        out = out + rel.values(a)
    out = out.transpose(1, 2).reshape(B, N, -1)
    norm = getattr(attn_mod, "norm", None)
    if norm is not None:
        out = norm(out)
    gate = getattr(attn_mod, "gate", None)
    if gate is not None:
        out = out * gate(x).sigmoid()
    out = attn_mod.proj_drop(attn_mod.proj(out))
    ctx.attn = a if ctx.need_attn else None
    ctx.keys = k.mean(1)
    ctx.keys_cat = k.transpose(1, 2).reshape(B, N, -1)
    ctx.values = v
    ctx.attn_out = out
    return out


def _valid_counts(x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
    if ctx.size is None or ctx.size.shape[1] != x.shape[1]:
        return torch.full((x.shape[0],), float(x.shape[1]))
    return (ctx.size[..., 0] > 1e-3).sum(1).float()


def _scaled(blk: nn.Module, ls_name: str, gamma_name: str, t: torch.Tensor) -> torch.Tensor:
    ls = getattr(blk, ls_name, None)
    if ls is not None:
        return ls(t)
    g = getattr(blk, gamma_name, None)
    if g is not None:
        return g * t
    return t


def _drop_path(blk: nn.Module, name: str):
    dp = getattr(blk, name, None)
    return dp if dp is not None else getattr(blk, "drop_path", nn.Identity())


def block_forward(blk: nn.Module, x: torch.Tensor, ctx: Ctx, reducer: Optional[Callable],
                  bias: Optional[torch.Tensor] = None, rope: Optional[torch.Tensor] = None, rel=None) -> torch.Tensor:
    n = _valid_counts(x, ctx)
    ctx.tokens_in.append(n.mean().item())
    ctx.tokens_in_sq.append((n ** 2).mean().item())
    skip = getattr(blk, "skip_lam", 1.0)
    x = x + _drop_path(blk, "drop_path1")(_scaled(blk, "ls1", "gamma_1", attention_forward(blk.attn, blk.norm1(x), ctx, bias, rope, rel))) / skip
    after_mlp = getattr(reducer, "after_mlp", False)
    if reducer is not None and not after_mlp:
        x = reducer(x, ctx)
    ctx.tokens_mlp.append(_valid_counts(x, ctx).mean().item())
    x = x + _drop_path(blk, "drop_path2")(_scaled(blk, "ls2", "gamma_2", blk.mlp(blk.norm2(x)))) / skip
    if reducer is not None and after_mlp:

        x = reducer(x, ctx)
    ctx.layer += 1
    return x


class ReducedViT(nn.Module):

    def __init__(self, vit: nn.Module, reducer: Optional[Callable] = None, prop_attn: bool = True):
        super().__init__()
        self.vit = vit
        self.reducer = reducer
        self.prop_attn = prop_attn
        self.last_ctx: Optional[Ctx] = None

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        v = self.vit
        x = v.patch_embed(img)
        x = v._pos_embed(x)
        x = v.patch_drop(x)
        x = v.norm_pre(x)
        B, N, _ = x.shape
        ctx = Ctx(size=torch.ones(B, N, 1, device=x.device, dtype=x.dtype), prop_attn=self.prop_attn)
        for blk in v.blocks:
            x = block_forward(blk, x, ctx, self.reducer)
        x = v.norm(x)
        self.last_ctx = ctx
        return v.forward_head(x)


def macs_per_image(vit: nn.Module, ctx: Ctx) -> float:
    d = vit.embed_dim
    hidden = vit.blocks[0].mlp.fc1.out_features
    p = vit.patch_embed.patch_size[0]
    n_patch = vit.patch_embed.num_patches
    macs = n_patch * (3 * p * p) * d
    for n_in, n_sq, n_mlp in zip(ctx.tokens_in, ctx.tokens_in_sq, ctx.tokens_mlp):
        macs += 4 * n_in * d * d + 2 * n_sq * d
        macs += 2 * n_mlp * d * hidden
    macs += d * vit.head.out_features
    return macs + ctx.extra_macs
