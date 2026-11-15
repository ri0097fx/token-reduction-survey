from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .vit import Ctx

DEFAULT_LOCS = (3, 6, 9)


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _cls_attn(ctx: Ctx) -> torch.Tensor:
    return ctx.attn[:, :, 0, 1:].mean(1)


def _keep_by_score(x, ctx, score, n_keep):
    idx = score.topk(n_keep, dim=1).indices + 1
    idx, _ = idx.sort(dim=1)
    cls = torch.zeros(x.shape[0], 1, dtype=idx.dtype, device=idx.device)
    full = torch.cat([cls, idx], 1)
    return full


def _apply_index(x, ctx, full_idx):
    ctx.size = _gather(ctx.size, full_idx)
    if ctx.pos is not None:
        ctx.pos = ctx.pos.gather(1, full_idx)
    if ctx.source is not None:
        ctx.source = _gather(ctx.source, full_idx)
    return _gather(x, full_idx)


def _n_keep(n_patch, keep_rate):
    return max(1, int(math.ceil(keep_rate * n_patch)))


def topk(keep_rate: float = 0.7, locs: Sequence[int] = DEFAULT_LOCS, fuse: bool = False, n_remove: int = 0):

    def f(x, ctx: Ctx):
        if ctx.layer not in locs:
            return x
        score = _cls_attn(ctx)

        n_patch = x.shape[1] - 1
        k = max(1, n_patch - n_remove) if n_remove else _n_keep(n_patch, keep_rate)
        idx = score.topk(n_patch, dim=1).indices + 1
        keep, drop = idx[:, :k], idx[:, k:]
        keep_sorted, _ = keep.sort(1)
        cls = torch.zeros(x.shape[0], 1, dtype=idx.dtype, device=idx.device)
        keep_idx = torch.cat([cls, keep_sorted], 1)
        out = _gather(x, keep_idx)
        size = _gather(ctx.size, keep_idx)
        pos = ctx.pos.gather(1, keep_idx) if ctx.pos is not None else None
        src = _gather(ctx.source, keep_idx) if ctx.source is not None else None
        if fuse and drop.shape[1] > 0:
            w = score.gather(1, drop - 1).unsqueeze(-1)
            fused = (_gather(x, drop) * w).sum(1, keepdim=True)
            out = torch.cat([out, fused], 1)
            size = torch.cat([size, _gather(ctx.size, drop).sum(1, keepdim=True)], 1)
            if pos is not None:
                pos = torch.cat([pos, ctx.pos.gather(1, drop[:, :1])], 1)
            if src is not None:
                src = torch.cat([src, _gather(ctx.source, drop).sum(1, keepdim=True)], 1)
            ctx.stash["has_fused"] = True
        ctx.size = size
        ctx.pos = pos
        ctx.source = src
        return out

    return f


def ats(keep_rate: float = 0.7, locs: Sequence[int] = DEFAULT_LOCS):

    def f(x, ctx: Ctx):
        if ctx.layer not in locs:
            return x
        B, N, _ = x.shape
        a = ctx.attn[:, :, 0, 1:]
        vnorm = ctx.values[:, :, 1:].norm(dim=-1)
        s = (a * vnorm).sum(1)
        s = s / s.sum(1, keepdim=True)
        K = _n_keep(N - 1, keep_rate)
        order = s.argsort(1)
        cdf = s.gather(1, order).cumsum(1)
        grid = (torch.arange(K, device=x.device, dtype=x.dtype) + 0.5) / K
        pos = torch.searchsorted(cdf, grid.expand(B, -1).contiguous()).clamp(max=N - 2)
        chosen = order.gather(1, pos)
        mask = torch.zeros(B, N - 1, dtype=torch.bool, device=x.device)
        mask.scatter_(1, chosen, True)
        n_unique = mask.sum(1)
        kmax = int(n_unique.max())

        rank_score = s + mask.to(s.dtype) * 2.0
        idx = rank_score.topk(kmax, dim=1).indices
        valid = mask.gather(1, idx)
        idx_sorted, perm = idx.sort(1)
        valid = valid.gather(1, perm)
        cls = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
        full = torch.cat([cls, idx_sorted + 1], 1)
        out = _gather(x, full)
        size = _gather(ctx.size, full)
        size[:, 1:] = size[:, 1:] * valid.unsqueeze(-1).to(size.dtype) + (~valid).unsqueeze(-1).to(size.dtype) * 1e-6
        ctx.size = size
        ctx.stash.setdefault("ats_kept", []).append(n_unique.float().mean().item())
        return out

    return f


def _official_tps_module():
    import importlib.util
    import os
    import sys
    import types
    root = os.path.join(os.environ["VTR_WORK"], "official", "TPS-CVPR2023", "torch_codebase", "models")
    if "tps_models.dynamic_vit" not in sys.modules:
        pkg = types.ModuleType("tps_models")
        pkg.__path__ = [root]
        sys.modules["tps_models"] = pkg
        spec = importlib.util.spec_from_file_location("tps_models.dynamic_vit", os.path.join(root, "dynamic_vit.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tps_models.dynamic_vit"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["tps_models.dynamic_vit"].TPSModule(temperature=1).eval()


def etps(keep_rate: float = 0.7, locs: Sequence[int] = DEFAULT_LOCS, official: int = 1):
    tps = _official_tps_module() if official else None

    def f(x, ctx: Ctx):
        if ctx.layer not in locs:
            return x
        B, N, C = x.shape
        score = _cls_attn(ctx)
        k = _n_keep(N - 1, keep_rate)
        idx = score.topk(N - 1, dim=1).indices + 1
        if tps is not None:
            keep, drop = idx[:, :k], idx[:, k:]
            ctx.extra_macs += (N - 1 - k) * k * C
            ctx.size = torch.cat([ctx.size[:, :1], _gather(ctx.size, keep)], 1)
            if ctx.pos is not None:
                ctx.pos = torch.cat([ctx.pos[:, :1], ctx.pos.gather(1, keep)], 1)
            return torch.cat([x[:, :1], tps(_gather(x, keep), _gather(x, drop), None, None, None)], 1)
        keep, drop = idx[:, :k].sort(1)[0], idx[:, k:]
        if ctx.pos is not None:
            ctx.pos = torch.cat([ctx.pos[:, :1], ctx.pos.gather(1, keep)], 1)
        xk, xd = _gather(x, keep), _gather(x, drop)
        sk, sd = _gather(ctx.size, keep), _gather(ctx.size, drop)
        sim = F.normalize(xd, dim=-1) @ F.normalize(xk, dim=-1).transpose(1, 2)
        best = sim.argmax(-1)
        w = sim.gather(2, best.unsqueeze(-1)).clamp_min(0).exp()
        num = xk * sk
        den = sk.clone()
        num = num.scatter_add(1, best.unsqueeze(-1).expand(-1, -1, C), xd * sd * w)
        den = den.scatter_add(1, best.unsqueeze(-1), sd * w)
        xk = num / den
        size_k = sk.scatter_add(1, best.unsqueeze(-1), sd)
        ctx.extra_macs += (N - 1 - k) * k * C
        ctx.size = torch.cat([ctx.size[:, :1], size_k], 1)
        return torch.cat([x[:, :1], xk], 1)

    return f


def _bipartite(metric: torch.Tensor, r: int, protect_cls: bool = True):
    B, N, _ = metric.shape
    r = min(r, (N - 1) // 2)
    if r <= 0:
        return None
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = metric[:, ::2], metric[:, 1::2]
    scores = a @ b.transpose(-1, -2)
    if protect_cls:
        scores[:, 0, :] = -math.inf
    node_max, node_idx = scores.max(-1)
    edge_idx = node_max.argsort(-1, descending=True)[..., None]
    unm_idx = edge_idx[:, r:]
    src_idx = edge_idx[:, :r]
    if protect_cls:

        unm_idx = unm_idx.sort(dim=1)[0]
    dst_idx = node_idx[..., None].gather(1, src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[:, ::2], x[:, 1::2]
        n, t1, c = src.shape
        unm = src.gather(1, unm_idx.expand(n, t1 - r, c))
        s = src.gather(1, src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(1, dst_idx.expand(n, r, c), s, reduce=mode)
        return torch.cat([unm, dst], 1)

    def prune(x: torch.Tensor) -> torch.Tensor:
        src, dst = x[:, ::2], x[:, 1::2]
        n, t1, c = src.shape
        unm = src.gather(1, unm_idx.expand(n, t1 - r, c))
        return torch.cat([unm, dst], 1)

    return merge, prune, (src_idx, dst_idx, unm_idx, r)


def _weighted_merge(merge, x, size):
    x = merge(x * size, "sum")
    size = merge(size, "sum")
    return x / size, size


def tome(r: int = 13):

    def f(x, ctx: Ctx):
        res = _bipartite(ctx.keys, r)
        if res is None:
            return x
        merge, prune, _ = res
        N, C = x.shape[1], ctx.keys.shape[-1]
        ctx.extra_macs += (N // 2) * ((N + 1) // 2) * C
        x, ctx.size = _weighted_merge(merge, x, ctx.size)
        if ctx.pos is not None:
            ctx.pos = prune(ctx.pos.unsqueeze(-1)).squeeze(-1)
        if ctx.source is not None:
            ctx.source = merge(ctx.source, "sum")
        return x

    return f


def tofu(r: int = 13, prune_layers: int = 6, prune_size: int = 0, wmean: int = 0):

    def f(x, ctx: Ctx):
        res = _bipartite(ctx.keys, r)
        if res is None:
            return x
        merge, prune, _ = res
        N, C = x.shape[1], ctx.keys.shape[-1]
        ctx.extra_macs += (N // 2) * ((N + 1) // 2) * C
        if ctx.pos is not None:
            ctx.pos = prune(ctx.pos.unsqueeze(-1)).squeeze(-1)
        if ctx.layer < prune_layers:
            ctx.size = merge(ctx.size, "sum") if prune_size else prune(ctx.size)
            return prune(x)
        norm = x.norm(dim=-1, keepdim=True)
        if wmean:
            xm, size = _weighted_merge(merge, x, ctx.size)
        else:
            xm, size = merge(x, "mean"), merge(ctx.size, "sum")
        max_norm = merge(norm, "amax")
        xm = F.normalize(xm, dim=-1) * max_norm
        ctx.size = size
        return xm

    return f


def pitome(r: int = 13, margin_start: float = 0.9, margin_end: float = 0.0, alpha: float = 1.0):

    def f(x, ctx: Ctx):
        B, N, C = x.shape
        L = ctx.stash.setdefault("depth", 12)
        rr = min(r, (N - 1) // 2)
        if rr <= 0:
            return x
        m = margin_start - (margin_start - margin_end) * ctx.layer / L
        k = F.normalize(ctx.keys[:, 1:], dim=-1)
        sim = k @ k.transpose(1, 2)
        e = torch.where(sim >= m, sim, alpha * (torch.exp(sim - m) - 1)).mean(-1)
        ctx.extra_macs += (N - 1) ** 2 * k.shape[-1]
        order = e.argsort(1, descending=True) + 1
        merge_idx, prot_idx = order[:, : 2 * rr], order[:, 2 * rr:]
        a_idx, b_idx = merge_idx[:, ::2], merge_idx[:, 1::2]
        ka, kb = _gather(F.normalize(ctx.keys, dim=-1), a_idx), _gather(F.normalize(ctx.keys, dim=-1), b_idx)
        s = ka @ kb.transpose(1, 2)
        dst = s.argmax(-1)
        xa, xb = _gather(x, a_idx), _gather(x, b_idx)
        sa, sb = _gather(ctx.size, a_idx), _gather(ctx.size, b_idx)
        num = (xb * sb).scatter_add(1, dst.unsqueeze(-1).expand(-1, -1, C), xa * sa)
        den = sb.scatter_add(1, dst.unsqueeze(-1), sa)
        xb = num / den
        keep_idx = torch.cat([prot_idx, b_idx], 1)
        perm = keep_idx.argsort(1)
        new_x = torch.cat([_gather(x, prot_idx), xb], 1).gather(1, perm.unsqueeze(-1).expand(-1, -1, C))
        new_s = torch.cat([_gather(ctx.size, prot_idx), den], 1).gather(1, perm.unsqueeze(-1))
        ctx.size = torch.cat([ctx.size[:, :1], new_s], 1)
        if ctx.pos is not None:
            ctx.pos = torch.cat([ctx.pos[:, :1], ctx.pos.gather(1, keep_idx).gather(1, perm)], 1)
        return torch.cat([x[:, :1], new_x], 1)

    return f


def ppt(keep_rate: float = 0.7, locs: Sequence[int] = DEFAULT_LOCS, tau: float = 7e-5):

    def f(x, ctx: Ctx):
        if ctx.layer not in locs:
            return x
        B, N, C = x.shape
        a = ctx.attn[:, :, 0, 1:]
        vnorm = ctx.values[:, :, 1:].norm(dim=-1)
        s = (a * vnorm).sum(1)
        s = s / s.sum(1, keepdim=True)
        var = s.var(1)
        n_remove = min((N - 1) - _n_keep(N - 1, keep_rate), (N - 1) // 2)

        k = (N - 1) - n_remove
        idx = s.topk(k, dim=1).indices.sort(1)[0] + 1
        cls = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
        full = torch.cat([cls, idx], 1)
        xp, sp = _gather(x, full), _gather(ctx.size, full)

        res = _bipartite(ctx.keys, n_remove)
        merge, prune, _ = res
        xm, sm = _weighted_merge(merge, x, ctx.size)
        use_prune = (var > tau).view(B, 1, 1)
        ctx.stash.setdefault("ppt_prune_frac", []).append(use_prune.float().mean().item())
        ctx.extra_macs += (N // 2) * ((N + 1) // 2) * ctx.keys.shape[-1]
        ctx.size = torch.where(use_prune, sp, sm)
        if ctx.pos is not None:
            ctx.pos = torch.where(use_prune.view(B, 1), ctx.pos.gather(1, full), prune(ctx.pos.unsqueeze(-1)).squeeze(-1))
        return torch.where(use_prune, xp, xm)

    return f


def _wpr_importance(ctx: Ctx, iters: int, var_range=(0.01, 0.7), cls_init: float = 1.0):
    B, H, N, _ = ctx.attn.shape
    s = torch.full((B, H, N), 1.0, device=ctx.attn.device, dtype=ctx.attn.dtype)
    s[:, :, 0] = cls_init
    s = s / s.sum(-1, keepdim=True)
    for _ in range(iters):
        s = torch.einsum("bhij,bhi->bhj", ctx.attn, s)
    ctx.extra_macs += iters * H * N * N
    p = s[:, :, 1:]

    var = (p / p.mean(-1, keepdim=True)).var(-1)
    keep = (var >= var_range[0]) & (var <= var_range[1])
    keep = keep | (~keep.any(1, keepdim=True))
    w = keep.to(p.dtype).unsqueeze(-1)
    return ((p ** 2) * w).sum(1).div(w.sum(1)).sqrt()


def zero_tprune(keep_rate: float = 0.7, locs: Sequence[int] = DEFAULT_LOCS, iters: Sequence[int] = (30, 5, 1),
                s_frac: float = 0.5, cls_init: float = 1.0):

    def f(x, ctx: Ctx):
        if ctx.layer not in locs:
            return x
        B, N, C = x.shape
        it = iters[min(list(locs).index(ctx.layer), len(iters) - 1)]
        score = _wpr_importance(ctx, it, cls_init=cls_init)
        n_remove = (N - 1) - _n_keep(N - 1, keep_rate)
        n_s = int(round(s_frac * n_remove))
        order = score.argsort(1) + 1
        half = (N - 1) // 2
        A_idx, B_idx = order[:, :half], order[:, half:]
        k = F.normalize(ctx.keys, dim=-1)
        sim = _gather(k, A_idx) @ _gather(k, B_idx).transpose(1, 2)
        best = sim.max(-1).values
        prune_pos = best.topk(n_s, dim=1).indices
        prune_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        prune_mask.scatter_(1, A_idx.gather(1, prune_pos), True)
        ctx.extra_macs += half * (N - 1 - half) * k.shape[-1]

        rest = torch.cat([torch.full((B, 1), -1.0, device=x.device, dtype=score.dtype), score], 1)
        rest = rest.masked_fill(prune_mask, float("inf"))
        rest[:, 0] = float("inf")
        n_i = n_remove - n_s
        drop_i = rest.masked_fill(prune_mask, float("inf")).topk(n_i, dim=1, largest=False).indices
        prune_mask.scatter_(1, drop_i, True)
        keep_idx = (~prune_mask).nonzero()[:, 1].view(B, -1)
        return _apply_index(x, ctx, keep_idx)

    return f


def _wpr(attn: torch.Tensor, iters: int, cls_init: float, var_range=(0.01, 0.7), eir: int = 1,
         cls_fix: int = 0) -> torch.Tensor:
    B, H, N, _ = attn.shape
    s = torch.ones(B, H, N, device=attn.device, dtype=attn.dtype)
    s[:, :, 0] = cls_init
    s = s / s.sum(-1, keepdim=True)
    for _ in range(iters):
        s = torch.einsum("bhij,bhi->bhj", attn, s)
        if cls_fix:
            s = s.clone()
            s[:, :, 0] = cls_init * s[:, :, 1:].mean(-1)
            s = s / s.sum(-1, keepdim=True)
    p = s[:, :, 1:]

    var = (p / p.mean(-1, keepdim=True)).var(-1)
    keep = (var >= var_range[0]) & (var <= var_range[1])
    keep = keep | (~keep.any(1, keepdim=True))
    w = keep.to(p.dtype).unsqueeze(-1)
    if not eir:
        return (p * w).sum(1).div(w.sum(1))
    return ((p ** 2) * w).sum(1).div(w.sum(1)).sqrt()


def zero_tprune_paper(locs: Sequence[int] = (1, 3, 6, 9, 11), retention: Sequence[float] = (1, 0.9, 0.8, 0.7, 1),
                      s: int = 10, iters: Sequence[int] = (), cls_sqrt: int = 1, renorm: int = 1, depth: int = 12,
                      vhf: int = 1, eir: int = 1, kcat: int = 0, rank_cls: int = 0, rank_rand: int = 0,
                      cls_fix: int = 0):
    locs = [int(l) for l in (locs if isinstance(locs, (list, tuple)) else [locs])]
    retention = list(retention if isinstance(retention, (list, tuple)) else [retention] * len(locs))
    iters = list(iters if isinstance(iters, (list, tuple)) else [iters])
    vr = (0.01, 0.7) if vhf else (-math.inf, math.inf)

    def n_iter(i, l):
        if iters:
            return int(iters[i])
        return 30 if l <= 3 else (1 if l > depth - 3 else 5)

    def f(x, ctx: Ctx):
        l = ctx.layer + 1
        if l not in locs:
            return x
        i = locs.index(l)
        a = ctx.attn
        B, H, N, _ = a.shape
        cls_init = math.sqrt(N) if cls_sqrt else 1.0
        dev = x.device
        keep = torch.arange(N, device=dev).expand(B, N)
        if s > 0:
            score = _wpr(a, 1, cls_init, vr, eir, cls_fix)
            ctx.extra_macs += H * N * N
            order = score.argsort(1) + 1
            half = (N - 1) // 2
            A_idx, B_idx = order[:, :half], order[:, half:]
            k = F.normalize(ctx.keys_cat if kcat else ctx.keys, dim=-1)
            best = (_gather(k, A_idx) @ _gather(k, B_idx).transpose(1, 2)).max(-1).values
            ctx.extra_macs += half * (N - 1 - half) * k.shape[-1]
            pruned = A_idx.gather(1, best.topk(min(s, half), dim=1).indices)
            mask = torch.ones(B, N, dtype=torch.bool, device=dev)
            mask.scatter_(1, pruned, False)
            keep = mask.nonzero()[:, 1].view(B, -1)
        r = float(retention[i])
        if r < 1:
            n = keep.shape[1]
            sub = a.gather(2, keep[:, None, :, None].expand(B, H, n, N))
            sub = sub.gather(3, keep[:, None, None, :].expand(B, H, n, n))
            if renorm:
                sub = sub / sub.sum(-1, keepdim=True).clamp_min(1e-12)
            it = n_iter(i, l)
            if rank_rand:
                score = torch.rand(B, n - 1, device=dev)
            elif rank_cls:
                score = sub[:, :, 0, 1:].mean(1)
            else:
                score = _wpr(sub, it, math.sqrt(n) if cls_sqrt else 1.0, vr, eir, cls_fix)
            ctx.extra_macs += it * H * n * n
            n_keep = max(1, int(round(r * (n - 1))))
            top = score.topk(n_keep, dim=1).indices + 1
            top, _ = top.sort(dim=1)
            keep = torch.cat([keep[:, :1], keep.gather(1, top)], 1)
        if keep.shape[1] == N:
            return x
        return _apply_index(x, ctx, keep)

    f.after_mlp = True
    return f


def mctf(r: int = 16, tau_sim: float = 1.0, tau_info: float = 1.0 / 20, tau_size: float = 1.0 / 40):


    def f(x, ctx: Ctx):
        B, N, C = x.shape
        r_tot = min(r, (N - 1) // 2)
        if r_tot <= 0:
            return x
        r1, r2 = (r_tot + 1) // 2, r_tot // 2
        a = ctx.attn.mean(1).mean(1)
        xn = F.normalize(x, dim=-1)
        ctx.extra_macs += (N // 2) * ((N + 1) // 2) * C

        def weights(i_idx, j_idx):
            sim = (_gather(xn, i_idx) @ _gather(xn, j_idx).transpose(1, 2) + 1) / 2
            ai, aj = a.gather(1, i_idx), a.gather(1, j_idx)
            si, sj = ctx.size[..., 0].gather(1, i_idx), ctx.size[..., 0].gather(1, j_idx)
            w_info = 1.0 / (ai.unsqueeze(-1) * aj.unsqueeze(1)).clamp_min(1e-12)
            w_size = 1.0 / (si.unsqueeze(-1) * sj.unsqueeze(1))
            return sim.clamp_min(1e-12) ** tau_sim * w_info ** tau_info * w_size ** tau_size

        idx = torch.arange(N, device=x.device).expand(B, -1)
        ev, od = idx[:, 2::2], idx[:, 1::2]
        W = weights(ev, od)
        tok = x * (a.unsqueeze(-1) * ctx.size)
        wt = a.unsqueeze(-1) * ctx.size
        size = ctx.size.clone()
        srcm = ctx.source.clone() if ctx.source is not None else None

        def fuse_step(src, dst, Wsd, tok, wt, size, srcm, rr):
            best_w, best_j = Wsd.max(-1)
            order = best_w.argsort(1, descending=True)
            sel = order[:, :rr]
            src_sel = src.gather(1, sel)
            dst_sel = dst.gather(1, best_j.gather(1, sel))
            tok = tok.scatter_add(1, dst_sel.unsqueeze(-1).expand(-1, -1, C), _gather(tok, src_sel))
            wt = wt.scatter_add(1, dst_sel.unsqueeze(-1), _gather(wt, src_sel))
            size = size.scatter_add(1, dst_sel.unsqueeze(-1), _gather(size, src_sel))
            if srcm is not None:
                srcm = srcm.scatter_add(1, dst_sel.unsqueeze(-1).expand(-1, -1, srcm.shape[-1]), _gather(srcm, src_sel))
            keep_src = order[:, rr:].sort(1)[0]
            return src.gather(1, keep_src), keep_src, tok, wt, size, srcm

        ev_left, keep_pos, tok, wt, size, srcm = fuse_step(ev, od, W, tok, wt, size, srcm, r1)

        W2 = W.gather(1, keep_pos.unsqueeze(-1).expand(-1, -1, W.shape[2])).transpose(1, 2)
        od_left, _, tok, wt, size, srcm = fuse_step(od, ev_left, W2, tok, wt, size, srcm, r2)
        keep = torch.cat([idx[:, :1], ev_left, od_left], 1).sort(1)[0]
        ctx.size = _gather(size, keep)
        if ctx.pos is not None:
            ctx.pos = ctx.pos.gather(1, keep)
        if srcm is not None:
            ctx.source = _gather(srcm, keep)
        out = _gather(tok, keep) / _gather(wt, keep).clamp_min(1e-12)
        out[:, 0] = x[:, 0]
        return out

    return f


def atc(keep_rate: float = 0.7, locs: Sequence[int] = DEFAULT_LOCS, linkage: str = "average"):
    from scipy.cluster.hierarchy import fcluster, linkage as sp_linkage
    import numpy as np

    def f(x, ctx: Ctx):
        if ctx.layer not in locs:
            return x
        B, N, C = x.shape
        K = _n_keep(N - 1, keep_rate)
        keys = F.normalize(ctx.keys[:, 1:], dim=-1).double().cpu().numpy()
        outs, sizes, poss = [], [], []
        for b in range(B):
            Z = sp_linkage(keys[b], method=linkage, metric="cosine")
            lab = torch.as_tensor(fcluster(Z, t=K, criterion="maxclust") - 1, device=x.device)
            n_cl = int(lab.max()) + 1
            sz = ctx.size[b, 1:]
            num = torch.zeros(n_cl, C, device=x.device, dtype=x.dtype).index_add_(0, lab, x[b, 1:] * sz)
            den = torch.zeros(n_cl, 1, device=x.device, dtype=x.dtype).index_add_(0, lab, sz)
            outs.append(torch.cat([x[b, :1], num / den], 0))
            sizes.append(torch.cat([ctx.size[b, :1], den], 0))
            if ctx.pos is not None:
                first = torch.full((n_cl,), N, device=x.device, dtype=torch.long)
                first = first.scatter_reduce(0, lab, torch.arange(1, N, device=x.device), "amin")
                poss.append(torch.cat([ctx.pos[b, :1], ctx.pos[b].gather(0, first)], 0))
        L = max(o.shape[0] for o in outs)

        out = torch.zeros(B, L, C, device=x.device, dtype=x.dtype)
        size = torch.full((B, L, 1), 1e-6, device=x.device, dtype=x.dtype)
        pos = torch.zeros(B, L, device=x.device, dtype=torch.long) if ctx.pos is not None else None
        for b in range(B):
            out[b, : outs[b].shape[0]] = outs[b]
            size[b, : sizes[b].shape[0]] = sizes[b]
            if pos is not None:
                pos[b, : poss[b].shape[0]] = poss[b]
        ctx.extra_macs += (N - 1) ** 2 * ctx.keys.shape[-1]
        ctx.size = size
        ctx.pos = pos
        return out

    return f


def token_pooling(r: int = 13, iters: int = 5, ks: Sequence[int] = (), sq: int = 0):
    ks = list(ks if isinstance(ks, (list, tuple)) else [ks])

    def f(x, ctx: Ctx):
        B, N, C = x.shape
        if ks:
            K = int(ks[ctx.layer]) if ctx.layer < len(ks) else N - 1
            if K <= 0:
                ctx.size = ctx.size[:, :1]
                return x[:, :1]
        else:
            K = (N - 1) - min(r, (N - 1) // 2)
        if K >= N - 1:
            return x
        w = ctx.attn.sum(2).mean(1)[:, 1:]
        z = x[:, 1:]
        d = torch.cdist(z, z)
        dm = d ** 2 if sq else d
        med = w.topk(K, dim=1).indices
        for _ in range(iters):
            assign = d.gather(2, med.unsqueeze(1).expand(-1, N - 1, -1)).argmin(-1)

            onehot = F.one_hot(assign, K).to(x.dtype) * w.unsqueeze(-1)
            cost = dm @ onehot
            member = F.one_hot(assign, K).bool()
            cost = cost.masked_fill(~member, float("inf"))
            med = cost.argmin(1)
        assign = d.gather(2, med.unsqueeze(1).expand(-1, N - 1, -1)).argmin(-1)
        ww = (w.unsqueeze(-1) * ctx.size[:, 1:])
        num = torch.zeros(B, K, C, device=x.device, dtype=x.dtype).scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, C), z * ww)
        den = torch.zeros(B, K, 1, device=x.device, dtype=x.dtype).scatter_add_(1, assign.unsqueeze(-1), ww)
        sz = torch.zeros(B, K, 1, device=x.device, dtype=x.dtype).scatter_add_(1, assign.unsqueeze(-1), ctx.size[:, 1:])

        ctx.extra_macs += (N - 1) ** 2 * C + iters * (N - 1) ** 2 * K
        ctx.size = torch.cat([ctx.size[:, :1], sz.clamp_min(1e-6)], 1)
        return torch.cat([x[:, :1], num / den.clamp_min(1e-12)], 1)

    return f


def _greedy_pairs(sim: torch.Tensor, m: int):
    B, N, _ = sim.shape
    sim = sim.clone()
    eye = torch.eye(N, dtype=torch.bool, device=sim.device)
    sim.masked_fill_(eye, -math.inf)
    src = torch.empty(B, m, dtype=torch.long, device=sim.device)
    dst = torch.empty_like(src)
    rows = torch.arange(B, device=sim.device)
    for t in range(m):
        flat = sim.view(B, -1).argmax(-1)
        i, j = flat // N, flat % N
        src[:, t], dst[:, t] = i, j
        for v in (i, j):
            sim[rows, v, :] = -math.inf
            sim[rows, :, v] = -math.inf
    return src, dst


def long_bat(keep_rate: float = 0.7, locs: Sequence[int] = DEFAULT_LOCS, pair_frac: float = 0.1, knn: int = 5,
             paper: int = 0):

    def f(x, ctx: Ctx):
        if ctx.layer not in locs:
            return x
        B, N, C = x.shape
        total = _n_keep(N - 1, keep_rate)
        m = max(1, int(round(pair_frac * total)))
        n_cl = m
        n_att = total - n_cl + m
        score = _cls_attn(ctx)
        order = score.argsort(1, descending=True) + 1
        att, ina = order[:, :n_att], order[:, n_att:]
        xa, xi = _gather(x, att), _gather(x, ina)
        sa, si = _gather(ctx.size, att), _gather(ctx.size, ina)
        wa, wi = score.gather(1, att - 1).unsqueeze(-1), score.gather(1, ina - 1).unsqueeze(-1)

        d = torch.cdist(xi, xi)
        if paper:
            log_rho = -(d ** 2).sum(-1)
            higher = log_rho.unsqueeze(1) > log_rho.unsqueeze(2)
            delta = d.masked_fill(~higher, float("inf")).min(-1).values
            delta = torch.where(torch.isinf(delta), d.max(-1).values, delta)

            centres = (log_rho + delta.clamp_min(1e-12).log()).topk(n_cl, dim=1).indices
            dc = d.gather(2, centres.unsqueeze(1).expand(-1, d.shape[1], -1))
            ok = log_rho.gather(1, centres).unsqueeze(1) > log_rho.unsqueeze(2)
            dc_ok = dc.masked_fill(~ok, float("inf"))
            assign = torch.where(ok.any(-1), dc_ok.argmin(-1), dc.argmin(-1))
        else:
            kk = min(knn, d.shape[1] - 1)
            rho = (-(d.topk(kk + 1, largest=False).values[..., 1:] ** 2).mean(-1)).exp()
            higher = rho.unsqueeze(1) > rho.unsqueeze(2)
            delta = d.masked_fill(~higher, float("inf")).min(-1).values
            delta = torch.where(torch.isinf(delta), d.max(-1).values, delta)
            centres = (rho * delta).topk(n_cl, dim=1).indices
            assign = d.gather(2, centres.unsqueeze(1).expand(-1, d.shape[1], -1)).argmin(-1)
        assign.scatter_(1, centres, torch.arange(n_cl, device=x.device).expand(B, -1))
        wsum = wi * si
        num = torch.zeros(B, n_cl, C, device=x.device, dtype=x.dtype).scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, C), xi * wsum)
        den = torch.zeros(B, n_cl, 1, device=x.device, dtype=x.dtype).scatter_add_(1, assign.unsqueeze(-1), wsum)
        szc = torch.zeros(B, n_cl, 1, device=x.device, dtype=x.dtype).scatter_add_(1, assign.unsqueeze(-1), si)
        clus = num / den.clamp_min(1e-12)

        xn = F.normalize(xa, dim=-1)
        if paper:
            src, dst = _greedy_pairs(xn @ xn.transpose(1, 2), min(m, n_att // 2))
        else:
            Aa, Bb = torch.arange(0, n_att, 2, device=x.device), torch.arange(1, n_att, 2, device=x.device)
            sim = xn[:, Aa] @ xn[:, Bb].transpose(1, 2)
            bw, bj = sim.max(-1)
            sel = bw.topk(min(m, Aa.numel()), dim=1).indices
            src = Aa[sel]
            dst = Bb[bj.gather(1, sel)]
        wtok = (wa * sa)
        tok = xa * wtok
        tok = tok.scatter_add(1, dst.unsqueeze(-1).expand(-1, -1, C), _gather(tok, src))
        wtok2 = wtok.scatter_add(1, dst.unsqueeze(-1), _gather(wtok, src))
        sa2 = sa.scatter_add(1, dst.unsqueeze(-1), _gather(sa, src))
        keep = torch.ones(B, n_att, dtype=torch.bool, device=x.device)
        keep.scatter_(1, src, False)
        kidx = keep.nonzero()[:, 1].view(B, -1)
        att_out = _gather(tok / wtok2.clamp_min(1e-12), kidx)
        if ctx.pos is not None:
            pa = ctx.pos.gather(1, att).gather(1, kidx)
            pc = ctx.pos.gather(1, ina).gather(1, centres)
            ctx.pos = torch.cat([ctx.pos[:, :1], pa, pc], 1)
        ctx.size = torch.cat([ctx.size[:, :1], _gather(sa2, kidx), szc], 1)
        ctx.extra_macs += (N - 1 - n_att) ** 2 * C + (n_att // 2) ** 2 * C
        return torch.cat([x[:, :1], att_out, clus], 1)

    return f


PROP_ATTN = {"none": False, "topk": False, "evit": False, "etps": False, "zero_tprune": False, "zero_tprune_v1": False,
             "ats": True, "ppt": True, "tome": True, "tofu": True, "pitome": True, "mctf": True,
             "atc": True, "token_pooling": False, "long_bat": False, "ltmp": True}

REGISTRY = {
    "none": lambda **kw: None,
    "topk": topk,
    "evit": lambda **kw: topk(fuse=True, **kw),
    "ats": ats,
    "etps": etps,
    "tome": tome,
    "tofu": tofu,
    "pitome": pitome,
    "ppt": ppt,
    "zero_tprune": zero_tprune_paper,
    "zero_tprune_v1": zero_tprune,
    "mctf": mctf,
    "atc": atc,
    "token_pooling": token_pooling,
    "long_bat": long_bat,
    "ltmp": lambda path: _load_ltmp(path),
}


def _load_ltmp(path):
    import json
    from .ltmp import ltmp_infer
    d = json.load(open(path))
    return ltmp_infer(d["theta_p"], d["theta_m"])
