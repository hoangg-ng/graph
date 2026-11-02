"""MAGE with a swappable temporal encoder.

The backbone (MAGE blocks, prompt pool, decoders, hyper-residual wiring) is the official code,
imported unchanged. Only two things differ from official/src/models/MAGE.py:
  1. `encoder` is chosen by name (see ENCODERS); 'linear' is the official single projection.
  2. MAGE_Block's 6-operand einsum is contracted in an explicit right-to-left order. It is the
     same function (checked in cost.py), but without opt_einsum torch contracts left-to-right and
     materialises a (b, N, K, 2, N) tensor, i.e. O(N^2). The explicit order is O(N * dG).
All encoders act on each node's own window independently, so they add O(N * T * d) work.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'official'))
from src.models import MAGE as M  # noqa: E402


def linear_mage_block_forward(self, x, bias):
    route, topk_indices = self.selector(x, bias)            # (b, n, k)
    coef = self.cal_coefficient()                           # (e,)
    E1 = torch.softmax(self.E1, dim=-1)                     # (e, k, n, d)
    E2 = torch.softmax(self.E2, dim=-1)                     # (e, k, d, n)
    b, n, f = x.shape
    xo = x @ self.cal_outer()                                            # (b, n, f)
    Z = torch.einsum('ekdj,bjf->bekdf', E2, xo) * coef.view(1, -1, 1, 1, 1)  # (b, e, k, dG, f)
    A = route.unsqueeze(1).unsqueeze(-1) * E1.permute(0, 2, 1, 3).unsqueeze(0)  # (b, e, n, k, dG)
    A = A.permute(0, 2, 1, 3, 4).reshape(b, n, -1)                       # (b, n, e*k*dG)
    y = torch.bmm(A, Z.reshape(b, -1, f))                                # (b, n, f)
    return y, topk_indices


# ---------------------------------------------------------------- temporal encoders
# input x: (b, T, n, C) -> output (b, n, d)

def _pool(x, s):
    """Average-pool the time axis by factor s. x: (b, T, n, C) -> (b, n, T//s * C)."""
    b, T, n, C = x.shape
    if s > 1:
        x = x[:, T % s:]  # drop the oldest steps if T is not divisible
        x = x.reshape(b, T // s, s, n, C).mean(2)
    return x.transpose(1, 2).reshape(b, n, -1)


class LinearEnc(nn.Module):
    """Official MAGE: one linear map of the flattened window."""
    def __init__(self, T, C, d, **_):
        super().__init__()
        self.proj = nn.Linear(T * C, d)

    def forward(self, x):
        return self.proj(_pool(x, 1))


class MultiScaleEnc(nn.Module):
    """Proposed: parallel branches over average-pooled views, summed.

    nonlinear=True  -> each branch is Linear-GELU-Linear (the encoder under study)
    nonlinear=False -> each branch is a single Linear; the sum then collapses to one linear map
                       of the window (a re-parameterisation control, no added expressiveness).
    """
    def __init__(self, T, C, d, scales=(1, 2, 4), hidden=None, nonlinear=True, **_):
        super().__init__()
        self.scales = [s for s in scales if T // s >= 1]
        h = hidden or d
        def branch(s):
            i = (T // s) * C
            return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, d)) if nonlinear else nn.Linear(i, d)
        self.branches = nn.ModuleList(branch(s) for s in self.scales)

    def forward(self, x):
        return sum(br(_pool(x, s)) for s, br in zip(self.scales, self.branches))


class GatedMultiScaleEnc(nn.Module):
    """research_design.md encoder:  H = X W0 + sum_m g_m(node) * f_m(pool_m(X)).

    gate='node'   -> g = softmax(W_g e_n + b), e_n = MAGE's node embedding (model 5, proposed)
    gate='global' -> g = softmax(theta), one weight vector shared by all nodes (model 4)
    gate=None     -> plain sum (used by model 3, which has a single branch)
    Gates start uniform (zero init). f_m = Linear-GELU-Linear, hidden d unless given.
    """
    def __init__(self, T, C, d, scales=(1, 2, 4), gate='node', hidden=None, **_):
        super().__init__()
        self.scales = [s for s in scales if T // s >= 1]
        self.W0 = nn.Linear(T * C, d)
        h = hidden or d
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Linear((T // s) * C, h), nn.GELU(), nn.Linear(h, d)) for s in self.scales)
        self.gate_kind = gate
        M_ = len(self.scales)
        if gate == 'node':
            self.gate = nn.Linear(d, M_)
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        elif gate == 'global':
            self.gate = nn.Parameter(torch.zeros(M_))

    def gates(self, node_emb):
        if self.gate_kind == 'node':
            return torch.softmax(self.gate(node_emb), -1)          # (n, M)
        if self.gate_kind == 'global':
            return torch.softmax(self.gate, -1).expand(node_emb.shape[0], -1)
        return None

    def forward(self, x, node_emb=None):
        out = self.W0(_pool(x, 1))
        g = self.gates(node_emb)
        for m, (s, br) in enumerate(zip(self.scales, self.branches)):
            y = br(_pool(x, s))
            out = out + (y if g is None else g[:, m:m + 1] * y)
        return out


class _MovingAvg(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):  # x: (B, L, D) along time L
        p = (self.k - 1) // 2
        xp = torch.cat([x[:, :1].expand(-1, p, -1), x, x[:, -1:].expand(-1, self.k - 1 - p, -1)], 1)
        return F.avg_pool1d(xp.transpose(1, 2), self.k, 1).transpose(1, 2)


class TimeMixerEnc(nn.Module):
    """TimeMixer-lite (Wang et al., ICLR 2024) adapted as a per-node encoder.

    Multiscale views by average pooling -> channel embedding -> one Past-Decomposable-Mixing
    layer (moving-average season/trend split, bottom-up seasonal mixing, top-down trend mixing,
    each mixer two linear layers with GELU along time) -> per-scale flatten+linear, summed
    (the FMM ensemble, applied to the embedding instead of to predictions).
    """
    def __init__(self, T, C, d, scales=(1, 2, 4), dc=16, ma=5, **_):
        super().__init__()
        self.scales = [s for s in scales if T // s >= 1]
        self.L = [T // s for s in self.scales]
        self.embed = nn.Linear(C, dc)
        self.decomp = _MovingAvg(ma)
        mlp = lambda a, b: nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, b))
        self.up = nn.ModuleList(mlp(self.L[m - 1], self.L[m]) for m in range(1, len(self.L)))
        self.down = nn.ModuleList(mlp(self.L[m + 1], self.L[m]) for m in range(len(self.L) - 1))
        self.out_cross = nn.Sequential(nn.Linear(dc, dc), nn.GELU(), nn.Linear(dc, dc))
        self.heads = nn.ModuleList(nn.Linear(L * dc, d) for L in self.L)

    def forward(self, x):
        b, T, n, C = x.shape
        xs = []
        for s in self.scales:
            xm = x[:, T % s:] if s > 1 else x
            if s > 1:
                xm = xm.reshape(b, T // s, s, n, C).mean(2)
            xs.append(self.embed(xm.permute(0, 2, 1, 3).reshape(b * n, -1, C)))  # (bn, L, dc)
        trend = [self.decomp(z) for z in xs]
        season = [z - t for z, t in zip(xs, trend)]
        mix = lambda f, z: f(z.transpose(1, 2)).transpose(1, 2)  # mix along time
        for m in range(1, len(season)):                     # bottom-up: fine -> coarse
            season[m] = season[m] + mix(self.up[m - 1], season[m - 1])
        for m in range(len(trend) - 2, -1, -1):             # top-down: coarse -> fine
            trend[m] = trend[m] + mix(self.down[m], trend[m + 1])
        out = 0
        for m, z in enumerate(xs):
            z = z + self.out_cross(season[m] + trend[m])
            out = out + self.heads[m](z.reshape(b * n, -1))
        return out.reshape(b, n, -1)


ENCODERS = {
    'linear': lambda T, C, d: LinearEnc(T, C, d),
    'mslin': lambda T, C, d: MultiScaleEnc(T, C, d, nonlinear=False),
    'mlp1': None,  # built below: single-scale MLP, hidden widened to match 'ms' params
    'ms': lambda T, C, d: MultiScaleEnc(T, C, d),
    'tmix': lambda T, C, d: TimeMixerEnc(T, C, d),
}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def default_scales(T):
    # Fixed before any long-window result: short windows (T <= 48) use pooling factors {1, 2, 4},
    # as in the T = 12 study; long windows use {1, 4, 16} (5-min, 20-min, 80-min resolution),
    # so the coarsest view at T = 96 still has 6 points and at T = 288 has 18.
    return (1, 2, 4) if T <= 48 else (1, 4, 16)


def build_encoder(name, T, C, d):
    # research_design.md models: m1 = 'linear', m2 = 'mslin' (pooled views as extra input
    # channels + one linear projection: Lin(concat) == sum of Lin per view), m3..m5 below.
    sc = default_scales(T)
    if name == 'm4':
        return GatedMultiScaleEnc(T, C, d, scales=sc, gate='global')
    if name == 'm5':
        return GatedMultiScaleEnc(T, C, d, scales=sc, gate='node')
    if name == 'm3':  # X W0 + one scale-1 MLP, hidden widened to match m5's encoder params
        target = n_params(GatedMultiScaleEnc(T, C, d, scales=sc, gate='node'))
        base = n_params(nn.Linear(T * C, d))
        h = max(1, round((target - base - d) / (T * C + 1 + d)))
        return GatedMultiScaleEnc(T, C, d, scales=(1,), gate=None, hidden=h)
    if name == 'mlp1':
        target = n_params(MultiScaleEnc(T, C, d))
        h = max(1, round((target - d) / (T * C + 1 + d)))   # params of Lin(TC,h)+Lin(h,d)
        return MultiScaleEnc(T, C, d, scales=(1,), hidden=h)
    if name.startswith('ms_s'):                             # e.g. ms_s12 -> scales (1, 2)
        return MultiScaleEnc(T, C, d, scales=tuple(int(c) for c in name[4:].split('-')))
    return ENCODERS[name](T, C, d)


class MAGEX(M.MAGE):
    def __init__(self, model_args, encoder='linear', **kw):
        super().__init__(model_args, **kw)
        self.encoder = build_encoder(encoder, self.seq_len, self.input_dim, model_args.model_dim)

    def forward(self, x, label=None, bias=(0, 0, 0)):
        prompt = self.prompt(x)
        if isinstance(self.encoder, GatedMultiScaleEnc):
            enc = self.encoder(x, self.prompt.spatial_prompt)
        else:
            enc = self.encoder(x)
        h = x = enc + prompt
        z, _ = x, topk1 = self.backbone1(x, bias[0])
        x, topk2 = self.backbone2(x, bias[1])
        x, topk3 = self.backbone3(h - x, bias[2])
        x = self.decoder(x) + self.decoder2(z)
        b = x.shape[0]
        out = x.reshape(b, self.node_num, self.horizon, -1).transpose(1, 2)
        if self.training:
            K = self.args.recur_num
            return out, torch.stack([torch.bincount(t.flatten(), minlength=K) for t in (topk1, topk2, topk3)])
        return out


def use_linear_contraction(on=True):
    M.MAGE_Block.forward = linear_mage_block_forward if on else _ORIG_FWD


_ORIG_FWD = M.MAGE_Block.forward
