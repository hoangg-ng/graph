"""Step 8: MAGE with separate time-scale streams and scale-dependent graph depth (research_design.md).

The input window is average-pooled by s in {1, 2, 4, 8}. Each scale gets its own linear encoder
and its own decoder pair, and runs through MAGE's three blocks (same wiring, hyper-residual
h - x into block 3). Everything inside a block is shared across scales: router, E1/E2, lambda,
norms and SwiGLU FFN. Routing is computed once per block from the finest scale and reused.

Per block and scale the graph operator is S_m = P^k_m - lambda * S-, with S+/S- MAGE's
routed positive/negative mixtures and P = (I + S+)/2. Powers are applied by repeated factorised
multiplication, so no N x N matrix is formed and the cost stays O(N).
"""
import torch
import torch.nn as nn

import models
from models import M, _pool

SCALES = (1, 2, 4, 8)
ARMS = {
    # arm: (scales, k per scale or a mode name)
    'mage': ((1,), 'mage'),          # MAGE's own operator S+ - lambda S-, one scale: code check
    'lazy1': ((1,), (1,)),           # lazy operator only
    'shared': (SCALES, (1, 1, 1, 1)),
    'proposed': (SCALES, (1, 2, 4, 8)),
    'reversed': (SCALES, (8, 4, 2, 1)),
    'uniform': (SCALES, 'uniform'),
    'nomix': (SCALES, 'nomix'),
    'learned': (SCALES, 'learned'),
}
KMAX = 8


class GraphOps:
    """Routed positive / negative graph operators of one MAGE block, applied in factorised form."""

    def __init__(self, attn, route):
        E1 = torch.softmax(attn.E1, dim=-1)                  # (2, e, n, dG)
        self.E2 = torch.softmax(attn.E2, dim=-1)             # (2, e, dG, n)
        b, n, e = route.shape
        # A[s]: (b, n, e*dG), the node -> (expert, hub) weights including the route
        self.A = [(route.unsqueeze(-1) * E1[s].permute(1, 0, 2).unsqueeze(0)).reshape(b, n, -1) for s in (0, 1)]
        self.lam = attn.cal_lambda()                         # = -lambda (MAGE's coefficient of S-)

    def apply(self, s, Z):
        """S+ Z (s = 0) or S- Z (s = 1). Z: (b, n, f)."""
        b, n, f = Z.shape
        Zh = torch.einsum('edj,bjf->bedf', self.E2[s], Z).reshape(b, -1, f)   # (b, e*dG, f)
        return torch.bmm(self.A[s], Zh)

    def lazy(self, Z):
        return 0.5 * (Z + self.apply(0, Z))

    def power(self, Z, k):
        for _ in range(k):
            Z = self.lazy(Z)
        return Z


def stream_block(bk, Hs, bias, mode, gamma=None):
    """One MAGE block (Pre_Norm_Transformer `bk`) applied to every scale stream."""
    attn = bk.attn
    route, topk = attn.selector(bk.attn_norm(Hs[0]), bias)
    ops = GraphOps(attn, route)
    out = []
    for m, H in enumerate(Hs):
        X = bk.attn_norm(H)
        if mode == 'mage':
            S = ops.apply(0, X) + ops.lam * ops.apply(1, X)
        elif mode == 'nomix':
            S = torch.zeros_like(X)
        elif mode == 'uniform':
            S = X.mean(dim=1, keepdim=True).expand_as(X)
        elif mode == 'learned':
            w = torch.softmax(gamma[m], -1)
            Z, S = X, w[0] * X
            for k in range(1, KMAX + 1):
                Z = ops.lazy(Z)
                S = S + w[k] * Z
            S = S + ops.lam * ops.apply(1, X)
        else:                                                # tuple of powers
            S = ops.power(X, mode[m]) + ops.lam * ops.apply(1, X)
        U = H + S
        out.append(U + bk.ffn(bk.ffn_norm(U)))
    return out, topk


class MAGEStream(M.MAGE):
    def __init__(self, model_args, arm='proposed', **kw):
        super().__init__(model_args, **kw)
        self.arm = arm
        self.scales, self.mode = ARMS[arm]
        d, C, T, H = model_args.model_dim, self.input_dim, self.seq_len, self.horizon * self.output_dim
        del self.encoder, self.decoder, self.decoder2
        self.encoders = nn.ModuleList(nn.Linear((T // s) * C, d) for s in self.scales)
        self.decoders = nn.ModuleList(nn.Linear(d, H) for _ in self.scales)
        self.decoders2 = nn.ModuleList(nn.Linear(d, H) for _ in self.scales)
        self.gamma = nn.Parameter(torch.zeros(len(self.scales), KMAX + 1)) if self.mode == 'learned' else None

    def forward(self, x, label=None, bias=(0, 0, 0)):
        prompt = self.prompt(x)
        hs = [enc(_pool(x, s)) + prompt for s, enc in zip(self.scales, self.encoders)]
        zs, topk1 = stream_block(self.backbone1, hs, bias[0], self.mode, self.gamma)
        xs, topk2 = stream_block(self.backbone2, zs, bias[1], self.mode, self.gamma)
        xs, topk3 = stream_block(self.backbone3, [h - x for h, x in zip(hs, xs)], bias[2], self.mode, self.gamma)
        y = sum(D(x) + D2(z) for D, D2, x, z in zip(self.decoders, self.decoders2, xs, zs))
        b = y.shape[0]
        out = y.reshape(b, self.node_num, self.horizon, -1).transpose(1, 2)
        if self.training:
            K = self.args.recur_num
            return out, torch.stack([torch.bincount(t.flatten(), minlength=K) for t in (topk1, topk2, topk3)])
        return out

    def encoder_params(self):
        return models.n_params(self.encoders)
