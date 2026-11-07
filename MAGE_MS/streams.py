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
    'uniform0': (SCALES, 'uniform0'),   # pure node mean, no learned S- (proposal's 'global mean mixing')
    'nomix': (SCALES, 'nomix'),
    'nomix1': ((1,), 'nomix'),
    'lite': ((1,), 'lite'),           # MAGE with the (collapsed) graph experts replaced by c_b * sensor mean       # MAGE with the spatial operator set to 0: value of space
    'learned': (SCALES, 'learned'),
}
KMAX = 8


class GraphOps:
    """Routed positive / negative graph operators of one MAGE block, applied in factorised form."""

    def __init__(self, attn, route, lam_e=None):
        E1 = torch.softmax(attn.E1, dim=-1)                  # (2, e, n, dG)
        self.E2 = torch.softmax(attn.E2, dim=-1)             # (2, e, dG, n)
        b, n, e = route.shape
        # A[s]: (b, n, e*dG), the node -> (expert, hub) weights including the route
        self.A = [(route.unsqueeze(-1) * E1[s].permute(1, 0, 2).unsqueeze(0)).reshape(b, n, -1) for s in (0, 1)]
        self.lam = attn.cal_lambda()                         # = -lambda (MAGE's coefficient of S-)
        if lam_e is not None:
            # bounded variant (review of 2026-11-02): one sigmoid coefficient per expert,
            # S- := sum_e r_e * lambda_e * S-_e with lambda_e in (0, 1), applied with coefficient -1
            A1 = (route * lam_e).unsqueeze(-1) * E1[1].permute(1, 0, 2).unsqueeze(0)
            self.A[1] = A1.reshape(b, n, -1)
            self.lam = -1.0

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


def stream_block(bk, Hs, bias, mode, gamma=None, lam_e=None, c=None):
    """One MAGE block (Pre_Norm_Transformer `bk`) applied to every scale stream."""
    if mode == 'lite':   # no graph experts, router or lambda: U = H + c * mean_over_sensors(LN H)
        out = []
        for H in Hs:
            X = bk.attn_norm(H)
            U = H + c * X.mean(1, keepdim=True)
            out.append(U + bk.ffn(bk.ffn_norm(U)))
        return out, torch.zeros(Hs[0].shape[0], Hs[0].shape[1], 1, dtype=torch.long, device=Hs[0].device)
    attn = bk.attn
    route, topk = attn.selector(bk.attn_norm(Hs[0]), bias)
    ops = GraphOps(attn, route, lam_e)
    out = []
    for m, H in enumerate(Hs):
        X = bk.attn_norm(H)
        if mode == 'mage':
            S = ops.apply(0, X) + ops.lam * ops.apply(1, X)
        elif mode == 'uniform0':
            S = X.mean(1, keepdim=True).expand_as(X)
        elif mode == 'nomix':
            S = torch.zeros_like(X)
        elif mode == 'uniform':
            S = X.mean(1, keepdim=True).expand_as(X) + ops.lam * ops.apply(1, X)
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


class HierLinear(nn.Module):
    """Rank hierarchy (2026-11-07):  W(n, t) = W_general + U_t diag(z_t) V_t + U_s diag(z_n) V_s.

    W_general: full-rank Linear shared by all sensors and times (the general predictor).
    Time level: rank r, coefficients z_t from time-of-day (288 slots) and day-of-week embeddings.
    Sensor level: rank r, coefficients z_n per sensor (r numbers per sensor; bases U_s, V_s shared).
    V_t and V_s start at zero, so training starts from the general model (coarse to fine)."""

    def __init__(self, fin, fout, N, r):
        super().__init__()
        self.g = nn.Linear(fin, fout)
        self.Ut, self.Us = (nn.Parameter(torch.randn(fin, r) / fin ** 0.5) for _ in range(2))
        self.Vt, self.Vs = (nn.Parameter(torch.zeros(r, fout)) for _ in range(2))
        self.z_tod, self.z_dow = nn.Embedding(288, r), nn.Embedding(7, r)
        nn.init.ones_(self.z_tod.weight); nn.init.zeros_(self.z_dow.weight)
        self.z_n = nn.Parameter(torch.ones(N, r))

    def forward(self, x, tod, dow):  # x: (b, n, fin); tod, dow: (b,) long
        zt = (self.z_tod(tod) + self.z_dow(dow))[:, None, :]                    # (b, 1, r)
        return self.g(x) + ((x @ self.Ut) * zt) @ self.Vt + ((x @ self.Us) * self.z_n) @ self.Vs


class MAGEStream(M.MAGE):
    def __init__(self, model_args, arm='proposed', lam='mage', graph_fix=0, hier_rank=0, **kw):
        super().__init__(model_args, **kw)
        self.arm = arm
        self.scales, self.mode = ARMS[arm]
        d, C, T, H = model_args.model_dim, self.input_dim, self.seq_len, self.horizon * self.output_dim
        N = self.node_num
        del self.encoder, self.decoder, self.decoder2
        self.hier = hier_rank > 0
        lin = (lambda i, o: HierLinear(i, o, N, hier_rank)) if self.hier else nn.Linear
        self.encoders = nn.ModuleList(lin((T // s) * C, d) for s in self.scales)
        self.decoders = nn.ModuleList(lin(d, H) for _ in self.scales)
        self.decoders2 = nn.ModuleList(lin(d, H) for _ in self.scales)
        # graph_fix (2026-11-07): in released MAGE the E1/E2 logits start in U(0, 1), the softmax is near-uniform,
        # and Adam + L2 decay drives them to 0, so every graph expert becomes the sensor mean. Start them with a
        # real spread (N(0, 1)); train.py also exempts them from weight decay when this flag is set.
        if self.mode == 'lite':   # drop E1/E2, router, lambda and the unused outer projection entirely
            for bk in (self.backbone1, self.backbone2, self.backbone3):
                del bk.attn
            self.c_logit = nn.Parameter(torch.full((3,), 0.5))
        if graph_fix:
            for bk in (self.backbone1, self.backbone2, self.backbone3):
                nn.init.normal_(bk.attn.E1, 0, 1.0); nn.init.normal_(bk.attn.E2, 0, 1.0)
        self.gamma = nn.Parameter(torch.zeros(len(self.scales), KMAX + 1)) if self.mode == 'learned' else None
        # lam='mage': MAGE's unconstrained scalar lambda per block (default, all earlier runs).
        # lam='sigmoid': lambda_e = sigmoid(theta_be) per block and expert, started at MAGE's lambda_init.
        self.lam_kind = lam
        if lam == 'sigmoid':
            init = torch.tensor([self.backbone1.attn.lambda_init, self.backbone2.attn.lambda_init,
                                 self.backbone3.attn.lambda_init])
            self.lam_logit = nn.Parameter(torch.logit(init)[:, None].repeat(1, model_args.recur_num))
        elif lam != 'mage':
            raise ValueError(lam)

    def c(self, i):
        return torch.tanh(self.c_logit[i]) if self.mode == 'lite' else None

    def lam_e(self, i):
        return torch.sigmoid(self.lam_logit[i]) if self.lam_kind == 'sigmoid' else None

    def forward(self, x, label=None, bias=(0, 0, 0)):
        prompt = self.prompt(x)
        if self.hier:
            tod = (x[:, -1, 0, -2] * 288).round().long().clamp(0, 287)
            dow = (x[:, -1, 0, -1] * 7).round().long().clamp(0, 6)
            ap = lambda m, v: m(v, tod, dow)
        else:
            ap = lambda m, v: m(v)
        hs = [ap(enc, _pool(x, s)) + prompt for s, enc in zip(self.scales, self.encoders)]
        zs, topk1 = stream_block(self.backbone1, hs, bias[0], self.mode, self.gamma, self.lam_e(0), self.c(0))
        xs, topk2 = stream_block(self.backbone2, zs, bias[1], self.mode, self.gamma, self.lam_e(1), self.c(1))
        xs, topk3 = stream_block(self.backbone3, [h - x for h, x in zip(hs, xs)], bias[2], self.mode, self.gamma,
                                 self.lam_e(2), self.c(2))
        y = sum(ap(D, x) + ap(D2, z) for D, D2, x, z in zip(self.decoders, self.decoders2, xs, zs))
        b = y.shape[0]
        out = y.reshape(b, self.node_num, self.horizon, -1).transpose(1, 2)
        if self.training:
            K = self.args.recur_num
            return out, torch.stack([torch.bincount(t.flatten(), minlength=K) for t in (topk1, topk2, topk3)])
        return out

    def encoder_params(self):
        return models.n_params(self.encoders)


def graph_entropy(model):
    """Mean normalised entropy of the hub->sensor weights softmax(E2) per block (1.0 = collapsed to the mean)."""
    if getattr(model, 'mode', None) == 'lite':
        return None
    out = []
    for bk in (model.backbone1, model.backbone2, model.backbone3):
        p = torch.softmax(bk.attn.E2[0].detach().float(), -1)
        out.append((-(p * p.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(float(p.shape[-1])))).mean().item())
    return out
