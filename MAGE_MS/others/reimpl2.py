"""Reimplementations of MTGNN (Wu et al., KDD 2020), STAEformer (Liu et al., CIKM 2023) and ASTGCN (Guo et al.,
AAAI 2019; the `ASTGCN_r` recent-component model), following the official code (nnzhan/MTGNN net.py + layer.py,
XDZhelheim/STAEformer model/STAEformer.py, guoshnBJTU/ASTGCN-r-pytorch model/ASTGCN_r.py) layer by layer, with
the official default sizes. Written here because running the cloned repositories was not permitted.

Every model has a `uniform` flag: when set, its learned spatial mixing (adjacency or spatial attention) is replaced
by the exact uniform mean 1/N at inference. Attention models also record their spatial attention maps when
`record` is set, so their entropy can be measured.
"""
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from reimpl import override


def summarise(att):
    """Small per-call summary of attention maps (..., N, N) with rows over keys: mean normalised row entropy and
    the mean map (on CPU), so recording many test batches does not keep every map in GPU memory."""
    n = att.shape[-1]
    ent = (-(att * att.clamp_min(1e-12).log()).sum(-1) / np.log(n)).mean().item()
    return ent, att.reshape(-1, n, n).mean(0).cpu()


# ================================================================ MTGNN
class MTLayerNorm(nn.Module):
    """Official LayerNorm over (C, N, L) with per-node affine parameters."""

    def __init__(self, shape, eps=1e-5):
        super().__init__()
        self.shape, self.eps = tuple(shape), eps
        self.weight = nn.Parameter(torch.ones(*shape)); self.bias = nn.Parameter(torch.zeros(*shape))

    def forward(self, x, idx):
        return F.layer_norm(x, tuple(x.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.kernels = [2, 3, 6, 7]
        self.tconv = nn.ModuleList(nn.Conv2d(cin, cout // 4, (1, k), dilation=(1, dilation)) for k in self.kernels)

    def forward(self, x):
        xs = [c(x) for c in self.tconv]
        L = xs[-1].size(3)
        return torch.cat([t[..., -L:] for t in xs], 1)


class MixProp(nn.Module):
    """Official mixprop: a = (A + I) / rowsum, h <- alpha x + (1 - alpha) a h, gdep times, concat, 1x1 conv."""

    def __init__(self, cin, cout, gdep, dropout, alpha, owner):
        super().__init__()
        self.mlp = nn.Conv2d((gdep + 1) * cin, cout, 1)
        self.gdep, self.alpha, self.owner = gdep, alpha, [owner]

    def forward(self, x, adj):
        n = adj.size(0)
        o = override(self.owner[0], adj)
        if o is not None:
            a = o
        else:
            adj = adj + torch.eye(n, device=x.device)
            a = adj / adj.sum(1).view(-1, 1)
        h, out = x, [x]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum('ncwl,vw->ncvl', h, a)
            out.append(h)
        return self.mlp(torch.cat(out, 1))


class GraphConstructor(nn.Module):
    def __init__(self, N, k, dim, alpha=3):
        super().__init__()
        self.emb1, self.emb2 = nn.Embedding(N, dim), nn.Embedding(N, dim)
        self.lin1, self.lin2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.k, self.alpha = k, alpha

    def full(self, idx):
        v1 = torch.tanh(self.alpha * self.lin1(self.emb1(idx)))
        v2 = torch.tanh(self.alpha * self.lin2(self.emb2(idx)))
        return F.relu(torch.tanh(self.alpha * (v1 @ v2.t() - v2 @ v1.t())))

    def forward(self, idx):
        adj = self.full(idx)
        mask = torch.zeros_like(adj)
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        return adj * mask


class MTGNN(nn.Module):
    """Official defaults: gcn_depth 2, dropout 0.3, subgraph_size 20, node_dim 40, conv/residual 32, skip 64,
    end 128, 3 layers, dilation_exponential 1, propalpha 0.05, tanhalpha 3, in_dim 2, seq 12 -> 12."""

    def __init__(self, N, in_dim=2, out_dim=12, seq=12, layers=3, dropout=0.3, k=20, node_dim=40, gdep=2,
                 cc=32, rc=32, sc=64, ec=128, propalpha=0.05, tanhalpha=3):
        super().__init__()
        self.uniform, self.fixed, self.dropout, self.layers, self.seq = False, None, dropout, layers, seq
        self.gc = GraphConstructor(N, k, node_dim, tanhalpha)
        self.register_buffer('idx', torch.arange(N))
        ks = 7
        self.receptive_field = layers * (ks - 1) + 1
        self.start_conv = nn.Conv2d(in_dim, rc, 1)
        self.filter_convs, self.gate_convs, self.skip_convs, self.g1, self.g2, self.norm = (nn.ModuleList() for _ in range(6))
        L0 = max(seq, self.receptive_field)
        for j in range(1, layers + 1):
            rf_j = 1 + j * (ks - 1)
            self.filter_convs.append(DilatedInception(rc, cc)); self.gate_convs.append(DilatedInception(rc, cc))
            self.skip_convs.append(nn.Conv2d(cc, sc, (1, L0 - rf_j + 1)))
            self.g1.append(MixProp(cc, rc, gdep, dropout, propalpha, self)); self.g2.append(MixProp(cc, rc, gdep, dropout, propalpha, self))
            self.norm.append(MTLayerNorm((rc, N, L0 - rf_j + 1)))
        self.end_conv_1, self.end_conv_2 = nn.Conv2d(sc, ec, 1), nn.Conv2d(ec, out_dim, 1)
        self.skip0 = nn.Conv2d(in_dim, sc, (1, L0))
        self.skipE = nn.Conv2d(rc, sc, (1, L0 - self.receptive_field + 1) if seq > self.receptive_field else (1, 1))

    def adjacency(self):
        return self.gc(self.idx)

    def forward(self, x):                      # x: (b, in_dim, N, 12) -> (b, 12, N, 1)
        if self.seq < self.receptive_field:
            x = F.pad(x, (self.receptive_field - self.seq, 0, 0, 0))
        adp = self.gc(self.idx)
        h = self.start_conv(x)
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        for i in range(self.layers):
            res = h
            h = torch.tanh(self.filter_convs[i](h)) * torch.sigmoid(self.gate_convs[i](h))
            h = F.dropout(h, self.dropout, training=self.training)
            skip = self.skip_convs[i](h) + skip
            h = self.g1[i](h, adp) + self.g2[i](h, adp.t())
            h = self.norm[i](h + res[..., -h.size(3):], self.idx)
        skip = self.skipE(h) + skip
        return self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))


# ================================================================ STAEformer
class Attention(nn.Module):
    def __init__(self, d, heads, owner=None):
        super().__init__()
        self.h, self.hd, self.owner = heads, d // heads, [owner]
        self.FC_Q, self.FC_K, self.FC_V, self.out_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):                      # attention over dim -2
        B = x.shape[0]
        q, k, v = self.FC_Q(x), self.FC_K(x), self.FC_V(x)
        q, k, v = (torch.cat(torch.split(t, self.hd, dim=-1), 0) for t in (q, k, v))
        owner = self.owner[0]
        att = torch.softmax(q @ k.transpose(-1, -2) / self.hd ** 0.5, -1)
        if owner is not None:
            o = override(owner, att)
            att = att if o is None else o
        if owner is not None and owner.record is not None:
            owner.record.append(summarise(att.detach()))
        out = torch.cat(torch.split(att @ v, B, dim=0), -1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d, ff, heads, dropout, owner=None):
        super().__init__()
        self.attn = Attention(d, heads, owner)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(inplace=True), nn.Linear(ff, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, dim):
        x = x.transpose(dim, -2)
        out = self.ln1(x + self.d1(self.attn(x)))
        out = self.ln2(out + self.d2(self.ff(out)))
        return out.transpose(dim, -2)


class STAEformer(nn.Module):
    """Official PEMS config: input/tod/dow embeddings 24 each, adaptive embedding 80 (per step and node, xavier),
    feed-forward 256, 4 heads, 3 temporal + 3 spatial layers, dropout 0.1, mixed output projection."""

    def __init__(self, N, T=12, H=12, input_dim=3, ie=24, te=24, de=24, ae=80, ff=256, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.uniform, self.fixed, self.record, self.N, self.T, self.H = False, None, None, N, T, H
        self.input_proj = nn.Linear(input_dim, ie)
        self.tod_embedding, self.dow_embedding = nn.Embedding(288, te), nn.Embedding(7, de)
        self.adaptive_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(T, N, ae)))
        d = ie + te + de + ae
        self.attn_t = nn.ModuleList(SelfAttentionLayer(d, ff, heads, dropout) for _ in range(layers))
        self.attn_s = nn.ModuleList(SelfAttentionLayer(d, ff, heads, dropout, owner=self) for _ in range(layers))
        self.output_proj = nn.Linear(T * d, H)

    def forward(self, x):                      # x: (B, T, N, 3) = flow (normalised), tod (fraction), dow (0..6)
        B = x.shape[0]
        feats = [self.input_proj(x), self.tod_embedding((x[..., 1] * 288).long().clamp(0, 287)),
                 self.dow_embedding(x[..., 2].long().clamp(0, 6)), self.adaptive_embedding.expand(B, -1, -1, -1)]
        h = torch.cat(feats, -1)
        for l in self.attn_t:
            h = l(h, 1)
        for l in self.attn_s:
            h = l(h, 2)
        out = self.output_proj(h.transpose(1, 2).reshape(B, self.N, -1))
        return out.transpose(1, 2)             # (B, H, N)


# ================================================================ ASTGCN (recent component)
def cheb_polynomials(A, K):
    """Official scaled_Laplacian + cheb_polynomial: L = D - W, L~ = 2 L / lambda_max - I, T_k recursion."""
    D = np.diag(A.sum(1)); L = D - A
    lam = np.linalg.eigvals(L).real.max()
    Lt = 2 * L / lam - np.eye(A.shape[0])
    T = [np.eye(A.shape[0]), Lt.copy()]
    for _ in range(2, K):
        T.append(2 * Lt @ T[-1] - T[-2])
    return T


class SpatialAttention(nn.Module):
    def __init__(self, C, N, T, owner):
        super().__init__()
        self.W1, self.W2, self.W3 = nn.Parameter(torch.empty(T)), nn.Parameter(torch.empty(C, T)), nn.Parameter(torch.empty(C))
        self.bs, self.Vs = nn.Parameter(torch.empty(1, N, N)), nn.Parameter(torch.empty(N, N))
        self.owner = [owner]

    def forward(self, x):                      # (b, N, C, T) -> (b, N, N), softmax over dim 1 (official)
        lhs = (x @ self.W1) @ self.W2
        rhs = (self.W3 @ x).transpose(-1, -2)
        S = F.softmax(self.Vs @ torch.sigmoid(lhs @ rhs + self.bs), dim=1)
        o = self.owner[0]
        ov = override(o, S)
        if ov is not None:
            S = ov
        if o.record is not None:
            o.record.append(summarise(S.detach().transpose(1, 2)))   # softmax was over dim 1: rows = transpose
        return S


class TemporalAttention(nn.Module):
    def __init__(self, C, N, T):
        super().__init__()
        self.U1, self.U2, self.U3 = nn.Parameter(torch.empty(N)), nn.Parameter(torch.empty(C, N)), nn.Parameter(torch.empty(C))
        self.be, self.Ve = nn.Parameter(torch.empty(1, T, T)), nn.Parameter(torch.empty(T, T))

    def forward(self, x):
        lhs = (x.permute(0, 3, 2, 1) @ self.U1) @ self.U2
        rhs = self.U3 @ x
        return F.softmax(self.Ve @ torch.sigmoid(lhs @ rhs + self.be), dim=1)


class ChebConvSAt(nn.Module):
    def __init__(self, K, cheb, cin, cout):
        super().__init__()
        self.K = K
        self.register_buffer('cheb', torch.stack(cheb))
        self.Theta = nn.ParameterList(nn.Parameter(torch.empty(cin, cout)) for _ in range(K))

    def forward(self, x, satt):                # x (b, N, C, T), satt (b, N, N)
        outs = []
        for t in range(x.shape[-1]):
            g = x[..., t]
            o = 0
            for k in range(self.K):
                o = o + (self.cheb[k].mul(satt)).permute(0, 2, 1).matmul(g).matmul(self.Theta[k])
            outs.append(o.unsqueeze(-1))
        return F.relu(torch.cat(outs, -1))


class ASTGCNBlock(nn.Module):
    def __init__(self, cin, K, nf, nt, stride, cheb, N, T, owner):
        super().__init__()
        self.TAt, self.SAt = TemporalAttention(cin, N, T), SpatialAttention(cin, N, T, owner)
        self.cheb = ChebConvSAt(K, cheb, cin, nf)
        self.time_conv = nn.Conv2d(nf, nt, (1, 3), stride=(1, stride), padding=(0, 1))
        self.residual_conv = nn.Conv2d(cin, nt, (1, 1), stride=(1, stride))
        self.ln = nn.LayerNorm(nt)

    def forward(self, x):                      # (b, N, C, T)
        b, N, C, T = x.shape
        xt = (x.reshape(b, -1, T) @ self.TAt(x)).reshape(b, N, C, T)
        g = self.cheb(x, self.SAt(xt))
        tc = self.time_conv(g.permute(0, 2, 1, 3))
        r = self.residual_conv(x.permute(0, 2, 1, 3))
        return self.ln(F.relu(r + tc).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)


class ASTGCN(nn.Module):
    """Official PEMS config: 2 blocks, K = 3, 64 Chebyshev / 64 time filters, stride 1, in_channels 1, 12 -> 12;
    road adjacency A[i, j] = 1 for each edge of the distance file; xavier / uniform init."""

    def __init__(self, N, A, T=12, H=12, cin=1, blocks=2, K=3, nf=64, nt=64, sensor_emb=0):
        super().__init__()
        self.uniform, self.fixed, self.record = False, None, None
        # identity test (2026-11-08): optional learned sensor embedding of size `sensor_emb`, fed as extra input channels
        self.sensor_emb = nn.Parameter(torch.empty(N, sensor_emb)) if sensor_emb else None
        cin = cin + sensor_emb
        cheb = [torch.tensor(c, dtype=torch.float32) for c in cheb_polynomials(A, K)]
        self.blocks = nn.ModuleList([ASTGCNBlock(cin, K, nf, nt, 1, cheb, N, T, self)] +
                                    [ASTGCNBlock(nt, K, nf, nt, 1, cheb, N, T, self) for _ in range(blocks - 1)])
        self.final_conv = nn.Conv2d(T, H, (1, nt))
        for p in self.parameters():
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.uniform_(p)

    def forward(self, x):                      # x: (b, N, 1, T) -> (b, H, N)
        if self.sensor_emb is not None:
            e = self.sensor_emb[None, :, :, None].expand(x.shape[0], -1, -1, x.shape[-1])
            x = torch.cat([x, e], 2)
        for blk in self.blocks:
            x = blk(x)
        return self.final_conv(x.permute(0, 3, 1, 2))[..., -1]
