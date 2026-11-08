"""Reimplementations of Graph WaveNet (Wu et al., IJCAI 2019; adaptive-graph-only setting) and AGCRN
(Bai et al., NeurIPS 2020), written to follow the official code (nnzhan/Graph-WaveNet model.py, LeiBAI/AGCRN
model/AGCN.py, AGCRNCell.py, AGCRN.py) layer by layer: same layers, sizes, defaults and initialisation.
Written here because running the cloned repositories was not permitted. Each exposes `adjacency()` (the learned
graph) and a `uniform` switch that replaces it by the exact sensor mean 1/N at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def override(owner, like):
    """Fixed spatial mixing (2026-11-07): owner.uniform (inference swap) or owner.fixed == 'mean' -> exact sensor mean
    1/N; owner.fixed == 'self' -> identity (no spatial mixing). `like`: (..., N, N) tensor to match. None = learned."""
    if owner.uniform or getattr(owner, 'fixed', None) == 'mean':
        return torch.full_like(like, 1.0 / like.shape[-1])
    if getattr(owner, 'fixed', None) == 'self':
        return torch.eye(like.shape[-1], device=like.device, dtype=like.dtype).expand_as(like).clone()
    if getattr(owner, 'fixed', None) == 'random':      # fixed random mixing, different per sensor, never trained
        return owner.random_mix.to(like.dtype).expand_as(like).clone()
    return None


def make_random_mix(N, seed=0):
    """Graph WaveNet's own adaptive-graph construction at initialisation, frozen: softmax(relu(E1 E2)), E ~ N(0, 1)."""
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.relu(torch.randn(N, 10, generator=g) @ torch.randn(10, N, generator=g)), dim=1)


# ---------------------------------------------------------------- Graph WaveNet
class GCN(nn.Module):
    """Official `gcn`: diffusion over each support up to `order` hops, concat, 1x1 conv, dropout."""

    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super().__init__()
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, 1)
        self.dropout, self.order = dropout, order

    def forward(self, x, supports):           # x: (b, c, N, L)
        out = [x]
        for A in supports:
            x1 = torch.einsum('ncvl,vw->ncwl', x, A)          # official nconv
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = torch.einsum('ncvl,vw->ncwl', x1, A)
                out.append(x1)
        return F.dropout(self.mlp(torch.cat(out, 1)), self.dropout, training=self.training)


class GWNet(nn.Module):
    """Official defaults: blocks 4 x layers 2, kernel 2, residual/dilation 32, skip 256, end 512, dropout 0.3,
    node vectors of size 10 drawn from randn, adaptive adjacency softmax(relu(E1 E2)) (the --aptonly setting)."""

    def __init__(self, N, in_dim=2, out_dim=12, rc=32, dc=32, sc=256, ec=512, blocks=4, layers=2, k=2, dropout=0.3,
                 sensor_emb=False):
        super().__init__()
        # identity test (2026-11-08): optional STID-style learned sensor embedding added to the hidden channels
        self.sensor_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(rc, N))) if sensor_emb else None
        self.blocks, self.layers, self.uniform, self.fixed = blocks, layers, False, None
        self.start_conv = nn.Conv2d(in_dim, rc, 1)
        self.nodevec1 = nn.Parameter(torch.randn(N, 10))
        self.nodevec2 = nn.Parameter(torch.randn(10, N))
        self.filter_convs, self.gate_convs, self.skip_convs, self.bn, self.gconv = (nn.ModuleList() for _ in range(5))
        self.receptive_field = 1
        for _ in range(blocks):
            scope, d = k - 1, 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(rc, dc, (1, k), dilation=d))
                self.gate_convs.append(nn.Conv2d(rc, dc, (1, k), dilation=d))
                self.skip_convs.append(nn.Conv2d(dc, sc, 1))
                self.bn.append(nn.BatchNorm2d(rc))
                self.gconv.append(GCN(dc, rc, dropout, support_len=1))
                d *= 2; self.receptive_field += scope; scope *= 2
        self.end_conv_1 = nn.Conv2d(sc, ec, 1)
        self.end_conv_2 = nn.Conv2d(ec, out_dim, 1)

    def adjacency(self):
        A = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
        o = override(self, A)
        return A if o is None else o

    def forward(self, x):                      # x: (b, in_dim, N, L) -> (b, out_dim, N, 1)
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        x = self.start_conv(x)
        if self.sensor_emb is not None:
            x = x + self.sensor_emb[None, :, :, None]
        skip, sup = 0, [self.adjacency()]
        for i in range(self.blocks * self.layers):
            res = x
            x = torch.tanh(self.filter_convs[i](res)) * torch.sigmoid(self.gate_convs[i](res))
            s = self.skip_convs[i](x)
            skip = s + (skip[:, :, :, -s.size(3):] if torch.is_tensor(skip) else 0)
            x = self.gconv[i](x, sup)
            x = self.bn[i](x + res[:, :, :, -x.size(3):])
        return self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))


# ---------------------------------------------------------------- AGCRN
class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, owner):
        super().__init__()
        self.cheb_k, self.owner = cheb_k, [owner]           # owner holds the uniform switch (list: not a submodule)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, E):                   # x: (B, N, C), E: (N, D)
        n = E.shape[0]
        sup = F.softmax(F.relu(E @ E.t()), dim=1)
        o = override(self.owner[0], sup)
        if o is not None:
            sup = o
        sets = [torch.eye(n, device=E.device), sup]
        for _ in range(2, self.cheb_k):
            sets.append(2 * sup @ sets[-1] - sets[-2])
        sup = torch.stack(sets, 0)
        W = torch.einsum('nd,dkio->nkio', E, self.weights_pool)
        b = E @ self.bias_pool
        xg = torch.einsum('knm,bmc->bknc', sup, x).permute(0, 2, 1, 3)
        return torch.einsum('bnki,nkio->bno', xg, W) + b


class AGCRNCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, owner):
        super().__init__()
        self.hidden = dim_out
        self.gate = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim, owner)
        self.update = AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim, owner)

    def forward(self, x, state, E):            # official cell, including h = r*state + (1-r)*hc
        zr = torch.sigmoid(self.gate(torch.cat((x, state), -1), E))
        z, r = torch.split(zr, self.hidden, dim=-1)
        hc = torch.tanh(self.update(torch.cat((x, z * state), -1), E))
        return r * state + (1 - r) * hc


class AGCRN(nn.Module):
    """Official config: 2 layers, 64 hidden units, cheb_k 2, embed_dim 10 (PEMS04) / 2 (PEMS08), horizon 12,
    CNN predictor on the last hidden state; xavier-uniform init for >1-d parameters, uniform otherwise."""

    def __init__(self, N, input_dim=1, hidden=64, horizon=12, num_layers=2, embed_dim=10, cheb_k=2):
        super().__init__()
        self.uniform, self.fixed, self.N, self.hidden, self.horizon = False, None, N, hidden, horizon
        self.node_embeddings = nn.Parameter(torch.randn(N, embed_dim))
        self.cells = nn.ModuleList(AGCRNCell(input_dim if i == 0 else hidden, hidden, cheb_k, embed_dim, self)
                                   for i in range(num_layers))
        self.end_conv = nn.Conv2d(1, horizon, kernel_size=(1, hidden), bias=True)
        for p in self.parameters():
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.uniform_(p)

    def adjacency(self):
        E = self.node_embeddings
        A = F.softmax(F.relu(E @ E.t()), dim=1)
        o = override(self, A)
        return A if o is None else o

    def forward(self, x):                      # x: (B, T, N, 1) -> (B, horizon, N)
        B, T = x.shape[:2]
        cur = x
        for cell in self.cells:
            h = torch.zeros(B, self.N, self.hidden, device=x.device)
            seq = []
            for t in range(T):
                h = cell(cur[:, t], h, self.node_embeddings)
                seq.append(h)
            cur = torch.stack(seq, 1)
        out = self.end_conv(cur[:, -1:])       # (B, horizon, N, 1)
        return out[..., 0]
