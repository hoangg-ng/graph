"""Unit tests for streams.py (step 8). Run: python test_streams.py (uses cuda:0 if available).

1. Rows of S+ and of P sum to 1; rows of S_m sum to 1 - lambda.
2. The factorised P^k Z equals the dense matrix power times Z (N = 50, float64).
3. With the same weights, arm 'mage' gives the same forward pass as MAGEX(encoder='linear').
4. Forward + backward time and peak memory of arm 'proposed' grow about linearly in N.
"""
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import models
import streams

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def margs(N, bs):
    return SimpleNamespace(model_dim=128, recur_num=16, topk=4, node_num=N, feature_dim=1, second=0, minute=288,
                           hour=0, day=7, week=0, weekday=0, month=0, quarter=0, year=0, bs=bs)


def inputs(b, T, N):
    x = torch.randn(b, T, N, 3, device=dev)
    x[..., 1] = torch.randint(0, 288, (b, T, N), device=dev) / 288
    x[..., 2] = torch.randint(0, 7, (b, T, N), device=dev) / 7
    return x


def build(arm, N, T=96, bs=4, dtype=torch.float32):
    torch.manual_seed(0)
    return streams.MAGEStream(margs(N, bs), arm=arm, node_num=N, input_dim=3, output_dim=1,
                              seq_len=T, horizon=12).to(dev, dtype)


def ops_for(model, N, b=4, dtype=torch.float32):
    bk = model.backbone1
    H = torch.randn(b, N, 128, device=dev, dtype=dtype)
    route, _ = bk.attn.selector(bk.attn_norm(H), 0)
    return streams.GraphOps(bk.attn, route), route


@torch.no_grad()
def test_rows():
    N = 170
    m = build('proposed', N)
    ops, _ = ops_for(m, N)
    one = torch.ones(4, N, 1, device=dev)
    sp, sm, pk = ops.apply(0, one), ops.apply(1, one), ops.power(one, 8)
    lam = -ops.lam.item()
    s_rows = pk + ops.lam * sm
    err = max((sp - 1).abs().max().item(), (sm - 1).abs().max().item(), (pk - 1).abs().max().item(),
              (s_rows - (1 - lam)).abs().max().item())
    print(f'[1] row sums: max |error| = {err:.2e} (lambda = {lam:.3f})')
    assert err < 1e-5


@torch.no_grad()
def test_dense_power():
    N = 50
    m = build('proposed', N, dtype=torch.float64)
    ops, route = ops_for(m, N, dtype=torch.float64)
    E1 = torch.softmax(m.backbone1.attn.E1, -1)[0]          # (e, n, dG)
    E2 = torch.softmax(m.backbone1.attn.E2, -1)[0]          # (e, dG, n)
    Sp = torch.einsum('bie,eid,edj->bij', route, E1, E2)    # dense S+, (b, n, n)
    P = 0.5 * (torch.eye(N, device=dev, dtype=torch.float64) + Sp)
    Z = torch.randn(4, N, 8, device=dev, dtype=torch.float64)
    worst = 0.0
    for k in (1, 2, 4, 8):
        dense = torch.linalg.matrix_power(P, k) @ Z
        worst = max(worst, ((ops.power(Z, k) - dense).abs().max() / dense.abs().max()).item())
    print(f'[2] factorised vs dense P^k Z, k in 1,2,4,8: max rel. error = {worst:.2e}')
    assert worst < 1e-10


@torch.no_grad()
def test_mage_equivalence():
    N, T, b = 170, 96, 4
    models.use_linear_contraction(True)
    torch.manual_seed(1)
    ref = models.MAGEX(margs(N, b), encoder='linear', node_num=N, input_dim=3, output_dim=1,
                       seq_len=T, horizon=12).to(dev).eval()
    m = build('mage', N, T, b).eval()
    sd = {k: v for k, v in ref.state_dict().items()
          if not k.startswith(('encoder.', 'decoder.', 'decoder2.'))}
    sd.update({'encoders.0.weight': ref.encoder.proj.weight, 'encoders.0.bias': ref.encoder.proj.bias,
               'decoders.0.weight': ref.decoder.weight, 'decoders.0.bias': ref.decoder.bias,
               'decoders2.0.weight': ref.decoder2.weight, 'decoders2.0.bias': ref.decoder2.bias})
    missing = m.load_state_dict(sd, strict=True)
    x = inputs(b, T, N)
    bias = torch.randn(3, 16, device=dev) * 0.01
    y0, y1 = ref(x, bias=bias), m(x, bias=bias)
    err = ((y0 - y1).abs().max() / y0.abs().max()).item()
    print(f'[3] arm mage vs MAGEX linear, same weights: max rel. difference = {err:.2e} ({missing})')
    assert err < 1e-4


@torch.no_grad()
def test_uniform_skips_learned_graph_operators():
    N = 16
    m = build('uniform', N)
    H = torch.randn(4, N, 128, device=dev)
    with patch.object(streams.GraphOps, 'apply', side_effect=AssertionError('graph operator used')):
        outputs, _ = streams.stream_block(m.backbone1, [H] * 4, 0, 'uniform')
    assert len(outputs) == 4


def test_cost():
    print('[4] arm proposed, T = 96, batch 8, forward + backward:')
    rows = []
    for N in (500, 2000, 8000):
        m = build('proposed', N, bs=8).train()
        x = inputs(8, 96, N)
        for it in range(3):
            if it == 1:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            out, _ = m(x, bias=torch.zeros(3, 16, device=dev))
            out.sum().backward()
        torch.cuda.synchronize()
        t, mem = (time.time() - t0) / 2, torch.cuda.max_memory_allocated() / 2**20
        rows.append((N, t, mem))
        print(f'    N = {N:5d}: {t * 1000:7.1f} ms/step, peak {mem:8.1f} MB')
        del m, x, out; torch.cuda.empty_cache()
    sl_t = np.polyfit(np.log([r[0] for r in rows]), np.log([r[1] for r in rows]), 1)[0]
    sl_m = np.polyfit(np.log([r[0] for r in rows]), np.log([r[2] for r in rows]), 1)[0]
    print(f'    log-log slope vs N: time {sl_t:.2f}, memory {sl_m:.2f} (1 = linear, 2 = quadratic)')
    assert sl_m < 1.3


if __name__ == '__main__':
    test_rows()
    test_dense_power()
    test_mage_equivalence()
    test_uniform_skips_learned_graph_operators()
    test_cost()
    print('all tests passed')
