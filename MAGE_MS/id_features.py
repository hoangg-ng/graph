"""Computed sensor-identity features from published inductive methods (2026-11-09), as baselines for our profile.
Each returns an (N, F) tensor from the TRAINING span only; columns standardised across sensors. Sensors outside
`mask` (unseen sensors) never influence the fitted map, they are only passed through it.

  pca_embedding  : PCA embedding of "Unveiling the Inflexibility of Adaptive Embedding in Traffic Forecasting"
                   (arXiv 2411.11448, Eq. 2): each day of a sensor is a 288-slot vector; PCA fitted on training
                   sensors' days; a sensor's embedding = its day vectors projected and averaged over training days.
                   k = 8 (the paper's best for STAEformer on PEMS03).
  scpt_embedding : SCPT-style spatial encoder (Prabowo et al., arXiv 2305.05237, Sec. 3.3-3.4): three convolutions
                   (kernels 13, 12, 24; strides 1, 12, 24), mean / std / max pooling, FC -> 32; pre-trained with the
                   NT-Xent contrastive loss on two stochastic 2-day views per training sensor, then frozen. A sensor's
                   embedding = the encoder averaged over all 2-day windows of its training span. (SCPT's DCT
                   decoupling and gated adapter belong to its forecaster and are not used: here the encoder output
                   only replaces the sensor identity, exactly like our profile.)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _standardise(E):
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    return torch.tensor(np.nan_to_num(E), dtype=torch.float32)


def pca_embedding(data, train_end, mask=None, k=8):
    x = data[:train_end, :, 0]                                   # normalised flow, (T, N)
    slot = np.rint(data[:train_end, 0, -2] * 288).astype(int) % 288
    s0 = int(np.argmax(slot == 0))                               # align days to midnight
    D = (train_end - s0) // 288
    Z = x[s0:s0 + D * 288].reshape(D, 288, -1).transpose(0, 2, 1)   # (D, N, 288)
    fit = Z if mask is None else Z[:, mask]
    X = fit.reshape(-1, 288); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:k].T                                                 # (288, k)
    E = ((Z - mu) @ P).mean(0)                                   # (N, k)
    return _standardise(E)


class SCPTEncoder(nn.Module):
    def __init__(self, h=32, out=32):
        super().__init__()
        self.c1 = nn.Conv1d(1, h, 13, padding=6)
        self.c2 = nn.Conv1d(h, h, 12, stride=12)
        self.c3 = nn.Conv1d(h, h, 24, stride=24)
        self.fc = nn.Linear(3 * h, out)

    def forward(self, x):                                        # (B, L) -> (B, out)
        h = F.relu(self.c1(x[:, None]))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        return self.fc(torch.cat([h.mean(-1), h.std(-1), h.amax(-1)], -1))


def scpt_embedding(data, train_end, mask=None, dev='cuda', steps=2000, L=576, tau=0.1, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.tensor(data[:train_end, :, 0].T, dtype=torch.float32, device=dev)   # (N, T)
    N, T = x.shape
    tr = torch.arange(N, device=dev) if mask is None else torch.from_numpy(np.where(mask)[0]).to(dev)
    enc = SCPTEncoder().to(dev)
    head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)

    def view():                                                  # random 2-day crop + noise + 10 % masking
        st = torch.randint(0, T - L, (len(tr),), generator=g).to(dev)
        v = x[tr[:, None], st[:, None] + torch.arange(L, device=dev)]
        v = v + 0.1 * torch.randn_like(v)
        return v * (torch.rand_like(v) > 0.1)

    for it in range(steps):
        z1, z2 = F.normalize(head(enc(view())), dim=-1), F.normalize(head(enc(view())), dim=-1)
        z = torch.cat([z1, z2]); n = len(tr)
        sim = z @ z.T / tau; sim.fill_diagonal_(-1e9)
        target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(dev)
        loss = F.cross_entropy(sim, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == steps - 1:
            print(f'scpt pretrain step {it} nt-xent {loss.item():.3f}', flush=True)
    enc.eval()
    with torch.no_grad():
        W = x[:, :(T // L) * L].reshape(N, T // L, L)            # all non-overlapping 2-day windows
        E = torch.stack([enc(W[:, j]) for j in range(W.shape[1])]).mean(0).cpu().numpy()
    return _standardise(E)


def hourly_identity(data, train_end, mask, kind='raw', unseen_days=0, scale=(0.0, 1.0)):
    """2026-11-18: does dividing by the network mean give new sensors a better identity from a few days?
    24 values per sensor: log hourly flow profile ('raw') or log(sensor / network mean of the known sensors over the
    same days and hours) ('rel'). Known sensors (mask) use their whole training span; unseen sensors (~mask) use only
    the last `unseen_days` full days before the end of training (0 = whole span too). Columns standardised with the
    known sensors' statistics only."""
    mu, sd = scale
    x = data[:train_end, :, 0] * sd + mu                         # raw flow
    slot = np.rint(data[:train_end, 0, -2] * 288).astype(int) % 288
    s0 = int(np.argmax(slot == 0)); D = (train_end - s0) // 288
    X = np.clip(x[s0:s0 + D * 288], 0, None).reshape(D, 24, 12, -1).mean(2)   # (days, hours, N)
    m = X[:, :, mask].mean(2, keepdims=True)                     # network mean of known sensors, same days / hours
    A = np.log(X + 1) if kind == 'raw' else np.log((X + 1) / (m + 1))
    E = A.mean(0).T                                              # (N, 24), whole span
    if unseen_days:
        E[~mask] = A[D - unseen_days:].mean(0).T[~mask]          # new sensors: only their last k days
    E = (E - E[mask].mean(0)) / (E[mask].std(0) + 1e-6)
    return torch.tensor(np.nan_to_num(E), dtype=torch.float32)


def build(kind, data, train_end, mask=None, dev='cuda', seed=0, k=8, unseen_days=0, scale=(0.0, 1.0)):
    if kind in ('hraw', 'hrel'):
        return hourly_identity(data, train_end, np.ones(data.shape[1], bool) if mask is None else mask,
                               kind[1:], unseen_days, scale)
    if kind == 'pca':
        return pca_embedding(data, train_end, mask, k=k)
    if kind == 'scpt':
        return scpt_embedding(data, train_end, mask, dev=dev, seed=seed)
    raise ValueError(kind)


# ---------------------------------------------------------------- new-sensor identity (2026-11-16)
def day_features(kind, data, train_end, mask, flow_mean, flow_std, k=8, legacy_stats=False):
    """Per-DAY identity features over the training span, so an identity can be formed from any subset of days:
      'pca' : the PCA embedding above, one projection per day (D, N, k)
      'rel' : network-relative profile, log((sensor flow + 1) / (network mean flow + 1)) per hour (D, N, 24); the
              network mean is over the `mask` (training) sensors on the SAME day and hour, which cancels day effects
              (holiday, weather) from a short history.
    Returns dict(day=(D, N, F) float tensor, mu, sd (F,)) with identity(days) = (day[days].mean(0) - mu) / sd.
    mu / sd come from the training sensors' full-span identities (legacy_stats: all sensors, as pca_embedding did,
    to reproduce the features stored in the 2026-11-09/15 checkpoints)."""
    slot = np.rint(data[:train_end, 0, -2] * 288).astype(int) % 288
    s0 = int(np.argmax(slot == 0)); D = (train_end - s0) // 288
    x = data[s0:s0 + D * 288, :, 0]                                   # normalised flow
    if kind == 'pca':
        Z = x.reshape(D, 288, -1).transpose(0, 2, 1)                  # (D, N, 288)
        X = Z[:, mask].reshape(-1, 288); mu0 = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mu0, full_matrices=False)
        day = (Z - mu0) @ Vt[:k].T                                    # (D, N, k)
    elif kind == 'rel':
        raw = np.clip(x * flow_std + flow_mean, 0, None)
        H = raw.reshape(D, 24, 12, -1).mean(2)                        # (D, 24, N) hourly flow
        m = H[:, :, mask].mean(2, keepdims=True)
        day = np.log((H + 1) / (m + 1)).transpose(0, 2, 1)            # (D, N, 24)
    else:
        raise ValueError(kind)
    full = day.mean(0)
    ref = full if legacy_stats else full[mask]
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    t = lambda a: torch.tensor(np.nan_to_num(a), dtype=torch.float32)
    return dict(day=t(day), mu=t(mu), sd=t(sd), s0=s0, D=D)


def identity_from_days(feats, days):
    """(N, F) standardised identity from a list/array of day indices (the same days for every sensor)."""
    return (feats['day'][days].mean(0) - feats['mu']) / feats['sd']


class WindowIdentity:
    """2026-11-18 cold-start test: network-relative identity from a window of days, resampled during training.
    Per day and hour: log(sensor / network mean of known sensors) for flow, and with use_os also occupancy and speed
    (24 or 72 values per sensor). A window's identity = the average over its days (cumulative sums, so sampling is cheap).
      sample()          : training; each sensor gets a random window: whole span with prob p_full, else k ~ U{1..7}
                          days at a random position in the training span
      eval_identity(k)  : known sensors whole span; unseen sensors only their last k days (0 = whole span)
    Columns standardised with the known sensors' whole-span statistics."""

    def __init__(self, raw_path, data, train_end, mask, use_os=False, dev='cuda', p_full=0.3, kmax=7, seed=0, kind='hrel', k=8):
        r = np.load(raw_path)['data'][:train_end].astype(np.float64)        # (T, N, 3): flow, occupancy, speed
        slot = np.rint(data[:train_end, 0, -2] * 288).astype(int) % 288
        s0 = int(np.argmax(slot == 0)); D = (train_end - s0) // 288; N = r.shape[1]
        H = lambda c: np.clip(r[s0:s0 + D * 288, :, c], 0, None).reshape(D, 24, 12, N).mean(2)   # (D, 24, N)
        rel = lambda X, eps: np.log((X + eps) / (X[:, :, mask].mean(2, keepdims=True) + eps))
        if kind == 'hrel':
            parts = [rel(H(0), 1.0)] + ([rel(H(1), 1e-3), rel(H(2), 1.0)] if use_os else [])
            P = np.concatenate(parts, 1).transpose(0, 2, 1)                  # (D, N, F)
        else:   # 2026-11-18 PCA embedding (arXiv 2411.11448) per day; kind 'pcarel' = PCA of network-relative day curves
            if kind == 'pca':
                Z = data[s0:s0 + D * 288, :, 0].astype(np.float64).reshape(D, 288, N)            # normalised flow, as the paper
            else:
                F5 = np.clip(r[s0:s0 + D * 288, :, 0], 0, None).reshape(D, 288, N)
                Z = np.log((F5 + 1) / (F5[:, :, mask].mean(2, keepdims=True) + 1))
            Z = Z.transpose(0, 2, 1)                                         # (D, N, 288)
            X = Z[:, mask].reshape(-1, 288); mu5 = X.mean(0)
            _, _, Vt = np.linalg.svd(X - mu5, full_matrices=False)
            P = (Z - mu5) @ Vt[:k].T                                         # (D, N, k); window mean = projected mean curve
        full = P.mean(0)
        self.mu, self.sd = full[mask].mean(0), full[mask].std(0) + 1e-6
        self.C = torch.tensor(np.concatenate([np.zeros((1,) + P.shape[1:]), np.cumsum(P, 0)]), dtype=torch.float32, device=dev)
        self.mu_t = torch.tensor(self.mu, dtype=torch.float32, device=dev); self.sd_t = torch.tensor(self.sd, dtype=torch.float32, device=dev)
        self.D, self.N, self.mask, self.p_full, self.kmax, self.dev = D, N, mask, p_full, kmax, dev
        self.g = torch.Generator(device='cpu').manual_seed(seed)

    def _window(self, s, k):                                                # s, k: (N,) long on dev
        n = torch.arange(self.N, device=self.dev)
        return ((self.C[s + k, n] - self.C[s, n]) / k[:, None].float() - self.mu_t) / self.sd_t

    def sample(self):
        k = torch.randint(1, self.kmax + 1, (self.N,), generator=self.g)
        full = torch.rand(self.N, generator=self.g) < self.p_full
        k = torch.where(full, torch.full_like(k, self.D), k)
        s = (torch.rand(self.N, generator=self.g) * (self.D - k + 1).float()).long()
        return self._window(s.to(self.dev), k.to(self.dev))

    def eval_identity(self, unseen_days=0):
        k = torch.full((self.N,), self.D, dtype=torch.long)
        if unseen_days:
            k[torch.from_numpy(~self.mask)] = unseen_days
        return self._window((self.D - k).to(self.dev), k.to(self.dev))
