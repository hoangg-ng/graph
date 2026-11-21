# Forecasting new traffic sensors relative to the network mean

When a sensor is added to a road network, a traffic forecaster has to predict it with little or no history. Most recent forecasters tell sensors apart with a learned embedding per sensor, and that embedding is untrained for a new sensor. We model every sensor relative to the network mean. The network mean is the shared state of the network at each time step. A sensor's identity is its hourly profile divided by the network mean over the same days. The model is trained on identities estimated from random short windows, so it learns to use the noisy identity that a new sensor with a few days of data actually has.

Setting: PEMS04 (307 sensors, 59 days) and PEMS08 (170 sensors, 62 days), traffic flow every 5 minutes, chronological 6:2:2 split. 20 % of the sensors (fixed random draw, rng 123) are excluded from training and used only at test time ("unseen sensors"). Metric: test MAE on the unseen sensors, lower is better. One seed per configuration unless stated. Per-run tables and the full research history: [`experiment_log.md`](experiment_log.md), [`research_design.md`](research_design.md).

---

## 1. Problem

A new sensor arrives with 0, 1, 3 or 7 days of data. The forecaster must predict it without retraining. Three facts from our experiments set up the problem.

1. A learned per-sensor embedding fails on new sensors. In slim MAGE the unseen-sensor MAE is 32.40 (PEMS04) and 22.73 (PEMS08), against 21.81 and 16.95 for the same model with no identity at all. The new sensor's embedding row never receives a training signal, and the model reads it as information.
2. Without identity the model is safe but blunt. It predicts every sensor as a typical sensor, which costs 1.6 (PEMS04) and 0.55 (PEMS08) MAE on the known sensors compared with having an identity.
3. Identity computed from data, such as the published PCA embedding, transfers only with a long history. With 1 day of data it is worse than no identity (31.25 vs 21.81 on PEMS04).

Earlier work in this repository explains why identity matters so much. In six published adaptive-graph models, the learned graph beats a plain network-wide average only when the model has no sensor identity. MAGE's 16 learned graph experts collapse to the network average within one epoch. So in these benchmarks the spatial part of a forecaster reduces to a network-wide common signal plus a per-sensor identity (details in the log, entries of 2026-11-06 to 2026-11-08).

## 2. Method

The forecast for sensor *i* at time *t* follows a factor model, *y*_it = *λ*_i · *f*_t + *e*_it, where *f*_t is the common state of the network and *λ*_i describes how sensor *i* relates to it.

| Part | What it does | Estimates |
|---|---|---|
| Network mean | each layer adds tanh(*c*_b) · mean over sensors of the hidden states, shared by every sensor | *f*_t |
| Relative profile | 24 hourly values of log(sensor flow / network mean flow of the known sensors), on the same days and hours | *λ*_i |
| Window training | every batch, each sensor's identity is recomputed from a random 1–7 day window (whole span with probability 0.3) | trains the model on noisy *λ*_i |

Dividing by the network mean cancels day effects such as weekends, holidays and weather, because the sensor and the mean see the same day. A 1-day relative profile identifies the right sensor among all sensors 66 % of the time on PEMS04, against 39 % for the raw profile (PEMS08: 54 % vs 34 %). Window training removes the gap between how identity is seen in training (exact, whole span) and at test time (noisy, a few days). Implemented in slim MAGE (`MAGE_MS/streams.py`, arm `lite`) with `--identity hrel --id_window 1`; `--id_os 1` adds occupancy and speed profiles.

## 3. Results

Every saved checkpoint is re-tested on the same unseen sensors with 0 days (identity set to the mean identity of the known sensors), the last 1, 3 or 7 days, or the full training span (`MAGE_MS/eval_history.py`). Values are means over 3–4 identically trained checkpoints; their spread is at most 0.15 for window-trained models.

PEMS04, unseen-sensor MAE:

| Method | 0 d | 1 d | 3 d | 7 d | full |
|---|---|---|---|---|---|
| learned embedding (MAGE default) | 32.40 | 32.40 | 32.40 | 32.40 | 32.40 |
| no identity | **21.81** | **21.81** | 21.81 | 21.81 | 21.81 |
| SCPT-style encoder | – | – | – | – | 21.21 |
| PCA embedding, as published | 29.72 | 31.25 | 23.20 | 20.81 | 20.09 |
| relative profile, normal training | 28.86 | 26.65 | 22.29 | 20.93 | 20.21 |
| PCA + window training | 25.24 | 22.53 | 20.91 | **20.22** | 20.12 |
| relative profile + window training (ours) | 24.61 | 22.52 | **20.82** | 20.26 | **20.05** |
| ours + occupancy and speed | 25.76 | 22.40 | **20.82** | 20.30 | 20.12 |

PEMS08, unseen-sensor MAE:

| Method | 0 d | 1 d | 3 d | 7 d | full |
|---|---|---|---|---|---|
| learned embedding (MAGE default) | 22.73 | 22.73 | 22.73 | 22.73 | 22.73 |
| no identity | **16.95** | 16.95 | 16.95 | 16.95 | 16.95 |
| SCPT-style encoder | – | – | – | – | 17.10 |
| PCA embedding, as published | 21.20 | 21.46 | 16.90 | 16.58 | 16.55 |
| relative profile, normal training | 20.51 | 18.12 | 16.19 | 16.18 | 16.11 |
| PCA + window training | 18.69 | 16.87 | 16.15 | 16.07 | 16.05 |
| relative profile + window training (ours) | 18.59 | 16.70 | **16.03** | **15.98** | 15.99 |
| ours + occupancy and speed | 19.48 | **16.62** | 16.07 | 16.01 | **15.97** |

What the tables show:
- From 3 days on, our model beats no identity by 1.0–1.5 MAE on PEMS04 and about 0.9 on PEMS08, and beats the published PCA protocol at every history length (by 2.4 at 3 days on PEMS04).
- Window training is the main lever. It lowers the 1-day error of PCA from 31.25 to 22.53 and of the relative profile from 26.65 to 22.52. Once both use window training, PCA and the relative profile are within about 0.1 of each other.
- The relative profile helps most when the identity is noisy and the model was not trained for it (1 day, normal training: 26.65 vs 30.13 for the raw profile on PEMS04).
- With 0 days every identity model is worse than no identity, and with 1 day PEMS04 still is (22.40 vs 21.81; that single day is a Sunday). A fallback for very short histories is missing.

Same idea in a second backbone (U-Trans-style STAEformer with the network mean, 12 → 12 steps). A history encoder trained with cold-start episodes (each sensor gets no history 5 % of the time, the full span 15 %, and 1–7 days otherwise) covers the 0-day case:

| Unseen-sensor MAE | 0 d | 1 d | 3 d | 7 d | full |
|---|---|---|---|---|---|
| PEMS08 no identity | 16.91 | 16.91 | 16.91 | 16.91 | 16.91 |
| PEMS08 encoder + episodes | 17.77 | **16.26** | **15.79** | **15.67** | **15.67** |
| PEMS08 same encoder, full histories only | 23.57 | 20.45 | 18.76 | 17.67 | 16.92 |
| PEMS04 no identity | **22.37** | 22.37 | 22.37 | 22.37 | 22.37 |
| PEMS04 encoder + episodes | 22.73 | **22.19** | **20.97** | **20.57** | **20.50** |
| PEMS04 same encoder, full histories only | 33.87 | 31.14 | 25.97 | 23.22 | 21.27 |

In this backbone computed identity also memorises the training sensors: plain PCA is worse than no identity on PEMS08 (17.87 vs 16.91). Gaussian noise on the identity during training fixes it (16.40), while smaller identity capacity does not.

## 4. Related work

| Work | What it does | Relation to ours |
|---|---|---|
| U-Trans, Zhang et al., arXiv 2607.12462 (2026) | replaces spatial attention with a uniform network average plus a learned adaptive embedding | same network mean; learned embedding only, one backbone, no unseen sensors |
| PCA embedding, Wang et al., arXiv 2411.11448 (2024) | node embedding = PCA of each sensor's daily curves; zero-shot across cities | our main computed-identity baseline; collapses with 1–3 days unless trained with windows |
| SCPT, Prabowo et al., arXiv 2305.05237 | contrastive pre-trained encoder gives embeddings for new roads from 2 days of data | baseline; needs pre-training; below no identity on PEMS08 in our runs |
| VaryST, arXiv 2610.19813 (2026) | node representation from about 7 days of observations, no node ID, global mean token, masks the patches the model relies on most | closest in spirit (observation-based identity plus global mean); does not test 1–3 days or compare with PCA, SCPT or no identity |
| SC-SAMG, Sensors 26(15):4995 (2026) | cold-start forecasting at unseen sensors from a short support set via meta-graph learning | closest problem setting; full text not yet read, may overlap with window training |
| EvoXXLTraffic, arXiv 2605.29768 (2026) | benchmark with growing sensor networks | reports that node-indexed embeddings collapse on new sensors, as in our learned-embedding row |
| A2TTA, arXiv 2607.25875 (2026) | test-time adaptation with new embedding rows for new sensors | needs adaptation at test time; we use none |
| ST-FiT (AAAI 2025), MoGERNN (2025) | inductive forecasting with limited data; MoGERNN uses 2 days on new roads | related settings, neighbour-based |
| IGNNK, KITS, INCREASE | kriging: estimate unobserved locations from neighbours, 0 days of own data | different task (no own history); road neighbours carried no identity information in our check (R² 0.04) |
| ST-GFSL (KDD 2022), TransGTR | few-shot transfer across cities | city-level transfer, not single new sensors |
| STID (CIKM 2022), STAEformer (CIKM 2023) | identity embeddings make simple models strong | the learned identity that fails on new sensors |
| DropoutNet, Volkovs et al., NeurIPS 2017 | trains recommenders for cold start by dropping preference inputs | same principle as window training, in recommendation |
| Common Correlated Effects, Pesaran 2006; DeepGLO, NeurIPS 2019 | cross-sectional averages as proxies for common factors; global factors plus local models | theory behind the network mean as *f*_t |

### What is new and what is not

- Not new: the network mean (U-Trans, VaryST), identity computed from data (PCA, SCPT, VaryST), and training under simulated cold start as a principle (DropoutNet, meta-learning).
- Not found in the papers above:
  - an evaluation of unseen sensors at 0 / 1 / 3 / 7 days / full against a no-identity baseline, which shows the published PCA protocol failing with short histories;
  - window training applied to computed identities in traffic forecasting;
  - identity defined relative to the network mean;
  - the memorisation diagnosis for computed identity in an attention backbone;
  - the finding that learned graphs stand in for missing sensor identity.
- Open weakness: PCA with window training ties our relative profile. The method contribution is mostly the training procedure. Making the network mean also act as the prior for a new sensor (shrink a k-day identity toward "behaves like the network mean", with weight k / (k + *τ*)) would give the relative profile a role PCA cannot play and should fix the 0- and 1-day columns. Not yet run.

## 5. Limitations and next steps

One seed for the slim-MAGE comparison (spread over identically trained checkpoints ≤ 0.15; across seeds, no identity varied by 0.06). Two datasets for the unseen-sensor study; PEMS03 and PEMS07 exist only for the full-history profile. The unseen sensors' last days are weekend-heavy (PEMS04 1 day = Sunday). Next: shrinkage toward the network mean with 0-day windows, 3 seeds, PEMS03/07, window training in the U-Trans-style backbone, VaryST and SC-SAMG as baselines.

## Code

| What | Where |
|---|---|
| Slim MAGE with network mean and identity options | `MAGE_MS/streams.py`, `MAGE_MS/train.py` (`--identity`, `--holdout`, `--unseen_days`, `--id_window`, `--id_os`, `--eval_ckpt`) |
| Identity features: PCA, SCPT-style, raw / relative profile, window sampler | `MAGE_MS/id_features.py` |
| Fair evaluation by history length | `MAGE_MS/eval_history.py` → `MAGE_MS/results/history/` |
| Launchers | `MAGE_MS/run_relative.py`, `run_coldstart.py`, `run_pca_coldstart.py`, `run_inductive.py`; several jobs per process: `batch_run.py` |
| U-Trans-style backbone, identity transfer, history encoder | `MAGE_MS/others/reimpl2.py`, `train_other2.py`, `run_idtransfer.py`, `run_newsensor.py`, `newsensor_eval.py` |
