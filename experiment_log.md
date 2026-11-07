# Experiment log: multi-scale MAGE study

The important results, one entry per experiment and oldest first. Full design, tables and discussion are in `research_design.md`. The proposal under test is `proposal.md`.

Metric: masked test MAE on traffic flow, mean ± sd over 5 seeds unless stated. "4/5" means better in 4 of 5 seeds, paired by seed. Decision rule throughout: an idea "helps" only with a lower mean **and** ≥ 4/5 seeds better.

Code and result files are in `MAGE_MS/`.

---

## 2026-10-18 · Step 1: MAGE reproduction
- Official MAGE code, 3 seeds: PEMS08 13.95, PEMS04 18.29. Our re-implementation (model 1): 13.96 / 18.25. They match.
- Paper reports 13.66 / 18.16. On this data copy the official code is 0.1–0.3 worse.
- Files: `results/logs/PEMS0*_official_s*.log`, `official_repro/experiments/mage/`.

## 2026-10-18 · Step 2: multi-scale encoder, 12 → 12 (`results/runs/`)

| # | model | PEMS08 | PEMS04 |
|---|---|---|---|
| 1 | MAGE (linear encoder) | **13.96 ± 0.03** | 18.25 ± 0.05 |
| 2 | pooled views → linear | 13.97 ± 0.02 | 18.30 ± 0.06 |
| 3 | same-size MLP | 13.98 ± 0.04 | **18.21 ± 0.04** |
| 4 | multi-scale, global gate | 13.98 ± 0.05 | **18.21 ± 0.04** |
| 5 | multi-scale, per-node gate (proposed) | 14.06 ± 0.02 | 18.25 ± 0.02 |

**Verdict:** the encoder does not help.
- Model 5 vs MAGE: +0.10 on PEMS08 (0/5), a tie on PEMS04.
- The per-node gate is harmful: 0/10 seeds better against the global gate.
- Multi-scale structure gives nothing beyond a same-size MLP.

## 2026-10-18 · Step 3: cost (`results/cost_*.json`, `results/logs/cost_*.log`)
- All variants are linear in N (log-log slope about 0.93). Model 5 costs +9 % time and +8 % memory over MAGE.
- **Side finding:** the released MAGE einsum is quadratic without `opt_einsum` (slope 1.63, out of memory at 16k nodes). Our contraction order fixes this.

## 2026-10-18 · Steps 4–5: routing and regimes (`results/post.json`, `results/post_summary.md`)
- The encoder does not change graph-expert routing beyond seed noise.
- No variant improves congestion onset, recovery or congested periods. Model 5 loses exactly there on PEMS04: +0.14 / +0.26 / +0.34, each 0/5.

## 2026-10-18 · Error anatomy (`results/diag*.json`)
- **Measurement noise:** MAGE (12 → 12) is near the estimated noise floor, which is an upper bound on truly irreducible error. The floor is ≈ 12.0 MAE on PEMS08 and ≈ 17.9 on PEMS04.
- **Large errors dominate:** the top 10 % of errors make up about 40 % of total error.
- **Neighbours carry some leftover signal:** residuals stay correlated between road neighbours (0.11–0.19) vs random pairs (≤ 0.04).
- **Upstream sensors lead:** fast variation propagates with about a 5-min lag in free flow. In congestion the peak lag is 0, spreading to negative lags (backward propagation). The 5-min lag may be an aggregation artefact.

## 2026-10-19 · Step 7: long input windows, 120 runs (`results/long/`)

| | T | linear MAGE | m3 (MLP) | m4 | m5 |
|---|---|---|---|---|---|
| PEMS08 | 12 | **13.880** | 13.977 | 13.904 | 13.981 |
| PEMS08 | 96 | **13.802** | 13.990 | 13.854 | 14.024 |
| PEMS08 | 288 | **13.774** | 14.345 | 14.129 | 14.157 |
| PEMS04 | 12 | 18.345 | **18.211** | 18.222 | 18.253 |
| PEMS04 | 96 | 18.270 | 18.331 | **18.221** | 18.293 |
| PEMS04 | 288 | **18.248** | 18.795 | 18.590 | 18.649 |

**Verdict:**
- **Longer input helps MAGE slightly:** −0.1 MAE.
- **Multi-scale beats the same-size MLP at T ≥ 96** (5/5), but beats MAGE only on PEMS04 at T = 96 (−0.05, 4/5).
- **Nonlinear encoders overfit at T = 288.**
- **Caveat:** linear MAGE hit the 300-epoch cap in most long-window runs.

## 2026-10-27 · Pre-check for the proposal, data only (`results/precheck_hops*.json`)
- Coarser time scales show stronger correlation between sensors at every road distance, but not a longer reach. The decay shape over 1–8 hops is the same at every scale.
- After removing the network-wide common signal, correlation fades within about 3 hops (about 1 km) at every scale.
- **Implication:** scale changes the strength of spatial coupling, not its range.

## 2026-10-31 · Step 8: proposal.md, scale-dependent spatial mixing (`results/step8/`)
Setup: PEMS08, 96 → 12. Four scale streams (5 / 10 / 20 / 40 min) share one MAGE graph, and each stream applies P^k.

Unit tests passed:
- the operator rows sum to 1;
- factorised powers equal dense powers;
- the `mage` arm reproduces MAGE exactly;
- memory is linear in N.

| variant | seeds | MAE | vs linear MAGE 13.802 (paired) |
|---|---|---|---|
| `mage` (code check) | 1 | 13.736 | seed 0 baseline 13.745 ✓ |
| Shared (1, 1, 1, 1) | 5 | 13.787 ± 0.086 | −0.015, 3/5 |
| **Proposed (1, 2, 4, 8)** | 5 | 13.770 ± 0.099 | −0.032, 3/5 |
| Uniform (global mean) | 2 | 13.962 | +0.21, 0/2 |
| No spatial mixing | 2 | 13.820 | +0.06, 0/2 |
| Reversed (8, 4, 2, 1) | 5 | 13.758 ± 0.037 | −0.043, 4/5 (run 2026-11-02) |
| Independent | – | not run | |

**Verdict:**
- **Proposed vs Shared:** −0.017, 2/5. The mixing schedule does not matter.
- **Neither beats MAGE.**
- **Separate streams vs early merge (Shared vs m4): −0.067, 4/5.** This is the only positive.
- **Uniform hurts,** so learned structure is needed.
- **The graph as a whole is worth only about 0.06 MAE** on PEMS08.
- **Seed variance doubles** (sd 0.09 vs 0.044).

## 2026-11-02 · Review follow-up: Reversed, pure uniform, λ, and where space matters
Prompted by `comment.md` (review of the proposal against the code). Step 8 arms on PEMS08, T = 96, 5 seeds (`results/step8/`):

| variant | MAE | vs linear MAGE 13.802 (paired) |
|---|---|---|
| Shared (1, 1, 1, 1) | 13.787 ± 0.086 | −0.015, 3/5 |
| Proposed (1, 2, 4, 8) | 13.770 ± 0.099 | −0.032, 3/5 |
| **Reversed (8, 4, 2, 1)** | 13.758 ± 0.037 | −0.043, 4/5 |
| Pure uniform (node mean, no learned S⁻) | 13.858 ± 0.027 | +0.056 |

- **Proposed vs Reversed (C2, the key comparison): +0.012, Proposed better in 2/5. Direction does not matter.**
- Shared vs pure uniform: −0.071, 4/5. Learned structure helps modestly. The old `uniform` (mean − λS⁻, 13.962) was worse than the pure mean, so its "learned structure needed" reading was confounded.
- **λ diagnostic** (`diag_lambda.py`, `results/diag_lambda.json`): the λS⁻ term is 65–80 % of the operator output in blocks 1 and 3, and there the power schedule changes the output by only 2–8 %. Only block 2 (where λ is small or negative) shows 13–54 %. The treatment had little leverage.

**Value of space** = MAE(MAGE with spatial operator set to 0, arm `nomix1`) − MAE(MAGE), 2 seeds (`results/space/`, `run_space*.py`, `run_more.py`):

| dataset | T → H | value of space | MAGE paper w/o SE (T = 12) |
|---|---|---|---|
| PEMS03 (flow only) | 12 → 12 | +0.39 (+0.64 / +0.13) | +0.44 |
| PEMS03 (flow only) | 96 → 12 | +0.13 | – |
| PEMS04 | 96 → 12 | **+0.43** | +0.08 |
| PEMS07 (flow only) | 96 → 12 | **+0.41** | +0.04 |
| PEMS08 | 12 → 12 | +0.09 | +0.43 |
| PEMS08 | 96 → 12 | +0.22 | – |
| PEMS08 | 96 → 48 | ≈ 0 | – |

- Space matters in plain MAGE (0.1–0.4). The earlier "space worth 0.03–0.05" came from the 4-stream model: there removing mixing costs only 0.03 vs 0.22 in plain MAGE, so the scale streams recover most of what space gives (2 seeds, PEMS08 only).
- "Long history replaces space" is not supported: PEMS08 goes the other way (0.09 → 0.22), PEMS03 the expected way but with noisy seeds.
- At 4 h ahead space is worth nothing on PEMS08.

## 2026-11-06 · Step 8 screen on PEMS04, 1 seed (`MAGE_MS/run_pems04.py`, `results/pems04/`)
One A100 (achilleas GPU 3), TF32, 4 jobs at once. All arms and the MAGE baseline in the same setting. Test MAE, seed 0:

| arm | MAGE's λ | bounded λ (sigmoid, per expert) |
|---|---|---|
| MAGE (baseline) | 18.291 | – |
| Shared (1, 1, 1, 1) | 18.319 | 18.188 |
| Proposed (1, 2, 4, 8) | 18.253 | **18.110** |
| Reversed (8, 4, 2, 1) | 18.236 | 18.214 |
| 4 streams, no space | 18.579 | – |

- Screen rule (beat Shared and Reversed by > 0.1): **STOP for both λ forms.** With bounded λ, Proposed beats Reversed by 0.104 but Shared by only 0.078.
- With MAGE's λ the PEMS08 picture repeats: Proposed ≈ Reversed (+0.017).
- **New signal:** the bounded λ improves every arm (Shared −0.13, Proposed −0.14, Reversed −0.02) and gives the best number so far, 0.18 below MAGE. One seed (PEMS04 seed sd ≈ 0.08), so not established.
- Streams vs space: removing space costs 0.26 in the 4-stream model vs 0.40 in plain MAGE. Partial substitution, much weaker than on PEMS08 (0.03 vs 0.22).

## 2026-11-06 · Part A: is bounding MAGE's λ a fix? (PEMS04, plain MAGE, 5 seeds, A100 + TF32)

| | MAE | RMSE | MAPE |
|---|---|---|---|
| MAGE | 18.275 ± 0.032 | 29.977 | 12.65 |
| MAGE, bounded λ (sigmoid, per expert) | **18.232 ± 0.023** | **29.769** | **12.58** |
| paired Δ | −0.043 ± 0.037, better **5/5** | −0.21, better 5/5 | −0.07 |

- Passes the rule (lower mean, ≥ 4/5 seeds), but the MAE gain is below the ~0.05 interpretability threshold. Consistent but small; RMSE gain is clearer (−0.7 %). Exploratory finding (not preregistered), one dataset.
- In the screen, Proposed + bounded λ was −0.18 vs MAGE on seed 0; bounding λ alone gives −0.05 on that seed. So most of that screen gain is not from the bound alone. Part B (scale streams + bounded λ, seeds 1–4) tests the combination.

## 2026-11-06 · Finding: MAGE's graph experts collapse to the network-wide mean
Checked the trained graph parameters (E1, E2) in our MAGE runs (PEMS04 T = 96; PEMS08 T = 12 and 96) and in the **unmodified official code's own checkpoints** (`official_repro/experiments/mage/{4,8}/final_model_s0-2.pt`, T = 12, the paper's setting).
- E1 logits end at 1e-6 to 1e-41 (start: U(0, 1)); E2 logits at most 0.01–0.1. After the softmax every hub-to-sensor weight is uniform (entropy 1.0000 of max).
- So every graph expert is (1/N)·11ᵀ: the "mixture of adaptive graph experts" is a **global mean over sensors**, and routing among experts changes nothing. The spatial term is (1 − λ)·mean.
- Likely cause (not yet tested): logits start near 0, so the softmax is near-uniform and the gradient on E1/E2 is tiny; Adam with coupled L2 weight decay (1e-4, as in the official `main.py`) then shrinks them to zero.
- Consistent with earlier results: space is worth 0.1–0.4 MAE (= the common network-wide shock, as in the pre-check); depth powers P^k could not matter (powers of a mean operator only reweight the mean).
- Per-sensor embedding (`spatial_prompt`, 128-d) needs 28–31 / 33–38 / 46–50 dims for 90 / 95 / 99 % energy: ~40 dims of sensor-specific information.

## 2026-11-07 · Screen: graph-collapse fix + rank hierarchy (1 seed, `MAGE_MS/run_hier.py`, `results/hier/`)

| | PEMS08 test MAE | vs MAGE | PEMS04 test MAE | vs MAGE | epochs | graph entropy |
|---|---|---|---|---|---|---|
| MAGE as released | 13.785 | – | 18.291 | – | ~300 | 1.000 (collapsed) |
| + graph fix | 15.979 | +2.19 | 20.449 | +2.16 | 43–47 | 0.75–0.86 |
| + graph fix + hierarchy (128 / 32 / 32) | 15.644 | +1.86 | 19.589 | +1.30 | 41 | 0.76–0.85 |
| + both + bounded λ | 15.881 | +2.10 | 19.807 | +1.52 | 61–62 | 0.36–0.81 |

- **Much worse.** The fix keeps the graph structured, but the model **overfits**: on PEMS08 train MAE reaches 12.8 by epoch 40 (MAGE: 13.6 after 295 epochs) while val rises from 16.2 to 17.4, so early stopping ends the runs at epoch 41–62.
- So in released MAGE the collapse to the sensor mean acts as **implicit regularisation**: a free per-sensor graph (~6k parameters per sensor, no decay) memorises 2 months of training data.
- The hierarchy reduces the damage (PEMS04 +1.30 vs +2.16) but does not prevent it. Untested: the hierarchy *without* the graph fix (keeps the mean graph as the general level), and a regularised low-rank structured graph.

- **The fixed graph is not road-meaningful** (expert-averaged positive graph vs the ASTGNN road edge list): road neighbours get 0.97–1.36× the weight of a random pair; the 5 strongest links per sensor are road neighbours 0.6–3.4 % of the time vs 0.7–1.9 % by chance. The structure is mostly the random init, used to fit noise, hence the overfitting. A meaningful graph needs a prior (road edges), not free learning.

## 2026-11-07 · Fair test: rank hierarchy alone on released MAGE (1 seed, A100 + TF32)
Only change: encoder and both decoders become general (full rank) + time level (rank 32) + sensor level (rank 32); graph, λ, data and recipe as released.

| | MAGE test / val | + hierarchy test / val | Δ test | epochs |
|---|---|---|---|---|
| PEMS08 | 13.785 / 14.099 | 13.932 / 14.160 | **+0.147** | 295 / 204 |
| PEMS04 | 18.291 / 18.009 | 18.335 / 17.889 | **+0.044** | 300 / 248 |

- No gain: worse on test on both, better validation only on PEMS04. Early stopping at 204 / 248 epochs suggests mild overfitting. The graph stays collapsed (entropy 1.000).

## 2026-11-07 · Data check: does a long window see rush-hour changes a 1-hour window misses?
Anomaly = flow − the sensor's usual flow at that 5-min slot of the week (usual from the first 60 %), standardised. All four PEMS sets are very similar:
- Anomaly autocorrelation: 0.48–0.51 at 1 h, 0.36–0.39 at 2 h, 0.16–0.22 at 4 h, 0.10–0.12 at 8 h, 0.16–0.28 at 24 h.
- A heavier-than-usual morning rush predicts a heavier evening rush the same day: correlation 0.40–0.57.
- But an 8-hour window adds almost nothing over a 1-hour window for predicting the anomaly (linear R², next hour: 0.60–0.78 vs 0.60–0.79; hour 4 ahead: 0.15–0.35 vs 0.16–0.36). The day's deviation is already visible in the last hour.
- So on PEMS, long-window / coarse-scale information is redundant with the last hour; most of the 4-hour-ahead anomaly is not predictable from flow history at all. Consistent with step 7 (T 12 → 96 → 288 helps MAGE by ≤ 0.1).

## 2026-11-07 · Which of MAGE's parameters do useful work? (`MAGE_MS/diag_useful.py`, `results/useful/`)
Trained MAGE, seed 0, T = 96 → 12. Parameter share per component, and test MAE when that component is switched off at inference (no retraining).

| component | params PEMS04 / PEMS08 | switched off: Δ MAE PEMS04 / PEMS08 |
|---|---|---|
| graph experts E1, E2 → **replaced by the exact sensor mean** | **69.1 % / 55.9 %** | **−0.001 / +0.001** |
| feed-forward (SwiGLU), 3 blocks | 21.7 % / 31.7 % | +4.2 to +9.6 / +3.0 to +3.9 per block |
| λ + outer projection (outer1 never used) | 3.7 % / 5.3 % | – |
| encoder | 2.3 % / 3.3 % | – |
| sensor embedding | 1.4 % / 1.2 % | +14.0 / +8.3 |
| time-of-day / day-of-week embeddings | 1.4 % / 2.0 % | +0.7 / +1.1 (tod), +1.2 / +3.6 (dow) |
| router | 0.2 % / 0.3 % | (no effect: all experts identical) |
| decoders | 0.1 % / 0.2 % | +14.3 / +11.7 (final), +96 / +100 (block 1) |

- Replacing the graph experts by the exact mean changes test MAE by ≤ 0.001: **56–69 % of MAGE's parameters do nothing** (plus the router and the unused outer projection).
- The work is done by the feed-forward layers, the sensor and calendar embeddings, the encoder/decoders and the mean-mixing term (removing all spatial mixing at inference: +27 / +18). These are ~27–41 % of the parameters.
- Inference switch-offs measure what the trained model relies on, not what a retrained model would need (retrained without space: +0.4 / +0.2).

## 2026-11-07 · Slim MAGE: graph experts replaced by c_b · sensor mean (`streams.py` arm `lite`, `results/lite/`)
Same recipe, GPU and precision as the MAGE baselines; T = 96 → 12. c_b = tanh(θ_b), one per block; E1/E2, router, λ and the outer projection removed.

| | MAGE test (params) | slim test (params) | Δ |
|---|---|---|---|
| PEMS04 seed 0 | 18.291 (2,728,014) | 18.191 (735,771) | −0.100 |
| PEMS04 seed 1 | 18.250 | 18.247 | −0.003 |
| PEMS08 seed 0 | 13.785 (1,868,750) | 13.791 (718,235) | +0.006 |

- **Same accuracy with 73 % (PEMS04) / 62 % (PEMS08) fewer parameters.** 1–2 seeds; speed and memory not yet measured cleanly (all runs shared one GPU).

## 2026-11-07 · Cross-model check, partial (in progress; `MAGE_MS/others/`)
Graph WaveNet, AGCRN, MTGNN, STAEformer and ASTGCN, reimplemented from the official code (`others/reimpl.py`, `reimpl2.py`; running the cloned repositories was not permitted), each with its paper's recipe, 12 → 12, seed 0. Finished so far (PEMS08):

| model | test MAE | mixing uniformity (1 = mean) | road neighbours vs random | top-5 links on roads (chance) | MAE with mixing → mean (inference) |
|---|---|---|---|---|---|
| Graph WaveNet | 14.775 | 0.965 | 0.99× | 2.8 % (1.9 %) | +5.37 |
| ASTGCN | 19.050 | 0.760 | 0.93× | 0.7 % (1.9 %) | +10.21 |
| AGCRN | 15.788 | 0.997 | 1.03× | 3.5 % (1.9 %) | +1.66 |

- None learns road structure. AGCRN comes closest to the mean (0.997), but none collapses exactly, and all three rely on their learned mixing at inference. So "adaptive graphs collapse to the mean" is **not** general so far; only MAGE shows it.
- Inference swaps measure reliance, not necessity: a retrain with the mixing fixed to the mean is still needed for each model. MTGNN, STAEformer and all PEMS04 runs are still running.

## Not done yet (proposal.md and plan)
- **proposal.md variants:** Independent (Reversed done 2026-11-02).
- **Primary dataset:** PEMS04, which proposal.md calls primary.
- **Baselines:** TimeMixer, STID, and MAGE with engineered temporal features.
- **Metrics:** unmasked versions of all metrics.
- **Step-8 extras:** the `learned`, `lazy1` and `road` arms, and the operator diagnostics.
- **Older plan steps:**
  - model 5 without calendar embeddings;
  - graph ablation on PEMS04 and at T = 12;
  - LargeST-SD data and a FaST reproduction;
  - long-horizon runs;
  - a full literature check.
