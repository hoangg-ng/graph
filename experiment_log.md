# Experiment log: from multi-scale MAGE to what adaptive graphs learn

The important results, one entry per experiment and oldest first. Full design, tables and discussion are in `research_design.md`. The original proposal is summarised under "Archived documents" below; the main findings are in `README.md`.

Metric: masked test MAE on traffic flow, mean ± sd over 5 seeds unless stated. "4/5" means better in 4 of 5 seeds, paired by seed. Decision rule throughout: an idea "helps" only with a lower mean **and** ≥ 4/5 seeds better.

Code and result files are in `MAGE_MS/`.

---

> **Repository cleanup (2026-11-08).** Training stdout logs, Python caches, the five cloned upstream repositories and the one-off launchers (`run_more`, `run_space*`, `run_step8`, `run_long`, `run_grid`, `run_hier`, `run_pems04`, `run_lite`, `summarize_*`) were removed; the launchers are in git history (commit 76a7d67 and earlier). Every run's final metrics and per-epoch curve remain in its result `.json`. Checkpoints were kept only where an analysis loads them (`diag_lambda.py`, `diag_useful.py`, `others/what_graphs_learn.py`, `identity_check.py`, `rq_test.py`) plus the official-code MAGE checkpoints; references to other `.pt` files or `logs/` below are historical.

## Archived documents (summarised here and removed on 2026-11-08)

**Original proposal (`proposal.md`, 2026-11-01; in git history).** *Research question:* does stronger spatial mixing at coarser temporal resolutions improve traffic forecasting compared with uniform, reversed or independent mixing? *Method:* 96 → 12 steps on traffic flow; average-pool the input into 5 / 10 / 20 / 40-min scales with TimeMixer-style encoding; one shared MAGE-style spatial operator applied with a different power per scale. *Variants:* Shared (1, 1, 1, 1), Proposed (1, 2, 4, 8), Reversed (8, 4, 2, 1, the key equal-compute comparison), Independent (separate operator per scale), Uniform (global mean). *Design:* PEMS04 primary, PEMS08 confirmation, chronological 60/20/20, 5 / 3 seeds, MAE / RMSE / MAPE, parameters, runtime, memory. *Outcome:* negative (step 8 and the 2026-11-02 / 2026-11-06 entries below); it led to the graph-collapse and sensor-identity findings in `README.md`.

**Reviewer comments on the proposal vs. the code (`comment.md`, 2026-11-02).** Eleven concerns and how they were resolved (details in the 2026-11-02 entry and `research_design.md`):

| concern | resolution |
|---|---|
| TimeMixer PDM blocks not implemented | true; linear per-scale encoders chosen because nonlinear encoders overfit at T = 96 (step 7) |
| spatial-mixing powers implemented incorrectly | not supported: factorised powers equal dense matrix powers (error 3e-16) |
| one unconstrained λ instead of per-expert sigmoid-bounded ones | matches MAGE's own design; the bounded variant was tested (small gain) |
| hyper-residual connections not removed | kept on purpose to reproduce MAGE exactly; the proposal did not ask to remove them |
| RMSNorm instead of LayerNorm | MAGE's own norm; documentation wording fixed |
| prediction head uses an intermediate block | MAGE's own two-decoder head |
| Uniform baseline keeps a learned negative graph | valid; a pure-uniform arm was added (`uniform0`) |
| speed and occupancy used as inputs | matches MAGE's official PEMS configuration; flow-only is a deviation from the proposal, not from MAGE |
| forecast targets overlap across split boundaries | true but negligible (11 timesteps, ~0.3 % of samples); standard LargeST protocol |
| Independent variant missing | true; dropped once the direction was closed |
| masking does not handle all-zero targets or NaN | cannot occur in these datasets (no NaN, no all-zero time step); MAPE masking correct |

**Review and verdict of an earlier design (`REVIEW.md`, `VERDICT.md`, 2026-09-22).** They concern a different, earlier plan: temporal fixed-aggregation features (FAF-style neighbourhood summaries of trends) plus FAM-style input modulation, on synthetic and traffic data. Main points: with linear summaries the plan reduces to a spatial lag of trend features; its headline result was already published by SimST (fixed one-hop and neighbour-average features + MLP rival STGNNs at 39× throughput); `P²` leaks a node's own signal (diagonal mass 0.23) unless an exact 2-hop operator is used; **wide fixed neighbourhoods converge to the global mean** (correlation 0.44), consistent with the later finding that a global average is what learned graphs compute; and the open direction it recommended was **robustness of fixed vs. learned aggregation under temporal distribution shift**, still untested here.

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

## 2026-10-31 · Step 8: original proposal, scale-dependent spatial mixing (`results/step8/`)
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
Prompted by the reviewer comments on the proposal vs. the code (see "Archived documents"). Step 8 arms on PEMS08, T = 96, 5 seeds (`results/step8/`):

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

## 2026-11-07 · Cross-model check (`MAGE_MS/others/`; all 10 runs done)
Graph WaveNet, AGCRN, MTGNN, STAEformer and ASTGCN, reimplemented from the official code (`others/reimpl.py`, `reimpl2.py`; running the cloned repositories was not permitted), each with its paper's own recipe, 12 → 12, seed 0, same test data as our MAGE runs. "Uniformity" = mean row entropy / log(support): 1.0 = every sensor takes the plain average (MTGNN: over its 20 kept links + self). "Swap" = test MAE when the learned mixing is replaced by the exact average at inference, no retraining.

| model | data | test MAE | uniformity | road neighbours vs random pair | top-5 links on roads (chance) | swap |
|---|---|---|---|---|---|---|
| MAGE (for reference) | 04 / 08 | 18.29 / 13.79 | **1.000 / 1.000** | – | – | **≤ 0.001** |
| Graph WaveNet | 08 | 14.775 | 0.965 | 0.99× | 2.8 % (1.9 %) | +5.37 |
| Graph WaveNet | 04 | 19.137 | 0.971 | 1.00× | 0.7 % (0.7 %) | +11.75 |
| AGCRN | 08 | 15.788 | 0.997 | 1.03× | 3.5 % (1.9 %) | +1.66 |
| AGCRN | 04 | 19.410 | 0.997 | 1.22× | 7.2 % (0.7 %) | +0.42 |
| MTGNN (sparse, 20 links / sensor) | 08 | 15.243 | 0.963 | 1.64× | 4.6 % (1.9 %) | +2.47 |
| MTGNN | 04 | 19.177 | 0.968 | 2.77× | 3.1 % (0.7 %) | +5.47 |
| STAEformer (spatial attention) | 08 | 13.470 | 0.889 | 1.10× | 3.2 % (1.9 %) | +28.89 |
| STAEformer | 04 | 18.258 | 0.928 | 1.05× | 1.1 % (0.7 %) | +8.24 |
| ASTGCN (attention on the road graph) | 08 | 19.050 | 0.760 | 0.93× | 0.7 % (1.9 %) | +10.21 |
| ASTGCN | 04 | 22.042 | 0.744 | 0.91× | 0.7 % (0.7 %) | +11.38 |

- **Exact collapse is MAGE-specific.** AGCRN comes close (0.997) and Graph WaveNet / MTGNN are near-uniform (0.96–0.97), but none is exactly uniform, and every model relies on its learned mixing (swap +0.4 to +28.9), unlike MAGE (≤ 0.001). (STAEformer PEMS04 was rerun after the first run ran out of memory in the analysis step; the harness now saves the model before the analysis.)
- **Learned mixing barely reflects roads.** Graph WaveNet, STAEformer and ASTGCN are at chance; AGCRN (PEMS04) and MTGNN show a weak road preference (1.2–2.8× a random pair).
- Swaps measure reliance, not necessity: whether a model trained from scratch with the average (or average + a small low-rank graph) matches the original is still untested.
- Size of the learned spatial part: MAGE 56–69 % of parameters, ASTGCN 65–84 %, STAEformer up to 56–60 % (whole spatial stack), MTGNN 5 %, Graph WaveNet 1–2 %, AGCRN < 0.5 %. Parameter savings from removing it are only possible for MAGE, ASTGCN and STAEformer.

## 2026-11-07 · What do learned graphs encode, if not roads? (`MAGE_MS/others/what_graphs_learn.py`, `others/results/what_graphs_learn.json`)
The 10 trained cross-model runs plus MAGE with the graph fix. Learned mixing matrix M vs pairwise sensor properties from the training span: road hop distance, daily-profile similarity, commute direction (AM / PM peak ratio), volume, co-fluctuation (correlation of deviations from the usual weekly pattern). "top5" = mean percentile of the property over each sensor's 5 strongest links (0.50 = random).

| model | road | daily profile | commute | volume | co-fluctuation | hub share* (08 / 04) |
|---|---|---|---|---|---|---|
| Graph WaveNet | 0.54 / 0.51 | 0.48 / 0.46 | 0.50 / 0.48 | 0.48 / 0.47 | 0.54 / 0.52 | 26 % / 38 % |
| AGCRN | 0.57 / 0.57 | 0.53 / **0.70** | 0.51 / 0.61 | **0.62 / 0.70** | **0.72 / 0.66** | 50 % / 25 % |
| MTGNN | 0.62 / 0.55 | 0.59 / 0.61 | 0.51 / 0.59 | 0.63 / 0.61 | **0.73 / 0.78** | 35 % / 53 % |
| STAEformer | 0.59 / 0.58 | 0.31 / 0.28 | 0.30 / 0.19 | **0.08** / 0.23 | 0.43 / 0.40 | **100 % / 100 %** |
| ASTGCN | 0.45 / 0.44 | 0.47 / 0.52 | 0.48 / 0.57 | 0.42 / 0.40 | 0.48 / 0.43 | **80 % / 85 %** |
| MAGE, graph fix | 0.62 / 0.58 | 0.48 / 0.59 | 0.55 / 0.53 | 0.56 / 0.53 | **0.71 / 0.75** | 65 % / 68 % |

top5 percentiles shown as PEMS08 / PEMS04. *Hub share = share of all top-5 link slots that go to the 5 % most-chosen sensors (≈ 5 % if links were spread evenly).

- **Over all pairs, learned link strength is unrelated to roads, daily pattern or commute direction** (Spearman |ρ| ≤ 0.1 for most model–property pairs; largest: AGCRN with volume, ρ = 0.29–0.31).
- **The strongest links of the embedding-based graphs (AGCRN, MTGNN, MAGE graph fix) join sensors that co-fluctuate** (66–78th percentile), only weakly related to road distance (ρ = 0.21 / 0.04). Graph WaveNet's strongest links are at random on every property.
- **Every learned graph is hub-dominated**: 25–68 % of all top-5 slots go to 5 % of the sensors for the graph models, 80–100 % for the attention models (in STAEformer every sensor's 5 strongest links go to the same few sensors).
- **What the hubs are:** in MTGNN and MAGE (graph fix), and AGCRN on PEMS08, hubs track the network-wide signal better than average sensors (0.55–0.63 vs 0.34–0.46): the graph reaches the common signal through a few representative sensors. Graph WaveNet's hubs are high-volume sensors (ρ = 0.51–0.53 with volume). Attention hubs (STAEformer, ASTGCN) are neither representative nor high-volume, consistent with attention "sinks".
- **Reading:** learned spatial modules mostly learn a global summary (an exact mean in MAGE, a few representative or high-volume hubs in the graph models, a few sink sensors in attention) plus links between co-fluctuating sensors, not local road structure. 1 seed per model; hub and sink interpretations need a seed check.

## 2026-11-07 · Slim MAGE + structure, method A (grow from the mean), method B (switch after collapse), gain decomposition (`MAGE_MS/run_ab.py`)
All seed 0. Differences under ~0.05–0.08 MAE are seed noise.

**MAGE variants** (T = 96 → 12; `results/ab/`, baselines from `results/pems04`, `results/hier`, `results/lite`). Test MAE (validation MAE):

| variant | PEMS08 | PEMS04 | params (08 / 04) |
|---|---|---|---|
| MAGE | 13.785 (14.099) | 18.291 (18.009) | 1,868,750 / 2,728,014 |
| slim = average only (rank 0) | 13.791 (**14.111**) | 18.191 (**17.984**) | 718,235 / 735,771 |
| slim + fixed co-fluctuation graph | 13.873 (14.167) | 18.263 (17.965) | +3 |
| slim + fixed road graph | 13.870 (14.197) | 18.169 (17.983) | +3 |
| slim + learned rank-2 graph | 13.887 (14.184) | 18.220 (17.984) | +2k / +4k |
| slim + learned rank-8 graph | 13.853 (14.160) | 18.214 (17.991) | +8k / +15k |
| slim + learned rank-32 graph | 13.812 (14.152) | 18.290 (18.027) | +33k / +59k |
| method B: MAGE, switched to slim after epoch 1 | 13.828 (14.187) | 18.364 (18.072) | 718,235 / 735,771 after epoch 1 |

- No structure added on top of the average helps: co-fluctuation, road and learned low-rank graphs are all within noise of slim MAGE, and none improves validation consistently.
- **Method A** (start from the average, add rank 2 → 8 → 32 only while validation improves) **selects rank 0 on both datasets**: the average alone. The search costs two small models (rank 0, rank 2), never the full MAGE.
- **Method B** works mechanically (collapse detected after epoch 1 in both datasets, then trained as slim), but ends +0.04 / +0.07 above MAGE and slightly above slim trained from scratch.

**Gain decomposition, other models** (12 → 12; trained from scratch with spatial mixing off / fixed to the sensor average / learned). Test MAE:

| model | data | no mixing | average | learned | share of the spatial gain captured by the average |
|---|---|---|---|---|---|
| Graph WaveNet | 08 / 04 | 17.037 / 22.774 | 15.982 / 21.829 | **14.775 / 19.137** | 47 % / 26 % |
| AGCRN | 08 / 04 | 18.955 / 23.473 | **15.638 / 19.425** | 15.788 / 19.410 | 105 % / 100 % |
| MTGNN | 08 / 04 | 15.435 / 19.142 | **15.107 / 19.001** | 15.243 / 19.177 | > 100 % (learned no better than the average; on PEMS04 not better than no mixing) |
| STAEformer | 08 / 04 | **13.383** / 18.571 | 13.407 / 18.350 | 13.470 / **18.258** | spatial attention gives no gain on PEMS08; 71 % on PEMS04 |
| ASTGCN* | 08 / 04 | 20.974 / 26.331 | 20.498 / 25.700 | **19.050 / 22.042** | 25 % / 15 % |

*ASTGCN's attention multiplies road-graph Chebyshev filters, so "average" there means uniform attention on the road filters, not a global average; its learned gain is road-graph-based.

- **For AGCRN and MTGNN (and MAGE), the plain average gives all of the spatial gain**: their learned graphs are unnecessary. STAEformer's spatial attention gives nothing on PEMS08 and is mostly replaced by the average on PEMS04.
- **Graph WaveNet's learned graph is genuinely needed** (+1.2 / +2.7 MAE over the average), even though it is not road-aligned and its hubs are high-volume sensors. ASTGCN needs its road-based attention.

## 2026-11-08 · RQ test: does a learned graph beat the average only where local dependence remains? (`MAGE_MS/others/rq_test.py`, `others/results/rq_test.json`)
No training: saved learned and average-trained models (seed 0). Per sensor: gain = MAE(average model) − MAE(learned model), on test; "local" = leftover local dependence after regressing each sensor's anomaly on the network-average anomaly (mean |corr| with its 5 most correlated residuals, training span); "common" = share of the anomaly explained by the network average.

| model | data | ρ(gain, local) | ρ(gain, common) | mean gain: top-25 % local / bottom-25 % local / all |
|---|---|---|---|---|
| Graph WaveNet | 08 / 04 | −0.37 / +0.04 | **+0.65 / +0.47** | −0.05 / +1.62 / +1.20 ; +2.79 / +2.37 / +2.69 |
| AGCRN | 08 / 04 | −0.08 / +0.14 | +0.18 / −0.14 | gain ≈ 0 (−0.15 / +0.02 overall) |
| MTGNN | 08 / 04 | −0.07 / +0.11 | +0.05 / −0.03 | gain ≈ 0 (−0.14 / −0.18) |
| STAEformer | 08 / 04 | −0.04 / +0.12 | +0.05 / +0.07 | gain ≈ 0 (−0.06 / +0.09) |
| ASTGCN | 08 / 04 | −0.09 / +0.08 | **+0.32 / +0.26** | +1.23 / +1.28 / +1.45 ; +3.96 / +3.36 / +3.66 |

- **The local-dependence hypothesis is not supported**: where the learned graph helps, the gain is unrelated to leftover local dependence (|ρ| ≤ 0.14; Graph WaveNet on PEMS08 −0.37, the wrong sign).
- **Where it helps (Graph WaveNet, ASTGCN), it helps most on sensors that strongly follow the network signal** (ρ = +0.26 to +0.65). Reading: the graph gives each sensor its own coupling to the common signal (a sensor-specific weighting), which models with sensor-identity embeddings already have; it is not supplying local structure. Consistent with the identity hypothesis, which the identity test still has to confirm.

## 2026-11-08 · What do sensor-identity parameters encode? (`MAGE_MS/others/identity_check.py`, `others/results/identity_check.json`)
Cross-validated ridge R² (held-out sensors) of predicting each sensor's properties (training span) from a model's per-sensor parameters. PEMS08 / PEMS04:

| per-sensor parameters | volume | commute AM/PM | daily-profile shape (PC1) | loading on network signal | network-signal share |
|---|---|---|---|---|---|
| MAGE sensor embedding | 0.77 / 0.93 | 0.15 / 0.69 | 0.78 / 0.87 | 0.51 / 0.56 | 0.63 / 0.53 |
| STAEformer adaptive embedding | 0.78 / 0.92 | 0.53 / 0.71 | 0.53 / 0.88 | 0.64 / 0.66 | 0.56 / 0.60 |
| MTGNN per-sensor normalisation parameters | 0.89 / 0.93 | 0.20 / 0.57 | 0.65 / 0.78 | 0.66 / 0.61 | 0.62 / 0.59 |
| AGCRN node embedding (dim 2 / 10) | 0.33 / 0.82 | 0.00 / 0.29 | 0.06 / 0.61 | −0.07 / 0.15 | −0.06 / 0.18 |
| MTGNN graph embeddings | ≈ 0 | ≈ 0 | ≈ 0 / 0.22 | ≈ 0 | ≈ 0 |
| Graph WaveNet graph vectors | 0.21 / 0.03 | ≈ 0 / 0.19 | 0.17 / 0.09 | 0.09 / ≈ 0 | 0.10 / ≈ 0 |
| Graph WaveNet graph rows (whom each sensor listens to) | 0.13 / 0.02 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| random vectors (reference) | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |

- **A sensor-identity embedding is a learned profile of the sensor's typical behaviour**: its traffic volume, its daily shape (morning vs evening commuter road) and how strongly it follows the network-wide signal (R² 0.5–0.93).
- MAGE, STAEformer and MTGNN all carry this profile (MTGNN hides it in its per-sensor normalisation parameters, not in its graph) and none of them needs a learned graph. AGCRN carries it on PEMS04 (embedding size 10) but barely on PEMS08 (size 2).
- **Graph WaveNet carries no such profile anywhere**: neither its graph vectors nor its graph rows encode sensor properties (R² ≈ 0). So its learned graph does not act as a hidden profile either. Working explanation: with the average, Graph WaveNet treats every sensor identically (same spatial input, shared weights); the learned graph gives each sensor a different, even if arbitrary, mixture of others, which makes sensors distinguishable (STID's "indistinguishability" argument). Tests: average + sensor embedding, and a fixed random graph.

## 2026-11-08 · Identity test and random-graph test (Graph WaveNet, ASTGCN; `MAGE_MS/others/run_identity.sh`)
Trained from scratch, seed 0. "Average + sensor embedding": mixing fixed to the sensor average plus a learned sensor embedding (Graph WaveNet: added to the hidden channels; ASTGCN: 8 extra input channels). "Random graph": Graph WaveNet's graph construction at initialisation, frozen (ASTGCN: the same matrix as fixed attention). Test MAE, PEMS08 / PEMS04:

| | Graph WaveNet | ASTGCN |
|---|---|---|
| learned | 14.775 / 19.137 | 19.050 / 22.042 |
| average | 15.982 / 21.829 | 20.498 / 25.700 |
| **average + sensor embedding** | **14.665 / 18.801** | **19.035** / 23.996 |
| fixed random graph | 15.312 / 20.106 | 21.222 / 26.510 |

- Graph WaveNet: average + embedding beats the learned graph on both datasets; a random fixed graph closes 55 % / 64 % of the average-to-learned gap. The learned graph compensates for missing sensor identity and partly just makes sensors distinguishable.
- ASTGCN: average + embedding equals learned on PEMS08 and closes 47 % of the gap on PEMS04. Its attention multiplies road-graph Chebyshev filters, so part of its PEMS04 gain is road-based; random attention is worse than uniform.
- With the earlier runs: given sensor-identity parameters, the network average matches the learned graph in 5 of 6 models; the exception is ASTGCN on PEMS04.

## 2026-11-08 · Computed sensor identity and unseen sensors (slim MAGE, 96 → 12, seed 0; `MAGE_MS/run_identity_mage.sh`, `results/idty/`)
Identity = learned embedding (default), computed profile (52 statistics per sensor from the training span, one shared linear layer, no per-sensor parameters), or none. Unseen-sensor test: 20 % of sensors excluded from the loss, the network average and early stopping. Test MAE, PEMS08 / PEMS04:

| | learned identity | no identity | computed identity |
|---|---|---|---|
| all sensors (trained on all) | 13.791 / 18.191 | 14.607 / 19.874 | 13.854 / 18.254 |
| unseen 20 % (held out) | 22.588 / 32.216 | 16.954 / 21.787 | **16.609 / 20.521** |
| seen 80 % (same models) | 13.609 / 18.083 | 14.165 / 19.712 | 13.610 / 18.198 |

- The computed profile recovers 92 % / 96 % of the learned embedding's value with no per-sensor parameters.
- On unseen sensors a learned embedding (untrained) is worse than no identity; the computed profile is best (−2 % / −6 % vs no identity) while matching the learned embedding on seen sensors.

## Not done yet (original proposal and plan)
- **Original proposal variants:** Independent (Reversed done 2026-11-02).
- **Primary dataset:** PEMS04, which the original proposal calls primary.
- **Baselines:** TimeMixer, STID, and MAGE with engineered temporal features.
- **Metrics:** unmasked versions of all metrics.
- **Step-8 extras:** the `learned`, `lazy1` and `road` arms, and the operator diagnostics.
- **Older plan steps:**
  - model 5 without calendar embeddings;
  - graph ablation on PEMS04 and at T = 12;
  - LargeST-SD data and a FaST reproduction;
  - long-horizon runs;
  - a full literature check.
