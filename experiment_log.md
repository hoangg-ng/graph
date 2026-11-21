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

## 2026-11-08 · Profile study: does the computed sensor profile hold up? (`MAGE_MS/run_profile_study.py`, `results/profile/`)
Slim MAGE (network-mean aggregation), identity = learned embedding / computed profile / none. Test MAE; "unseen" = the 20 % of sensors excluded from training (loss, network mean, model selection). PEMS04 / PEMS08 at 96 → 12: mean of 3 seeds; other rows 1 seed.

| setting | learned | **profile** | none | unseen: learned | **unseen: profile** | unseen: none |
|---|---|---|---|---|---|---|
| PEMS04, 96 → 12 (3 seeds) | **18.209** | 18.252 | 19.928 | 32.40 | **20.36** | 21.81 |
| PEMS08, 96 → 12 (3 seeds) | **13.783** | 13.837 | 14.547 | 22.73 | **16.66** | 16.95 |
| PEMS03, 96 → 12 | 14.655 | **14.381** | 15.118 | 28.05 | **13.95** | 14.81 |
| PEMS07, 96 → 12 | 19.450 | **19.370** | 21.493 | 27.40 | **19.09** | 20.80 |
| PEMS04, 12 → 12 | 18.350 | **18.195** | 20.554 | 42.21 | **21.08** | 22.25 |
| PEMS08, 12 → 12 | 13.843 | **13.635** | 14.573 | 28.22 | 17.34 | **17.32** |

Profile from only the last few days of the training span (unseen sensors, 96 → 12, seed 0): PEMS04 7 days 20.63, 3 days 21.86 (full span 20.52); PEMS08 7 days 16.68, 3 days 17.31 (full 16.61).

- **All sensors:** the computed profile is slightly behind the learned embedding on PEMS04 / PEMS08 at 96 steps (+0.04 / +0.05, learned better in 3/3 and 2/3 seeds) but **better on PEMS03 (−0.27), PEMS07 (−0.08) and at 12 steps (−0.16 / −0.21)**. It is always far better than no identity (−0.7 to −2.4).
- **Unseen sensors:** the learned embedding fails everywhere (27–42 MAE). The profile beats no identity in 5 of 6 settings (−0.29 to −1.70; 3/3 seeds on PEMS04 and PEMS08) and ties in one (PEMS08, 12 steps).
- **Little history:** a 7-day profile is as good as the full span; 3 days falls to about the no-identity level, so a new sensor needs about one week of data.

## 2026-11-09 · U-Trans-style comparison (STAEformer + network mean; `MAGE_MS/others/run_utrans.py`, seed 0, 12 → 12)
Our STAEformer reimplementation with spatial attention fixed to the network mean (U-Trans-style; no congestion features), identity = learned adaptive embedding / computed profile / none, on all sensors and with the same 20 % unseen sensors as before. PEMS04 on GPU 2 (A6000), PEMS08 on GPU 0 (A4500). Test MAE, PEMS04 / PEMS08:

| | learned | computed profile | none |
|---|---|---|---|
| all sensors | 18.299 / **13.437** | **18.269** / 13.559 | 21.648 / 15.023 |
| unseen 20 % | 36.48 / 24.46 | 22.53 / 17.61 | **22.37 / 16.91** |
| seen 80 % (same models) | 18.31 / **13.22** | **18.10** / 13.37 | 21.52 / 14.74 |
| whole network in the holdout runs | 21.92 / 15.46 | **18.98 / 14.22** | 21.69 / 15.18 |

Our model (slim MAGE + network mean, 12 → 12, seed 0) for reference: all sensors 18.195 / 13.635 (computed profile); unseen 21.08 / 17.34 (profile), 22.25 / 17.32 (none); ~0.65 M parameters vs 1.36 M / 1.22 M for the U-Trans-style model.

- Known sensors: in the U-Trans-style backbone too, the computed profile matches the learned embedding (−0.03 / +0.12), and identity matters even more than in slim MAGE (no identity: +3.4 / +1.6).
- Unseen sensors: the learned embedding fails (36.5 / 24.5). Here the computed profile does **not** beat no identity (+0.16 / +0.70), unlike in slim MAGE; it is the best option only for the network as a whole, because no identity is much worse on the seen sensors.
- Ours vs U-Trans-style: about equal on all sensors, better on unseen sensors on PEMS04 (21.08 vs 22.53), about equal on PEMS08, with about half the parameters.

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

## 2026-11-09 · Published inductive identities vs ours on unseen sensors (PCA embedding, SCPT-style encoder)

Literature check first: SCPT (arXiv 2305.05237) already forecasts unseen roads from an embedding computed from each road's
own history; the PCA embedding (arXiv 2411.11448) is essentially our profile; EvoXXLTraffic (2605.29768) already reports
that per-node parameters fail on new sensors. None of the three trains a no-identity baseline or explains the failure
(unseen sensors' embeddings decay to 0 under weight decay). Code: `MAGE_MS/id_features.py` (both baselines, fitted on
training sensors only, same slot as the profile), `--identity pca|scpt` in `train.py` / `others/train_other2.py`,
launcher `MAGE_MS/run_inductive.py` (lerna GPUs 0 and 3). Seed 0, 20 % unseen sensors (rng 123).

Unseen-sensor MAE (seen-sensor MAE in brackets):

| backbone / data | learned | none | our profile | PCA emb. | SCPT-style |
|---|---|---|---|---|---|
| U-Trans-style PEMS04 | 36.48 (18.31) | 22.37 (21.52) | 22.53 (18.10) | **21.53** (18.13) | 23.08 (18.11) |
| U-Trans-style PEMS08 | 24.45 (13.22) | **16.91** (14.74) | 17.61 (13.37) | 17.87 (13.36) | 18.49 (13.26) |
| slim MAGE PEMS04 | 32.22 (18.08) | 21.79 (19.71) | 20.52 (18.20) | **20.09** (18.24) | 21.21 (18.40) |
| slim MAGE PEMS08 | 22.59 (13.61) | 16.95 (14.16) | 16.61 (13.61) | **16.29** (13.67) | 17.10 (13.67) |

All sensors, slim MAGE test MAE: learned 18.19 / 13.79, none 19.87 / 14.61, profile 18.25 / 13.85, PCA 18.48 / 13.98, SCPT 18.67 / 13.99.

Conclusions (1 seed; seed spread on unseen MAE was ~0.2):
- Our profile has no edge over the published PCA embedding (PCA better on unseen in 3 of 4, profile slightly better on known sensors). The computed-identity method is not novel.
- The no-identity baseline is competitive with the published inductive identities on unseen sensors: it beats SCPT in 3 of 4 settings and beats every computed identity in U-Trans PEMS08.
- Identity transfers poorly to new sensors: on known sensors computed identity gains 1.5 to 3.4 MAE over none; on unseen sensors the best gains 0 to 1.7. This "identity transfer gap" is unaddressed by the papers checked and is the open problem.

## 2026-11-15 · Why computed identity fails on unseen sensors in the U-Trans-style backbone

Hypothesis: the 80-dim identity concatenated to every token lets STAEformer memorise the training sensors. Test: PCA
identity, 20 % unseen sensors, seed 0, one change per run (`others/run_idtransfer.py`, controls in
`reimpl2.STAEformer`, flags `--pca_k --id_dim --id_noise --id_mix --id_drop --id_mode` in `train_other2.py`;
achilleas GPUs 6 and 7). Gain = MAE of the no-identity model minus MAE of the variant (positive = identity helps).

| variant | PEMS08 unseen / seen MAE | PEMS08 gain unseen / seen | PEMS04 unseen / seen MAE | PEMS04 gain unseen / seen |
|---|---|---|---|---|
| no identity | 16.91 / 14.74 | 0 / 0 | 22.37 / 21.52 | 0 / 0 |
| learned | 24.45 / 13.22 | -7.54 / +1.53 | 36.48 / 18.31 | -14.11 / +3.21 |
| PCA k 8 (baseline) | 17.87 / 13.36 | -0.96 / +1.38 | 21.53 / 18.13 | +0.84 / +3.39 |
| k 2 | 16.71 / 13.53 | +0.20 / +1.22 | 21.82 / 19.28 | +0.55 / +2.24 |
| k 4 | 18.12 / 13.30 | -1.21 / +1.45 | 22.88 / 18.42 | -0.51 / +3.10 |
| 8-wide bottleneck | 17.90 / 13.35 | -0.99 / +1.39 | 21.59 / 18.13 | +0.78 / +3.38 |
| **noise 0.5** | **16.40 / 13.07** | **+0.52 / +1.68** | **20.67** / 18.65 | **+1.70** / +2.87 |
| mix 0.5 | 16.64 / 13.00 | +0.27 / +1.74 | 21.03 / 18.46 | +1.34 / +3.06 |
| drop 0.3 | 18.25 / 13.14 | -1.34 / +1.60 | 21.07 / 18.15 | +1.30 / +3.37 |
| affine (output scale/shift only) | 16.55 / 14.51 | +0.37 / +0.24 | 21.78 / 21.03 | +0.59 / +0.49 |

Reading (1 seed):
- Restricting capacity (k 4, 8-wide bottleneck) does nothing; k 2 trades known-sensor accuracy for a little transfer.
  The affine pathway transfers but loses almost all of the known-sensor gain. So it is not a capacity problem.
- Perturbing the identity during training fixes it: noise 0.5 turns the unseen gain positive on PEMS08 (-0.96 -> +0.52,
  with known sensors also better) and doubles it on PEMS04 (+0.84 -> +1.70, known sensors -0.5). Mixing helps on both
  (+0.27 / +1.34 unseen) but less than noise; dropout helps on PEMS04 but not PEMS08.
- Noise-regularised PCA identity beats no identity and plain PCA on unseen sensors on both datasets. Consistent with
  memorisation of exact identity values, not with too much identity capacity.

## 2026-11-16 · New sensors with a short history: computed vs fitted identity, cold-start training

Setting: U-Trans-style STAEformer, seed 0, 20 % unseen sensors (rng 123). A new sensor has only the LAST k training days
(k = 1, 3, 7) plus its live input window. `others/newsensor_eval.py` rebuilds only the unseen sensors' identity from those
k days (computed) or fits it on forecasting error over those days with the model frozen (fitted; best of starting from
the computed value or from 0). Code: `id_features.day_features` (`rel` = log(sensor / network mean) per hour, network
mean over known sensors on the same days), `--identity rel`, `--id_resample k` (identity recomputed from k random days
every training step), learned-attention holdout masking in `reimpl2.Attention`; launchers `others/run_newsensor.py`
(NEWSENSOR_STAGE 1/2/3), lerna GPUs 0-3. CPU check first: the relative profile identifies a sensor among all sensors
from 1 day 66 % vs 39 % (raw, PEMS04), 54 % vs 34 % (PEMS08).

Unseen-sensor MAE, computed identity (fitted in brackets); no identity = 16.91 (PEMS08) / 22.37 (PEMS04):

| model | PEMS08 1 d | 3 d | 7 d | seen | PEMS04 1 d | 3 d | 7 d | seen |
|---|---|---|---|---|---|---|---|---|
| mean + learned emb. | 23.14 (20.32) | (16.44) | (16.11) | 13.18 | 34.67 (41.28) | (23.09) | (22.24) | 18.21 |
| attention + learned emb. | 24.06 (21.09) | (17.24) | (16.24) | 13.32 | 36.02 (37.14) | (23.24) | (22.14) | 18.62 |
| mean + PCA | 25.99 (18.98) | 18.89 (15.87) | 17.62 (15.68) | 13.36 | 45.35 (30.90) | 26.35 (21.04) | 23.16 (20.49) | 18.13 |
| attention + PCA | 29.21 (18.88) | 18.81 (15.71) | 18.20 (15.75) | 13.41 | 36.96 (29.92) | 25.66 (21.42) | 22.64 (20.95) | 18.12 |
| mean + PCA + noise 0.5 | 17.56 (16.89) | 16.50 (15.50) | 16.43 (15.36) | 13.07 | 23.88 (23.28) | 21.98 (20.44) | 20.92 (20.19) | 18.65 |
| mean + relative | 20.04 (20.20) | 16.81 (16.81) | 16.92 (15.92) | 13.21 | 31.39 (31.39) | 24.20 (22.08) | 23.50 (20.99) | 18.12 |
| **mean + relative, trained on 3-day windows** | 16.84 (17.23) | **15.70** (15.70) | 15.61 (15.20) | **13.04** | 23.13 (23.13) | **21.00** (20.93) | 20.57 (20.46) | 18.34 |
| mean + relative, 7-day windows | 17.69 | 15.86 (15.63) | 15.72 (15.22) | 12.97 | 24.79 | 21.31 | 20.56 | 18.18 |
| mean + PCA, 7-day windows | 17.60 (16.89) | 16.54 (15.52) | 16.18 (15.29) | 12.94 | 26.19 (25.96) | 21.75 (20.51) | 21.04 (20.42) | 18.23 |
| mean + relative + noise 0.5 | **16.13** | 15.97 (15.68) | 15.97 (15.56) | 13.62 | **22.82** | 21.26 (20.50) | 20.85 (20.45) | 19.21 |

Reading (1 seed):
- Fitting a standard learned embedding on the new sensor's days barely beats no identity (3 d: 16.44 / 23.09). Not "just retraining".
- Training under simulated short history (identity from random 3-day windows) gives the best computed identity: with 3 days
  it beats no identity by 1.2 / 1.4 MAE and beats PCA from the full history, with no test-time fitting and no loss on known
  sensors. Test-time fitting adds little on top of it.
- Relative (network-normalised) beats PCA under the same window training by 0.3-0.7.
- Learned attention vs network mean with the same identity: about equal on new sensors (PCA 3 d 18.81 vs 18.89, 25.66 vs
  26.35); the network mean is not what makes cold start work, but it is as good and 1.7x faster per epoch on PEMS04.
- Similar-sensor mixing (stage 3; half the heads read the mean of each sensor's 10 most correlated training sensors over
  the last 7 training days, chosen the same way for new sensors). Unseen / seen MAE, network mean -> mean + similar-10:
  PEMS08 no identity 16.91 -> 17.51 / 14.74 -> 13.89; PEMS08 PCA + noise 16.40 -> 16.87 / 13.07 -> 12.97;
  PEMS04 no identity 22.37 -> 21.76 / 21.52 -> 19.35; PEMS04 PCA + noise 20.67 -> 21.49 / 18.65 -> 18.17;
  learned identity, all sensors 13.44 -> 13.43 (PEMS08), 18.30 -> 18.40 (PEMS04).
  Worse on new sensors in 3 of 4; it helps known sensors mainly when the model has no identity (it acts as one), and adds
  nothing in the normal setting. The linear check overstated it (it ignores the sensor's own live input). Dropped.

## 2026-11-16 · Trained history encoder with cold-start episodes

Identity `enc` (`reimpl2.STAEformer`): each day of a sensor's history = 24 hourly log(sensor / network mean) values +
day-of-week one-hot -> shared MLP -> mean over the days the sensor has -> MLP (with log(1 + #days)) -> identity (80).
Trained end to end with the forecaster (network-mean backbone). Cold-start episodes: every step each sensor gets a random
history (5 % none, 15 % full, 80 % 1-7 consecutive days). A new sensor at test time: encoder on its last k days, nothing
fitted. Ablation `--enc_episodes 0`: always the full history while training. Launcher stage 4, lerna GPUs 0-3, seed 0.

Unseen-sensor MAE by days of history (seen-sensor MAE last):

| model | 0 d | 1 d | 2 d | 3 d | 5 d | 7 d | full | seen |
|---|---|---|---|---|---|---|---|---|
| PEMS08 no identity | 16.91 | | | | | | | 14.74 |
| PEMS08 fixed relative, 3-day windows | 20.34 | 16.84 | | 15.70 | | 15.61 | 15.52 | 13.04 |
| **PEMS08 encoder + episodes** | 17.77 | **16.26** | 15.91 | 15.79 | 15.73 | 15.67 | 15.67 | **12.95** |
| PEMS08 encoder, no episodes | 23.57 | 20.45 | 18.98 | 18.76 | 18.58 | 17.67 | 16.92 | 13.26 |
| PEMS04 no identity | 22.37 | | | | | | | 21.52 |
| PEMS04 fixed relative, 3-day windows | 27.01 | 23.13 | | 21.00 | | 20.57 | 20.35 | 18.34 |
| **PEMS04 encoder + episodes** | 22.73 | **22.19** | 21.66 | 20.97 | 20.65 | 20.57 | 20.50 | **18.19** |
| PEMS04 encoder, no episodes | 33.87 | 31.14 | 29.09 | 25.97 | 24.11 | 23.22 | 21.27 | 18.14 |

Normal setting (all sensors known): encoder 13.24 (PEMS08) / 18.34 (PEMS04) vs U-Trans-style learned identity 13.44 / 18.30.

Reading (1 seed):
- Cold-start episodes are what make it work: the same encoder trained only on full histories fails on new sensors
  (3 d: 18.76 / 25.97 vs 15.79 / 20.97).
- The trained encoder ties the fixed relative profile with 3+ days, is better with 0-1 day, degrades gracefully to about
  no-identity level with no history, and is the best on known sensors; no cost in the normal setting.

## 2026-11-18 · Does the network mean make a new sensor's identity reliable from a few days? (CPU check, no training)

Identity of each unseen sensor (20 %, rng 123) from k random training days vs its full-span identity. raw = log hourly
flow profile (24 values); relative = log(sensor / network mean of the known sensors, same days and hours). 30 draws.
"id" = share of unseen sensors whose k-day identity is closest to their own full-span identity among all sensors.

| data | days drawn from | k | raw R² / id | relative R² / id |
|---|---|---|---|---|
| PEMS04 | any | 1 | 0.769 / 39 % | 0.857 / 66 % |
| PEMS04 | any | 3 | 0.919 / 61 % | 0.949 / 81 % |
| PEMS04 | any | 7 | 0.974 / 84 % | 0.984 / 91 % |
| PEMS04 | weekdays | 1 / 3 / 7 | 77 / 83 / 91 % | 80 / 88 / 92 % |
| PEMS08 | any | 1 | 0.900 / 34 % | 0.956 / 54 % |
| PEMS08 | any | 3 | 0.964 / 55 % | 0.985 / 71 % |
| PEMS08 | any | 7 | 0.988 / 74 % | 0.995 / 83 % |
| PEMS08 | weekdays | 1 / 3 / 7 | 65 / 76 / 84 % | 70 / 80 / 86 % |

Reading: dividing by the network mean makes short-history identity clearly more reliable, most with 1 to 3 days of
mixed day types (identification 39 -> 66 % and 34 -> 54 % at 1 day). When the days are all weekdays the gain is small,
so most of the benefit is cancelling day-type effects (weekend / holiday / unusual days). The PEMS08 sensor 2 example
(3 days incl. a Saturday, identity far from its full-span value) was an unlucky case, not the rule. This measures
identity stability only, not forecast accuracy; that needs training runs.

## 2026-11-18 · Does a network-relative identity give better forecasts on unseen sensors? (slim MAGE)

`MAGE_MS/run_relative.py` (achilleas GPUs 6, 7; 2 batch processes per GPU via `batch_run.py`). Seed 0, 20 % unseen.
Identity = 24 hourly values, raw (log flow) or relative (log sensor / network mean of known sensors), `--identity hraw|hrel`
in `train.py`. Known sensors: whole training span; unseen sensors: only their last k days (`--unseen_days`).
Last days are weekend-heavy (PEMS04 k=1 Sunday, k=3 Fri-Sun; PEMS08 k=1 Saturday, k=3 Thu-Sat).

Unseen-sensor test MAE (known-sensor MAE in brackets):

| data | k days | raw | relative | relative - raw |
|---|---|---|---|---|
| PEMS04 (no identity 21.79 / 19.71) | 1 | 31.03 (19.25) | **26.47** (18.30) | -4.56 |
| | 3 | 23.76 (18.60) | **22.19** (18.22) | -1.57 |
| | 7 | 21.48 (18.25) | **21.03** (18.27) | -0.45 |
| | full | 20.27 (18.27) | 20.18 (18.30) | -0.09 |
| PEMS08 (no identity 16.95 / 14.16) | 1 | 20.85 (13.74) | **18.02** (13.62) | -2.83 |
| | 3 | 16.49 (13.65) | **16.19** (13.70) | -0.30 |
| | 7 | 16.19 (13.67) | 16.08 (13.69) | -0.11 |
| | full | 16.13 (13.69) | 16.21 (13.68) | +0.08 |

Reading (1 seed): the network-relative identity gives better unseen-sensor forecasts than the raw one whenever history is
short, by a lot at 1 day (-4.6 / -2.8) and still at 3 days; equal with the full span. So the network mean improves
new-sensor identity AND forecasts. But with 1 day (both data) and 3 days (PEMS04) even the relative identity is worse
than no identity: very short histories still need a fallback (shrink toward the average sensor / no identity).
Caveat: known-sensor MAE should be the same for a given identity kind across k (unseen rows never enter the loss) but
varies 18.2-19.3 on PEMS04 (raw k=1 stopped early at 137 epochs); either run noise (threads share the RNG) or the unseen
sensors' identities reach known sensors through a cross-sensor path. To check before relying on small differences.

## 2026-11-18 · Cold start: train with identities from random short windows

Check first (CPU): from 1 day, the network-relative identity of an unseen sensor already matches its whole-span identity
with R² 0.79 (PEMS04) / 0.96 (PEMS08); occupancy + speed raise PEMS04 to 0.89; road neighbours' identities predict
nothing (R² 0.04 / -0.24). So the 1-day failure is the model's sensitivity to identity error, not missing information.
Fix: `--id_window 1` resamples every sensor's identity from a random 1-7 day window each batch (whole span with prob 0.3);
`--id_os 1` adds occupancy and speed (`id_features.WindowIdentity`, launcher `MAGE_MS/run_coldstart.py`, GPUs 6/7).
Slim MAGE, seed 0, 20 % unseen sensors; unseen identity from the last k days (weekend-heavy for k <= 3).

Unseen-sensor MAE (known-sensor MAE in brackets). "fixed" = same identity, trained with whole-span identities only.

| data | k | no identity | fixed | window training | + occupancy, speed |
|---|---|---|---|---|---|
| PEMS04 | 1 | **21.79** (19.71) | 26.47 (18.30) | 22.53 (18.39) | 22.34 (18.28) |
| PEMS04 | 3 | 21.79 | 22.19 (18.22) | 20.86 (18.51) | **20.80** (18.32) |
| PEMS04 | 7 | 21.79 | 21.03 (18.27) | **20.25** (18.44) | 20.28 (18.34) |
| PEMS08 | 1 | 16.95 (14.16) | 18.02 (13.62) | 16.70 (13.77) | **16.62** (13.57) |
| PEMS08 | 3 | 16.95 | 16.19 (13.70) | **15.96** (13.69) | 16.02 (13.63) |
| PEMS08 | 7 | 16.95 | 16.08 (13.69) | 16.05 (13.68) | 16.05 (13.57) |

Reference, whole-span identity for unseen sensors (fixed training): 20.18 / 16.21.

Reading (1 seed): window training is the big lever. 1 day: -3.9 (PEMS04) and -1.3 (PEMS08) vs fixed training; 3 days:
-1.3 / -0.2. It now beats no identity in 5 of 6 settings, the exception being PEMS04 with a single Sunday (22.34 vs
21.79). With 3 to 7 days it matches or beats the whole-span identity of the earlier runs (PEMS08 3 days 15.96 vs 16.21).
Occupancy and speed add little (<= 0.2) but remove the small known-sensor cost of window training (PEMS04 18.3 vs 18.4-18.5).

## 2026-11-18 — Cold start: PCA comparison (run_pca_coldstart.py)
Slim MAGE, seed 0, 20 % unseen sensors, unseen identity from their last k days. Unseen MAE (known-sensor MAE ~18.2-18.8 / 13.6-14.0 for all).

| PEMS04 | k=1 | k=3 | k=7 |
|---|---|---|---|
| no identity | 21.79 | 21.79 | 21.79 |
| PCA, fixed training (as published) | 33.71 | 23.03 | 21.04 |
| relative, fixed training | 26.47 | 22.19 | 21.03 |
| PCA + window training | 22.49 | 20.85 | 20.27 |
| PCA-of-relative + window training | 22.57 | 20.97 | 20.58 |
| relative + window training (win) | 22.53 | 20.86 | 20.25 |
| relative + occ/speed + window (win_os) | 22.34 | 20.80 | 20.28 |

| PEMS08 | k=1 | k=3 | k=7 |
|---|---|---|---|
| no identity | 16.95 | 16.95 | 16.95 |
| PCA, fixed training (as published) | 21.67 | 16.67 | 16.57 |
| relative, fixed training | 18.02 | 16.19 | 16.08 |
| PCA + window training | 16.81 | 16.16 | 16.11 |
| PCA-of-relative + window training | 16.76 | 16.08 | 15.97 |
| relative + window training (win) | 16.70 | 15.96 | 16.05 |
| relative + occ/speed + window (win_os) | 16.62 | 16.02 | 16.05 |

Findings: (1) published PCA collapses with short histories (k=1: +11.9 / +4.7 MAE worse than no identity).
(2) Window training is what fixes it, for any identity: PCA + window ≈ our relative + window (differences ≤0.2, inside seed noise).
(3) Network-mean normalisation helps under fixed training (26.47 vs 33.71, 18.02 vs 21.67) but adds little once window training is used.
(4) PEMS04 k=1 (one Sunday) is still worse than no identity for every variant. Single seed.

## 2026-11-18 — Fair unseen-sensor comparison by history length (eval_history.py)
Each saved checkpoint re-tested on the same unseen sensors with 0 days (identity = mean of known sensors), last 1/3/7 days, or full span (train.py --eval_ckpt). Mean over identically trained checkpoints (n = 3-4; spread <= 0.15 for window models). Learned / none / SCPT: stored results (no checkpoints; learned and none ignore the history).

| PEMS04 | 0 | 1 | 3 | 7 | full |
|---|---|---|---|---|---|
| learned embedding | 32.40 | 32.40 | 32.40 | 32.40 | 32.40 |
| no identity | **21.81** | **21.81** | 21.81 | 21.81 | 21.81 |
| SCPT | - | - | - | - | 21.21 |
| PCA (published) | 29.72 | 31.25 | 23.20 | 20.81 | 20.09 |
| raw profile | 28.89 | 30.13 | 23.53 | 21.39 | 20.28 |
| relative profile | 28.86 | 26.65 | 22.29 | 20.93 | 20.21 |
| PCA + window | 25.24 | 22.53 | 20.91 | **20.22** | 20.12 |
| PCA of relative + window | 24.64 | 22.52 | 21.02 | 20.60 | 20.51 |
| relative + window (ours) | 24.61 | 22.52 | **20.82** | 20.26 | **20.05** |
| relative + occ/speed + window (ours) | 25.76 | 22.40 | **20.82** | 20.30 | 20.12 |

| PEMS08 | 0 | 1 | 3 | 7 | full |
|---|---|---|---|---|---|
| learned embedding | 22.73 | 22.73 | 22.73 | 22.73 | 22.73 |
| no identity | **16.95** | 16.95 | 16.95 | 16.95 | 16.95 |
| SCPT | - | - | - | - | 17.10 |
| PCA (published) | 21.20 | 21.46 | 16.90 | 16.58 | 16.55 |
| raw profile | 20.62 | 21.29 | 16.48 | 16.15 | 16.10 |
| relative profile | 20.51 | 18.12 | 16.19 | 16.18 | 16.11 |
| PCA + window | 18.69 | 16.87 | 16.15 | 16.07 | 16.05 |
| PCA of relative + window | 18.56 | 16.72 | 16.06 | 16.01 | 16.03 |
| relative + window (ours) | 18.59 | 16.70 | **16.03** | **15.98** | 15.99 |
| relative + occ/speed + window (ours) | 19.48 | **16.62** | 16.07 | 16.01 | **15.97** |

0 days: every identity model is worse than no identity (the mean identity is not a neutral input). 1 day: no identity still wins on PEMS04. 3+ days: window-trained models win; ours ~ PCA + window (within ~0.1).

## 2026-11-18 · Archived README (replaced by the unseen-sensor README; kept verbatim)

The text below was README.md until 2026-11-18. The graph-collapse and identity findings are summarised in the new README; the multi-scale proposal, other directions and per-model tables live here and in the entries above.

### Adaptive graphs learn the mean: sensor identity, not graph structure, drives spatial gains in traffic forecasting

Most recent traffic forecasters learn an adaptive graph between sensors. We ask **when such a learned graph actually beats a simple network-wide average of all sensors**. Across six published models on PEMS04 and PEMS08, the answer is: only when the model has no other way to tell sensors apart. With a sensor-identity representation, the plain average matches the learned graph in five of six models, and in MAGE (NeurIPS 2025) the learned graph collapses to that average on its own. We show that sensor identity is a compact profile of each sensor's typical behaviour, which can be computed directly from its data. This leads to a **learning-free spatial module**: a fixed network-wide average plus a computed sensor profile, with no learned graph and no per-sensor parameters. It matches MAGE with about 70 % fewer parameters, matches or beats learned sensor embeddings across four datasets and two input lengths, and forecasts best for sensors never seen in training once they have about a week of history.

Setting: PEMS04 (307 sensors, 59 days) and PEMS08 (170 sensors, 62 days), traffic flow every 5 minutes, chronological 6:2:2 split, test MAE (lower is better), one seed per configuration (differences under about 0.05–0.08 MAE are noise). Per-run tables: [`experiment_log.md`](experiment_log.md); full history: [`research_design.md`](research_design.md).

---

#### 1. Sensor identity: what it is and why a model needs it

**The data.** Each dataset is one array, *time steps × sensors × measurements* (flow, occupancy, speed), plus a list of road connections. A sensor is only a column index: the data carry no location, road type or direction.

**The problem: indistinguishable sensors.** Two sensors can show almost the same recent history and still have different futures. PEMS08, Tuesday 12 July 2016, 07:00 (flow every 15 minutes):

| | last hour | next hour (actual) |
|---|---|---|
| sensor 31 | 206, 235, 260, 285 | 263, 286, 289, **337** (keeps rising) |
| sensor 148 | 216, 235, 260, 276 | 270, 306, 291, **251** (peaks, then falls) |

A model that sees only recent values and the time of day receives the same input for both and must give both the same forecast.

**Sensor identity** is any information that tells the model which sensor it is looking at and how that sensor usually behaves. Models obtain it in three ways:

| Source of identity | Example models | For a new sensor? |
|---|---|---|
| a learned embedding per sensor | MAGE, STAEformer, STID | no (untrained) |
| other per-sensor parameters, or a learned graph that mixes different sensors for each sensor | MTGNN (normalisation layers), Graph WaveNet (its graph) | no |
| **computed from the sensor's own data (ours)** | – | **yes** |

**What identity encodes.** Predicting sensor properties from the learned per-sensor parameters (cross-validated R², PEMS08 / PEMS04):

| learned per-sensor parameters | traffic volume | daily shape | commute direction (AM vs PM peak) | follows the network signal |
|---|---|---|---|---|
| MAGE sensor embedding | 0.77 / 0.93 | 0.78 / 0.87 | 0.15 / 0.69 | 0.51 / 0.56 |
| STAEformer adaptive embedding | 0.78 / 0.92 | 0.53 / 0.88 | 0.53 / 0.71 | 0.64 / 0.66 |
| MTGNN normalisation parameters | 0.89 / 0.93 | 0.65 / 0.78 | 0.20 / 0.57 | 0.66 / 0.61 |
| Graph WaveNet graph vectors | 0.21 / 0.03 | 0.17 / 0.09 | ≈ 0 / 0.19 | ≈ 0 |

Identity is a **profile of the sensor's typical behaviour**: how busy it is, its daily shape, whether it is a morning or evening commuter road, and how strongly it moves with the whole network. In the example above, the profile separates the two sensors: sensor 148 normally peaks early in the morning (its peak is ending at 07:00), while sensor 31 is an evening-heavy road still ramping up.

**What identity contributes.** Removing it from a model with a fixed spatial average costs +0.82 / +1.68 MAE (PEMS08 / PEMS04), more than any spatial design choice we tested.

#### 2. Finding: a learned graph is needed only when identity is missing

Each model was trained from scratch with spatial mixing **off**, fixed to the **network-wide average** (every sensor receives the mean of all sensors), or **learned** (12 → 12 steps; MAGE via a slim variant, 96 → 12):

| Model | identity? | no mixing | average | learned | learned beats average? |
|---|---|---|---|---|---|
| MAGE | yes | – | 13.79 / 18.19 | 13.79 / 18.29 | **no** (its graph collapses to the average) |
| AGCRN | yes | 18.96 / 23.47 | **15.64 / 19.43** | 15.79 / 19.41 | **no** |
| MTGNN | yes | 15.44 / 19.14 | **15.11 / 19.00** | 15.24 / 19.18 | **no** |
| STAEformer | yes | 13.38 / 18.57 | 13.41 / 18.35 | 13.47 / 18.26 | **no** / slightly |
| Graph WaveNet | **no** | 17.04 / 22.77 | 15.98 / 21.83 | 14.78 / 19.14 | **yes** |
| ASTGCN | **no** | 20.97 / 26.33 | 20.50 / 25.70 | 19.05 / 22.04 | **yes** |

**Intervention: give the two exceptions identity, keep the plain average.**

| Test MAE (PEMS08 / PEMS04) | Graph WaveNet | ASTGCN |
|---|---|---|
| learned graph | 14.78 / 19.14 | 19.05 / 22.04 |
| average | 15.98 / 21.83 | 20.50 / 25.70 |
| **average + identity** | **14.67 / 18.80** (beats learned) | **19.04** / 24.00 (equal; half the gap) |
| fixed random graph, never trained | 15.31 / 20.11 | 21.22 / 26.51 |

In Graph WaveNet the learned graph was compensating for missing identity; even a random graph, which only makes sensors look different from each other, recovers 55–64 % of its gain. ASTGCN keeps part of its gain on PEMS04 because its attention operates on the real road graph.

**Supporting evidence.**
- **MAGE's graph collapses to the average.** In every trained MAGE, including the unmodified official code, its 16 graph experts become the plain sensor average within the first epoch; replacing them by the exact average changes MAE by ≤ 0.001, so 56–69 % of its parameters do no work. Preventing the collapse makes the graph random and the model overfits (+1.3 to +2.2 MAE).
- **Learned graphs do not learn roads.** Road neighbours get 0.9–1.2× the weight of a random pair in most models; a few hub sensors dominate the links.
- **Local structure does not explain the gain.** Dependence left after removing the network average does not predict where a learned graph helps (|ρ| ≤ 0.14 per sensor).

**Conclusion.** In these benchmarks, spatial modelling reduces to two parts: a **network-wide common signal** and **each sensor's identity**. Given identity, a learned graph is unnecessary, with one exception: attention built on the real road graph (ASTGCN on PEMS04).

#### 3. Proposed: a learning-free spatial module

Both parts can be provided without learning a graph or any per-sensor parameter:

1. **Fixed aggregation:** the network-wide average of all sensors, computed once per time step and shared by every sensor (one learned strength per layer).
2. **Computed identity:** a 52-number profile per sensor from its training data (hourly weekday and weekend shape, volume, morning/evening peak ratio, loading on the network average), passed through one shared linear layer.

Each sensor is then forecast from its own history, the network average and its profile. The forecasting backbone is still trained; the spatial module and the identity involve no learned graph and no per-sensor parameters. Implemented on slim MAGE (`MAGE_MS/streams.py`, arm `lite`, `--identity profile`), 96 → 12 steps:

| PEMS08 / PEMS04 | MAGE | average + learned identity | average, no identity | **average + computed identity** |
|---|---|---|---|---|
| test MAE | 13.79 / 18.29 | 13.79 / 18.19 | 14.61 / 19.87 | **13.85 / 18.25** |
| parameters | 1.87 M / 2.73 M | 0.72 M / 0.74 M | 0.70 M | **0.70 M** + 7 k shared |
| learned per-sensor parameters | ~6,300 per sensor | 128 per sensor | 0 | **0** |

The computed profile recovers 92–96 % of the value of a learned embedding and is within noise of MAGE.

**Across datasets, seeds and window lengths** (test MAE; "unseen" = 20 % of sensors excluded from training, profile computed from their history in the training period; PEMS04 / PEMS08 at 96 steps are 3-seed means, other rows 1 seed):

| setting | learned identity | **computed identity** | no identity | unseen: learned | **unseen: computed** | unseen: none |
|---|---|---|---|---|---|---|
| PEMS04, 96 → 12 | **18.21** | 18.25 | 19.93 | 32.40 | **20.36** | 21.81 |
| PEMS08, 96 → 12 | **13.78** | 13.84 | 14.55 | 22.73 | **16.66** | 16.95 |
| PEMS03, 96 → 12 | 14.66 | **14.38** | 15.12 | 28.05 | **13.95** | 14.81 |
| PEMS07, 96 → 12 | 19.45 | **19.37** | 21.49 | 27.40 | **19.09** | 20.80 |
| PEMS04, 12 → 12 | 18.35 | **18.20** | 20.55 | 42.21 | **21.08** | 22.25 |
| PEMS08, 12 → 12 | 13.84 | **13.64** | 14.57 | 28.22 | 17.34 | **17.32** |

- **Known sensors:** the computed profile matches the learned embedding (slightly behind on PEMS04 / PEMS08 at 96 steps, +0.04 / +0.05; ahead on PEMS03, PEMS07 and with 12-step inputs, −0.08 to −0.27) and is always far better than no identity.
- **New sensors:** a learned embedding fails (27–42 MAE, worse than no identity). The computed profile is the best option in 5 of 6 settings (−0.3 to −1.7 vs no identity) and ties in one.
- **How much history a new sensor needs:** a profile from the last 7 days is as good as the full training period; 3 days is not enough (about the no-identity level).

#### 4. Related work and novelty

| Work | Claim | Relation to ours |
|---|---|---|
| STID (CIKM 2022), STAEformer (CIKM 2023) | identity embeddings make simple models strong; sensors must be distinguishable | we show *why learned graphs help* in existing models (missing identity), confirm it by intervention, and measure what identity encodes |
| U-Trans (arXiv 2026) | a uniform average matches spatial attention | same aggregation; we explain when it suffices and the exceptions |
| FAF (ICML 2026), SGP (AAAI 2023), SimST | fixed neighbourhood aggregation replaces message passing | in traffic a **global** average suffices; road neighbourhoods added nothing on top |
| DeepGLO (NeurIPS 2019) | global factors + per-series models | same family; reached from what learned graphs actually do |
| SCPT, MoGERNN (2025) | forecasting for unseen roads / locations | SCPT learns a history encoder with pre-training; we use a few computed statistics, no pre-training |
| Common Correlated Effects (Pesaran 2006) | cross-sectional averages proxy common factors | the collapsed adaptive graph computes exactly this average |

The ingredients exist separately. The contribution is the **diagnosis**: in published adaptive-graph forecasters, the learned graph substitutes for sensor identity on top of a network-wide average, shown by intervention across six models, which yields a simple learning-free alternative.

#### 5. Limitations

Three seeds for the main comparison on PEMS04 / PEMS08, one seed elsewhere; four datasets (traffic flow only); the five comparison models are reimplemented from the official code; the proposed module is tested inside slim MAGE at 96 → 12 steps; the unseen-sensor profile requires some history for the new sensor. PEMS at a 1-hour horizon is close to its noise floor. Next: 3–5 seeds, PEMS03/07 and LargeST, MAGE's 12 → 12 setting, comparison with SCPT.

---

#### Appendix

**How we got here.** The original proposal (summarised in [`experiment_log.md`](experiment_log.md), "Archived documents") gave MAGE several time scales and let coarser scales mix further across the sensor graph. It failed (PEMS08, 5 seeds: ties with uniform and reversed mixing), because coarser scales are more correlated across sensors but not over longer road distances: most shared signal is one network-wide shock. Investigating that shock led to the collapse finding and then to the identity explanation.

**Other directions tried** (negative or inconclusive): multi-scale encoders and longer windows; separate scale streams; bounded λ (−0.04 MAE); co-fluctuation, road or learned low-rank graphs on top of the average; a general / time / sensor rank hierarchy; preventing MAGE's collapse. Growing a learned graph from the average only while validation improves selected the average alone.

**Code.**

| What | Where |
|---|---|
| Slim MAGE, computed identity, unseen-sensor test | `MAGE_MS/streams.py`, `MAGE_MS/train.py`, `run_identity_mage.sh` |
| Other models (reimplemented) and mixing / identity variants | `MAGE_MS/others/reimpl*.py`, `train_other*.py`, `run_identity.sh` |
| What identity encodes; graph analyses; per-sensor test | `MAGE_MS/others/identity_check.py`, `what_graphs_learn.py`, `rq_test.py` |
| Which MAGE parameters do work | `MAGE_MS/diag_useful.py` |

## 2026-11-20 · Repository cleanup
The first-phase code and results (multi-scale proposal, routing, cost, diagnostics, step 8, rank hierarchy, methods A / B,
official reproduction) moved to `MAGE_MS/archive/` (see `MAGE_MS/archive/README.md`); log entries above keep their
original paths. `others/results/staeformer_<ds>_s0_mean.json` had been overwritten by the 2026-11-09 U-Trans-style rerun
(learned identity, 18.299 / 13.437); the cross-model-check values (18.350 / 13.407) are restored there and the rerun is
saved as `staeformer_<ds>_s0_mean_idlearned.json`.
