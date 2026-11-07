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
| Reversed (8, 4, 2, 1) | – | not run | |
| Independent | – | not run | |

**Verdict:**
- **Proposed vs Shared:** −0.017, 2/5. The mixing schedule does not matter.
- **Neither beats MAGE.**
- **Separate streams vs early merge (Shared vs m4): −0.067, 4/5.** This is the only positive.
- **Uniform hurts,** so learned structure is needed.
- **The graph as a whole is worth only about 0.06 MAE** on PEMS08.
- **Seed variance doubles** (sd 0.09 vs 0.044).

## Not done yet (proposal.md and plan)
- **proposal.md variants:** Reversed (the key comparison) and Independent.
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

---

## Jiwon-work — Proposal Implementation and Matched Evaluation

**Date and source notes:** The November 3 and 5 changes below are verified by GitHub commits on the `Jiwon-work` branch. The November 6 entry is based on the server commands, logs, and partial report shared during the evaluation. The additional matched-evaluation scripts and results have not been verified as committed to GitHub. The original `main` experiment log above is retained unchanged; its "Not done yet" section describes the earlier state of `main`.

### 2026-11-03 · V1: ProposalMAGE and controlled preprocessing

**Starting point:** The team's existing MAGE implementation and earlier multi-scale encoder/stream experiments. The proposal implementation was added separately rather than replacing the original MAGE baseline.

**Added / modified — preprocessing**
- Fixed the forecasting task to **96 historical steps -> 12 future steps** at five-minute sampling intervals.
- Used **traffic flow + time-of-day + day-of-week** as inputs; removed occupancy and speed from the proposal's inputs. Traffic flow remains the prediction target.
- Changed the split procedure from a chronological split of sample/end-of-input indices to fixed **60/20/20 boundaries on the original time axis**. All 12 target steps for each sample must fall inside the designated train, validation, or test interval. Historical input windows can extend into a preceding interval.
- Computed flow mean and standard deviation from the **training time interval only**, applying the same statistics to validation and test.
- Added `MAGE_MS/prep_proposal.py`, dedicated `data_proposal96` preprocessing, metadata, and tests (`test_proposal_prep.py`).
- Updated the older `prep.py` to avoid target windows crossing split boundaries and updated `train.py` to compute masked metrics over **finite, strictly positive targets**.
- Changed the older `streams.py` Uniform variant to use a **pure global mean** instead of retaining a signed-graph contribution; added relevant tests.

**Added / modified — temporal model**
- Added `MAGE_MS/proposal_model.py` with a separate `ProposalMAGE` implementation.
- Built four streams by average pooling: **96 / 48 / 24 / 12 steps** (5 / 10 / 20 / 40 minutes).
- Added **TimeMixer-style Past-Decomposable-Mixing (PDM)**: moving-average trend/seasonal decomposition, seasonal mixing **fine -> coarse**, and trend mixing **coarse -> fine**. This is a TimeMixer-inspired adaptation, **not a reproduction of the complete official TimeMixer architecture**.
- Initial PDM settings: **2 PDM blocks**, width **16**, and moving-average kernel **5**.

**Added / modified — spatial model**
- Used **three spatial blocks**, low-rank learned graph expert pools (**16 experts**), **Top-4 routing**, and a shared fine-scale routing decision across all resolutions within each block.
- Shared the positive graph operator across temporal resolutions **within each spatial block**, but varied its repeated application count. Each block has its own operator parameters.
- Implemented five variants:

  | Variant | Positive spatial propagation per scale (fine -> coarse) |
  |---|---|
  | Shared | (1, 1, 1, 1) |
  | Proposed | (1, 2, 4, 8) |
  | Reversed | (8, 4, 2, 1) |
  | Independent | Separate expert pool per scale |
  | Uniform | Global mean mixing |

- Applied **direct positive graph propagation** `S_+^k` with a separate negative graph contribution. This differs from the earlier `main` Step-8 **lazy operator** `P^k`, where `P = (I + S_+)/2`.
- Added a prediction head for each scale and summed the per-scale predictions.
- Added training/evaluation scripts (`train_proposal.py`, `run_proposal.py`) and model tests (`test_proposal.py`). Kept a **linear-encoder MAGE baseline** for comparisons under the same 96 -> 12 preprocessing protocol; also added an optional MAGE + engineered-temporal-features baseline.

**Commits:** [Update MAGE_MS implementation and tests](https://github.com/hoangg-ng/graph/commit/a235447f1b53a6fab3142ed0ad2a291366671fa8); [Add proposal-only MAGE implementation](https://github.com/hoangg-ng/graph/commit/f9bf4ab593bf3186be8190730bd853fada2815b0).

### 2026-11-05 · V2: Separate positive and negative graph expert parameters

**Before:** The positive and negative graph operators reused the same factor parameters `E1` and `E2`.

**Modified**
- `proposal_model.py`: Introduced **separate `E1` and `E2` matrices** for positive and negative graph operators, allowing them to learn distinct spatial relationships.
- Continued to apply the **scale-dependent schedule to positive propagation**, while applying negative propagation separately with learned expert-specific coefficients.
- `test_proposal.py`: Updated operator tests for the separated positive/negative graph parameters.

**Added**
- Preprocessing metadata files for PEMS04 and PEMS08.

**Motivation / limitation:** This corrects the unintended sharing of graph parameters between the two signs. **No isolated before/after MAE experiment was recorded**, so there is no evidence yet that this change alone improves prediction accuracy.

**Commit:** [Separate signed graph expert parameters](https://github.com/hoangg-ng/graph/commit/4c6854bfa98c2e3830fe82e24c592399e422cf67).

### 2026-11-06 · V3: Hyperparameter selection and matched multi-seed evaluation (server-side)

**Added / modified — hyperparameter selection**
- Evaluated **10** candidate hyperparameter settings for the **Shared** variant, using up to **60 epochs** and early-stopping patience **15**.
- Selected a common recipe: **learning rate 0.005**, **weight decay 0.0001**, and **dropout 0.3**.
- Selected trial validation MAE: **18.1777**; this is **not a test MAE**.
- Saved the selected recipe on the server as `results/tune_shared/selected_recipe.json`.

**Added / modified — matched evaluation**
- Started **60 independent training jobs**: **2 datasets (PEMS04, PEMS08) x 6 variants (MAGE, Shared, Proposed, Reversed, Independent, Uniform) x 5 seeds (0–4)**.
- Used a common **96 -> 12 task**, **flow/calendar inputs**, **target-disjoint chronological split**, normalization protocol, and model-independent training settings across variants.
- Main settings: maximum **300 epochs**, early-stopping patience **30**, up to **4 concurrent GPU processes** (one per GPU, three CPU threads per process).
- Used server-side `run_matched_eval.py` and `report_matched_eval.py` to schedule runs, record results, and generate paired-seed comparisons and Markdown/CSV summaries. The helper scripts and result files have **not been confirmed as pushed to GitHub**.
- Result location: `results/eval_proposal96` on the server.

**Preliminary PEMS04 results — snapshot when 6/60 jobs had finished**

| Variant | Completed seeds | Masked test MAE |
|---|---:|---:|
| MAGE | 2/5 | 18.093 +/- 0.016 |
| Shared (1,1,1,1) | 1/5 | 18.118 |
| Proposed (1,2,4,8) | 1/5 | 18.171 |
| Reversed (8,4,2,1) | 1/5 | 18.132 |
| Independent | 1/5 | 18.095 |
| Uniform | 0/5 | Pending |

- MAGE seed 0: **test MAE 18.1038**, best validation MAE **18.125**, early stopped at epoch **132**.
- In the only complete paired seed, **Proposed did not beat MAGE, Shared, or Reversed**. Independent was slightly better than MAGE in that seed.
- At this snapshot, **PEMS08 had 0/30 completed** and **54/60 jobs were still outstanding**. This is a historical snapshot, **not a live progress update**.
- **Do not interpret partial results as five-seed averages or conclude that a variant wins/loses overall yet.**

**Comparability note:** The original MAGE paper's **12 -> 12** results, the earlier team's experiments using **different features/splits/operators**, and the current **flow-only matched 96 -> 12** results are distinct evaluation protocols; their absolute MAE values must not be treated as matched comparisons.


### 2026-11-07 · V4: Completed matched evaluation (60 runs)

**Status:** The server-side matched evaluation is reported as completed. The results below are based on the five-seed evaluation summary. Individual training result JSONs and checkpoints remain on the server and are not committed to GitHub.

**Experimental protocol**
- Datasets: PEMS04 and PEMS08.
- Input/output: **96 historical steps → 12 future steps** (5-minute sampling).
- Variants: MAGE, Shared, Proposed, Reversed, Independent, and Uniform.
- Seeds: **0–4** for each variant and dataset.
- Total: **2 datasets × 6 variants × 5 seeds = 60 runs**.
- Shared configuration: learning rate **0.005**, weight decay **0.0001**, dropout **0.3**, batch size **64**, maximum **300 epochs**, and early-stopping patience **30**.
- Primary metric: masked test MAE; lower is better.
- Result directory: `MAGE_MS/results/eval_proposal96/`.

**Matched test MAE (mean ± SD, 5 seeds)**

| Variant | PEMS04 | PEMS08 |
|---|---:|---:|
| MAGE | 18.092 ± 0.022 | **13.813 ± 0.020** |
| Shared (1,1,1,1) | 18.148 ± 0.056 | 13.930 ± 0.024 |
| Proposed (1,2,4,8) | 18.130 ± 0.070 | 13.911 ± 0.031 |
| Reversed (8,4,2,1) | 18.136 ± 0.050 | 13.929 ± 0.031 |
| Independent | 18.114 ± 0.029 | 13.913 ± 0.012 |
| Uniform | **17.998 ± 0.039** | 13.827 ± 0.068 |

**Findings**
- Proposed achieved slightly lower mean MAE than Shared and Reversed on both datasets.
- These reductions were below the prespecified **1% mean MAE improvement threshold**. The original support criterion was therefore not satisfied.
- Proposed performed worse than MAGE on both datasets.
- Uniform achieved the lowest mean MAE on PEMS04; MAGE achieved the lowest on PEMS08.
- The results do not establish a forecasting advantage from assigning more positive spatial propagation steps to coarser temporal resolutions.
- Results from the earlier `main` Step-8 experiments must not be pooled with these results because their model implementations and evaluation protocols differ.

**Related evaluation files**
- `MAGE_MS/run_matched_eval.py`
- `MAGE_MS/report_matched_eval.py`
- `MAGE_MS/results/eval_proposal96/matched_mae_table.md`
- `MAGE_MS/results/eval_proposal96/matched_mae_summary.csv`
- `MAGE_MS/results/eval_proposal96/matched_mae_per_seed.csv`



### 2026-11-07 · V7: Completed follow-up MAE evaluation and computational profiling

**Status:** The follow-up evaluation is complete, with 40 additional training runs. Together with the original 60 matched runs, the study now contains **100 training runs across 10 model variants**. Computational profiling was performed separately using saved seed-0 checkpoints.

#### Experimental setup

- **Datasets:** PEMS04 and PEMS08.
- **Forecasting task:** 96 historical steps → 12 future steps, with five-minute sampling.
- **Original evaluation:** 6 variants × 2 datasets × 5 seeds = 60 runs.
- **Follow-up evaluation:** 4 variants × 2 datasets × 5 seeds = 40 runs.
- **Seeds:** 0–4.
- **Hyperparameters:** Learning rate 0.005, weight decay 0.0001, dropout 0.3.
- **Training:** Maximum 300 epochs, early-stopping patience 30, batch size 64.
- **Primary metric:** Masked test MAE, reported as mean ± sample standard deviation.
- **Computational profiling:** Seed-0 checkpoint per model and dataset, with inference batch sizes 1 and 64.

#### Final matched MAE results

The following table combines the original evaluation with the completed follow-up evaluation.

| Model | PEMS04 MAE | PEMS08 MAE |
|---|---:|---:|
| MAGE | 18.092 ± 0.022 | 13.813 ± 0.020 |
| Shared (1,1,1,1) | 18.148 ± 0.056 | 13.930 ± 0.024 |
| Proposed (1,2,4,8) | 18.130 ± 0.070 | 13.911 ± 0.031 |
| Reversed (8,4,2,1) | 18.136 ± 0.050 | 13.929 ± 0.031 |
| Independent | 18.114 ± 0.029 | 13.913 ± 0.012 |
| Uniform | 17.998 ± 0.039 | 13.827 ± 0.068 |
| `mage_features` | 18.0183 ± 0.0217 | 13.7602 ± 0.0146 |
| `no_spatial` | 18.2481 ± 0.0354 | 13.8168 ± 0.0313 |
| **`grouped`** | **17.8589 ± 0.0211** | **13.6781 ± 0.0369** |
| `proposed_no_pdm` | 18.0497 ± 0.0300 | 13.8876 ± 0.0320 |

**Accuracy findings**

- **Grouped achieved the lowest mean test MAE on both datasets** among all ten evaluated variants.
- Relative to the original Proposed model, Grouped reduced mean MAE by approximately **1.50% on PEMS04** and **1.67% on PEMS08**.
- Relative to the strongest original model on each dataset, Grouped reduced mean MAE by approximately **0.77% on PEMS04** (versus Uniform) and **0.98% on PEMS08** (versus MAGE).
- The engineered-feature MAGE baseline also improved over the original MAGE baseline on both datasets.
- No-Spatial performed worse than Grouped on both datasets, suggesting that learned spatial propagation can remain useful after multiscale temporal representations are combined.
- Proposed-No-PDM outperformed Proposed on PEMS04 but underperformed it on PEMS08. The available results do not establish a consistent benefit or disadvantage from removing PDM.
- The original hypothesis that increasing spatial propagation strength at coarser temporal resolutions improves forecasting was **not supported by the prespecified decision criterion**.

**Interpretation**

Grouped combines four PDM-derived temporal representations into a single representation **before spatial processing**. In contrast, Proposed applies scale-dependent spatial propagation to four separate streams and combines their predictions afterward.

The results suggest that early integration of multiscale temporal representations may be a more effective architectural choice than separate scale-dependent spatial propagation.

However, Grouped also changes the number of spatial streams, prediction heads, parameter count, and propagation operations. Therefore, the observed improvement cannot be attributed exclusively to the temporal fusion order.

#### Computational profiling: follow-up variants

The following measurements were obtained from saved seed-0 checkpoints. FLOPs are the operations counted by the PyTorch profiler, not necessarily every operation performed by the model.

| Dataset | Variant | Params (M) | Counted GFLOPs | Graph apps (+/−) | Latency B=1 (ms) | Throughput B=64 (samples/s) |
|---|---|---:|---:|---:|---:|---:|
| PEMS04 | `grouped` | 3.056 | 1.6528 | 3/3 | 3.815 | 1746.5 |
| PEMS04 | `mage_features` | 2.765 | 0.9412 | Not comparable | 2.270 | 7714.1 |
| PEMS04 | `no_spatial` | 1.103 | 2.2152 | 0/0 | 3.748 | 1462.5 |
| PEMS04 | `proposed_no_pdm` | 2.594 | 6.0576 | 45/12 | 8.180 | 1557.8 |
| PEMS08 | `grouped` | 2.197 | 0.9152 | 3/3 | 3.860 | 3034.1 |
| PEMS08 | `mage_features` | 1.906 | 0.5212 | Not comparable | 2.206 | 12367.7 |
| PEMS08 | `no_spatial` | 1.085 | 1.2267 | 0/0 | 3.717 | 2591.9 |
| PEMS08 | `proposed_no_pdm` | 1.735 | 3.3543 | 45/12 | 7.994 | 2557.5 |

**Computational findings**

- Grouped uses **6 total graph applications per forward pass**, compared with 57 for Proposed-No-PDM.
- Grouped has lower counted GFLOPs than No-Spatial and Proposed-No-PDM despite using learned spatial propagation.
- Grouped requires approximately **72.7% fewer counted FLOPs** than Proposed-No-PDM on both datasets.
- However, MAGE + Features is computationally more efficient than Grouped in the reported measurements.
- On PEMS04, Grouped has approximately **1.76× the counted GFLOPs** and **1.68× the batch-1 inference latency** of MAGE + Features.
- On PEMS08, the corresponding ratios are approximately **1.76×** and **1.75×**.
- Grouped therefore provides the best observed predictive accuracy, while MAGE + Features provides a favorable accuracy-efficiency alternative.

### 2026-11-08 · V8: Controlled Grouped Ablation Study

**Objective:** Identify the contribution of PDM and learned spatial propagation to the Grouped architecture, following the completed 100-run evaluation.

#### Experimental setup

The ablation study introduced three variants of the existing Grouped model:

| Variant | Modification | Purpose |
|---|---|---|
| `grouped_no_pdm` | Replace PDM with independent linear encoders for each temporal scale | Evaluate the contribution of PDM |
| `grouped_no_spatial` | Remove cross-node spatial propagation while retaining temporal processing, spatial prompts, and FFN layers | Evaluate the contribution of learned spatial propagation |
| `grouped_uniform` | Replace learned spatial propagation with global-mean mixing | Compare learned spatial relationships with simple global mixing |

All variants retain four temporal resolutions (5, 10, 20, and 40 minutes), early temporal representation fusion, and a single prediction head.

**Experimental protocol**
- Datasets: PEMS04 and PEMS08
- Input/output: 96 historical steps → 12 future steps
- Sampling interval: 5 minutes
- Seeds: 0–4 for Grouped and Grouped-No-PDM; 0–1 for the spatial ablations
- Learning rate: 0.005
- Weight decay: 0.0001
- Dropout: 0.3
- Batch size: 64
- Maximum epochs: 300
- Early-stopping patience: 30
- Selection metric: Best validation MAE
- Evaluation mode: Validation-only; no test-set evaluation

**Completed training runs**
- Initial ablations: 4 variants × 2 datasets × 2 seeds = 16 runs
- Additional PDM ablations: 2 variants × 2 datasets × 3 seeds = 12 runs
- Total: **28 new training runs**
- Cumulative experiments: **100 earlier test-evaluation runs + 28 validation-only runs = 128 training runs**

#### 1. PDM ablation: Five-seed validation results

| Dataset | Grouped (mean ± SD) | Grouped-No-PDM (mean ± SD) | Relative improvement | Paired wins |
|---|---:|---:|---:|---:|
| PEMS04 | **17.9618 ± 0.0181** | 18.0245 ± 0.0159 | −0.35% | 0/5 |
| PEMS08 | 13.8996 ± 0.0416 | **13.8717 ± 0.0547** | +0.20% | 3/5 |

**Findings**

- On PEMS04, removing PDM increased mean validation MAE by approximately 0.0627, corresponding to a 0.35% deterioration. The original Grouped model outperformed Grouped-No-PDM across all five paired seeds.
- On PEMS08, Grouped-No-PDM reduced mean validation MAE by approximately 0.0279 (0.20%), but achieved better results in only three of five seeds.
- The PEMS08 result does not satisfy the previously adopted four-of-five-seeds consistency criterion.
- The contribution of PDM therefore appears dependent on the dataset. The results do not support removing PDM universally.

#### 2. Spatial ablation: Two-seed validation results

| Variant | PEMS04 (mean ± SD) | PEMS08 (mean ± SD) |
|---|---:|---:|
| `grouped` | **17.9553 ± 0.0088** | **13.9073 ± 0.0257** |
| `grouped_no_spatial` | 18.1361 ± 0.0104 | 13.9705 ± 0.0669 |
| `grouped_uniform` | 18.0745 ± 0.0804 | 14.0398 ± 0.0166 |

**Findings**

- Removing spatial propagation increased mean validation MAE by approximately 1.01% on PEMS04 and 0.45% on PEMS08.
- Replacing the learned spatial operator with global-mean mixing increased mean validation MAE by approximately 0.66% on PEMS04 and 0.95% on PEMS08.
- Grouped outperformed both spatial ablations in all paired two-seed comparisons on both datasets.
- These results provide preliminary evidence that learned spatial propagation contributes to the predictive performance of Grouped.
- However, the ablations also alter the number of model parameters. The contribution of spatial propagation has not been isolated under matched model capacity.

#### 3. Interpretation

The original proposal hypothesized that coarser temporal resolutions would benefit from stronger spatial propagation. Earlier experiments did not demonstrate a substantial or consistent advantage from this design.

Subsequently, Grouped achieved the lowest test MAE among the ten previously evaluated variants by combining multiscale temporal representations before spatial processing.

The present ablation study provides additional information about that architecture:

1. PDM is beneficial for Grouped on PEMS04 under the validation protocol, but its contribution on PEMS08 is inconclusive.
2. Removing learned spatial propagation or replacing it with global-mean mixing degrades validation performance in the two-seed comparisons.
3. The advantage of Grouped cannot yet be attributed exclusively to early temporal fusion because Grouped also changes the number of spatial processing streams, propagation operations, residual pathways, and prediction heads.

#### 4. Limitations

- The new ablation results are based on validation MAE, whereas the previous main experiments reported test MAE. The two sets of results must not be directly compared as identical evaluation outcomes.
- Grouped and Grouped-No-PDM were evaluated with five seeds; Grouped-No-Spatial and Grouped-Uniform were evaluated with only two seeds.
- Differences in parameter count remain a possible confounding factor for the spatial ablations.
- Statistical significance and generalization to other datasets have not been established.
- The earlier 1.5% test-MAE improvement target relative to MAGE has not yet been achieved on both datasets.
- 
**Conclusion:** The results favor retaining learned spatial propagation in the Grouped architecture. PDM contributes differently across PEMS04 and PEMS08 and requires further analysis. Future work will prioritize isolating the effects of temporal fusion and prediction-head design rather than further increasing scale-dependent spatial propagation depth.

#### Result locations

- `MAGE_MS/results/ablation_val96/`
- `MAGE_MS/results/ablation_val96_s234/`
- `MAGE_MS/results/ablation_val96_5seed/`
- `MAGE_MS/proposal_model.py`
- `MAGE_MS/run_ablation_eval.py`
- `MAGE_MS/test_grouped_ablation.py`
- `MAGE_MS/report_grouped_ablation.py`

The experimental code and results are maintained separately from the original MAGE implementation.
