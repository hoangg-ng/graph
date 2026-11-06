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

### Follow-up experiments — planned, not yet implemented

- [ ] Complete all 60 matched runs, check run signatures, and analyze **paired-seed wins**, average MAE, standard deviations, parameter counts, and training costs.
- [ ] Compare **early fusion** of multi-scale temporal features against **separate temporal streams**.
- [ ] Run a **no-PDM ablation** to isolate the effect of temporal decomposition and cross-scale mixing.
- [ ] Diagnose possible **oversmoothing from repeated spatial propagation**; consider reduced schedules as separately labeled exploratory variants.
- [ ] Add external baselines such as **TimeMixer and STID**, if needed for the final project report.
- [ ] Update Overleaf to describe the **implemented** PDM, shared routing, signed graph parameters, preprocessing, and common evaluation recipe.

**Documentation rule:** Keep the original team's results unchanged. Distinguish committed code changes, server-side runs, preliminary test snapshots, and future plans. Preserve the current 60-run experiment as a fixed reference when testing later modifications.
