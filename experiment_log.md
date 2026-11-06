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

  Jiwon-work — proposal implementation and matched evaluation
Source of dates: November 3 and 5 are verified GitHub commit dates. November 6 is based on the server evaluation output shared by Jiwon; the additional evaluation scripts and results are not verified as committed. This section follows the original main experiment log and does not replace its historical entries.
2026-11-03 — V1: ProposalMAGE and controlled data processing
Starting point: Existing MAGE and Hoang's multiscale experiments.

Added / modified — preprocessing
- Fixed the task to 96 input steps -> 12 output steps (5-minute sampling).
- Used traffic flow + time-of-day + day-of-week, removing occupancy and speed as input features.
- Changed splitting from chronological sample-index splits to 60/20/20 boundaries on the original time axis. Every prediction target window belongs entirely to its designated split; historical input windows may reach into a preceding interval.
- Computed the flow mean and standard deviation only on the training time interval, and used the same statistics for validation and test.
- Added prep_proposal.py, data_proposal96 metadata, and preprocessing tests.
- Updated legacy preprocessing and masked metrics to handle split boundaries and finite positive targets; changed the legacy Uniform arm to use a pure global mean.
Added / modified — model
- Added proposal_model.py and a separate ProposalMAGE, without replacing the existing MAGE implementation.
- Built four temporal streams using average pooling: 96/48/24/12 steps (5/10/20/40 minutes).
- Added a TimeMixer-inspired PDM module: moving-average decomposition into trend and seasonal components, seasonal mixing from fine to coarse, trend mixing from coarse to fine. This is an adaptation, not the full official TimeMixer model.
- Added three spatial blocks with low-rank graph experts, 16 experts, Top-4 routing, and shared routing/expert pools across temporal resolutions within each block.
- Applied repeated positive graph propagation according to the scale-specific schedule; the new direct S_+^k differs from Hoang's earlier lazy P^k, where P=(I+S_+)/2.
- Implemented Shared (1,1,1,1), Proposed (1,2,4,8), Reversed (8,4,2,1), Independent (separate expert pools per scale), and Uniform (global mean).
- Combined per-scale prediction heads; added model tests and a common-protocol MAGE baseline.
Commits: November 3 code/tests, Proposal implementation.
2026-11-05 — V2: Separate positive and negative graph experts
Before: Positive and negative spatial operators reused the same graph factor matrices (E1 and E2).

After:
- Introduced separate E1 and E2 parameters for positive and negative operators.
- Repeated positive spatial propagation by schedule while applying negative propagation separately with learned expert-specific coefficients.
- Updated operator tests and committed PEMS04/PEMS08 preprocessing metadata.
Motivation: Allow positive and negative graph relations to be learned independently. The MAE effect of this isolated modification has not been measured, so no performance gain is claimed.

Commit: November 5 signed graph experts.
2026-11-06 — V3: Hyperparameter tuning and matched evaluation (server-side)
Added / modified:
- Evaluated ten hyperparameter settings for the Shared arm (up to 60 epochs, patience 15).
- Selected a common recipe: learning rate 0.005, weight decay 0.0001, dropout 0.3. Selected validation MAE: 18.1777 (not a test result).
- Launched 60 matched runs: two datasets (PEMS04, PEMS08) x six arms (MAGE, Shared, Proposed, Reversed, Independent, Uniform) x five seeds (0–4).
- Main settings: 300 maximum epochs, patience 30, four GPUs with one process per GPU and three CPU threads per process.
- Used matched flow/calendar inputs, chronological target-disjoint splits, common model-independent training settings, and paired seed reporting.
- Added server-side evaluation and reporting helpers (run_matched_eval.py, report_matched_eval.py); outputs are in results/eval_proposal96. These helpers have not been verified as pushed to GitHub.
Preliminary PEMS04 snapshot — 6 of 60 jobs completed:
Model	Runs completed	Masked test MAE
MAGE	2/5	18.093 +/- 0.016
Shared	1/5	18.118
Proposed	1/5	18.171
Reversed	1/5	18.132
Independent	1/5	18.095
Uniform	0/5	pending


- MAGE seed 0: test MAE 18.1038, best validation MAE 18.125, stopped at epoch 132.
- In the only completed paired seed, Proposed did not beat MAGE, Shared or Reversed. These partial results cannot establish five-seed performance.
- PEMS08: 0/30 completed at the time of this snapshot. The snapshot is not a live status; consult the server for updated results.
Comparability note: The original MAGE paper (12 -> 12), Hoang's earlier experiments (different features/preprocessing), and this flow-only matched 96 -> 12 study use different protocols and should not be directly compared as if they were identical experiments.
Follow-up work (not implemented; decide after the 60-run evaluation)
- [ ] Finish and analyze all matched runs, including paired-seed wins and parameter/compute comparisons.
- [ ] Compare early-fused multiscale features with separate temporal streams.
- [ ] Run a no-PDM ablation to isolate temporal mixing effects.
- [ ] Diagnose excessive propagation/smoothing and, if needed, test reduced schedules as new exploratory variants.
- [ ] Add TimeMixer/STID external baselines if required for the report.
- [ ] Update Overleaf to describe the actual PDM, shared routing, positive/negative graph experts, and matched preprocessing/training.
Documentation rule: Keep completed runs, partial snapshots, and planned experiments separate. Preserve the original 60-run results when testing new versions. Changes selected after inspecting test data should be labeled exploratory and independently validated.
