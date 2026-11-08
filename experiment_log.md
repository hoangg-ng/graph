# Experiment log: multi-scale MAGE study

**Record updated:** 2026-11-08  
**Repository branch verified:** `Jiwon-work` (`2c0dbbc110bb9a7f3bba3c2efe746db728ada300`)  
**Repository:** https://github.com/hoangg-ng/graph

This file records implementations, evaluation protocols, numbers stored in result files, and identified verification limits. It does not assign qualitative ratings to architectures or infer causal effects from non-isolated comparisons.

## Recording conventions

- Unless explicitly identified as validation MAE, **MAE means the test-set MAE** stored in the corresponding experiment output. For the newer 96→12 evaluations, this is the masked MAE over finite targets with positive observed flow values.
- `mean ± SD` is the arithmetic mean and **sample** standard deviation across independent seeds. A single-run value has no reported SD.
- A signed difference `ΔMAE = MAE(A) − MAE(B)` is negative when A has a lower recorded error. A relative decrease is `(MAE(B) − MAE(A)) / MAE(B) × 100%`. These definitions state arithmetic differences, not statistical significance.
- `k/n` paired counts the number of seed IDs for which the named first model has lower MAE than the stated comparator. An entry with two seeds is not a five-seed statistic.
- **Do not pool** the early 12→12 experiments, earlier 96→12 Step-8 experiments, the matched 96→12 test evaluations, and the validation-only Grouped ablations. Their protocol, code version, or split differs.
- Source files are linked relative to the repository root. Files in `results/` are under `MAGE_MS/results/` unless a full path is shown.
- Earlier observations are retained as historic entries. Where the underlying result summary was not independently recomputed, the entry says so; file presence alone is not proof of model equivalence.

---

## 2026-10-18 — Step 1. MAGE reproduction

The prior experiment log recorded three-seed averages for the official MAGE implementation and the local reimplementation:

| Implementation | PEMS08 test MAE | PEMS04 test MAE | Seed count |
|---|---|---|---|
| Official implementation | 13.95 | 18.29 | 3 |
| Local reimplementation (linear encoder) | 13.96 | 18.25 | 3 |

The original values above are reproduced from the pre-existing log and are rounded to two decimal places; this revision did not independently reconstruct the three official-run averages. The corresponding run records are in `MAGE_MS/official_repro/experiments/mage/` and `MAGE_MS/results/logs/`. Paper-reported values are not mixed with this local reproduction record.

## 2026-10-18 — Step 2. Temporal encoder comparison (12→12)

- Input: 12 five-minute observations; output: 12 future observations.
- Data: PEMS04 and PEMS08.
- The values below are five-seed test-MAE means and SDs from the earlier log and the `results/runs/` output set.

| Encoder | Implementation | PEMS08 test MAE | PEMS04 test MAE |
|---|---|---|---|
| `linear` | MAGE linear encoder | 13.96 ± 0.03 | 18.25 ± 0.05 |
| `mslin` | Pooled temporal views followed by linear projection | 13.97 ± 0.02 | 18.30 ± 0.06 |
| `m3` | Same-width MLP encoder | 13.98 ± 0.04 | 18.21 ± 0.04 |
| `m4` | Multiscale encoder with global gate | 13.98 ± 0.05 | 18.21 ± 0.04 |
| `m5` | Multiscale encoder with node-specific gate | 14.06 ± 0.02 | 18.25 ± 0.02 |

Arithmetic differences using the rounded values in this table: `m5 − linear` = +0.10 MAE (PEMS08) and 0.00 MAE (PEMS04); `m5 − m4` = +0.08 MAE (PEMS08) and +0.04 MAE (PEMS04). Exact unrounded differences should be calculated from the per-seed JSON files if required.

Files: `MAGE_MS/results/runs/PEMS{04,08}_T12_*_s{0..4}.json`.

## 2026-10-18 — Step 3. Computational scaling measurements

Source: `MAGE_MS/results/cost_bs16_250-500-1000-2000-4000-8000-16000.json`.

The following entries are recorded values from the `linear` encoder and the two contraction methods. GPU runtime and memory measurements depend on the measurement environment.

| Node count N | Local contraction time (ms) | Local memory (MiB) | Released einsum time (ms) | Released einsum memory (MiB) |
|---|---|---|---|---|
| 250 | 19.8647 | 313.0767 | 20.8864 | 282.3149 |
| 1,000 | 24.5690 | 1,027.8354 | 36.6156 | 1,182.72998 |
| 8,000 | 165.3036 | 7,604.4902 | 1,083.9945 | 21,624.8604 |
| 16,000 | 324.9104 | 15,133.2832 | not recorded | not recorded |

The released-einsum experiment has a missing time and memory value at N=16,000; the stored entry does not supply a complete numerical measurement for that configuration. The original log also reported fitted scaling slopes of approximately 0.93 for the local contraction and 1.63 for the released contraction. Those fitted slopes were not recomputed in this revision.

Other source: `MAGE_MS/results/cost_bs64_170-307.json`.

## 2026-10-18 — Steps 4–5. Expert routing and traffic regimes

Source: `MAGE_MS/results/post_summary.md` (`linear`, `mslin`, `m3`, `m4`, `m5`). The following table records selected regime-specific test MAE values and the difference `m5 − linear` using the two-decimal numbers saved in that summary.

| Dataset | Regime | Linear MAE | M5 MAE | ΔMAE (M5−Linear) |
|---|---|---|---|---|
| PEMS04 | Congestion onset | 34.64 | 34.79 | +0.15 |
| PEMS04 | Recovery | 34.01 | 34.27 | +0.26 |
| PEMS04 | Congested | 35.96 | 36.30 | +0.34 |
| PEMS08 | Congestion onset | 31.81 | 31.87 | +0.06 |
| PEMS08 | Recovery | 29.35 | 29.61 | +0.26 |
| PEMS08 | Congested | 28.12 | 28.37 | +0.25 |

The original `post_summary.md` reports PEMS04 M5-versus-linear paired seed counts of 0/5 for onset, 0/5 for recovery, and 0/5 for congested periods. The displayed difference for PEMS04 onset is +0.15 from the rounded cell values; the source report separately writes **+0.14**, reflecting calculation before display rounding. That rounding discrepancy is retained here instead of treating the displayed subtraction as the original unrounded difference.

The report also stores the Jensen–Shannon divergence (JSD) of global expert usage for `linear` versus `m5` over the three spatial blocks:

| Dataset | Block 1 | Block 2 | Block 3 |
|---|---|---|---|
| PEMS04 | 0.0334 | 0.0198 | 0.0334 |
| PEMS08 | 0.0110 | 0.0225 | 0.0214 |

The original report also contains model-wise routing entropy, expert shares and gate values. No causal relation between routing entropy and forecast error is asserted in this record.

## 2026-10-18 — Error diagnostics

Source files: `MAGE_MS/results/diag.json`, `MAGE_MS/results/diag_space.json`, and `MAGE_MS/results/diag_lag.json`.

| Dataset | MAGE test MAE (diagnostic run) | MAE of 5-seed ensemble | Largest 10% errors: share of total absolute error | Residual correlation: road-neighbor lag 0, horizon 1 | Residual correlation: random pairs lag 0, horizon 1 |
|---|---|---|---|---|---|
| PEMS04 | 18.1843 | 18.0621 | 40.72% | 0.1162 | 0.0148 |
| PEMS08 | 13.9710 | 13.7487 | 39.78% | 0.1094 | 0.0393 |

The diagnostic files use the earlier experiment implementation. They are not output from the later matched `eval_proposal96` protocol. The stored lag analysis evaluates offsets from −30 to +30 minutes in five-minute increments. The fast-variation cross-correlation arrays have their largest stored value at +5 minutes for both datasets; the sign of a lag follows the original diagnostic script's convention. No irreducible noise-floor estimate is asserted from these outputs.

## 2026-10-19 — Step 7. Historical window length comparison

- Forecast horizon: 12 steps.
- Historical input length: 12, 96 or 288 steps.
- Models: `linear`, `m3`, `m4`, `m5`.
- Configuration count: 2 datasets × 3 input lengths × 4 encoders × 5 seeds = **120 runs**.
- Table values: five-seed mean **test MAE** recorded in the prior log and backed by the `MAGE_MS/results/long/` per-run output set.

| Dataset | Input steps | Linear | M3 | M4 | M5 |
|---|---|---|---|---|---|
| PEMS08 | 12 | 13.880 | 13.977 | 13.904 | 13.981 |
| PEMS08 | 96 | 13.802 | 13.990 | 13.854 | 14.024 |
| PEMS08 | 288 | 13.774 | 14.345 | 14.129 | 14.157 |
| PEMS04 | 12 | 18.345 | 18.211 | 18.222 | 18.253 |
| PEMS04 | 96 | 18.270 | 18.331 | 18.221 | 18.293 |
| PEMS04 | 288 | 18.248 | 18.795 | 18.590 | 18.649 |

For PEMS08 linear MAGE, the recorded mean at input length 12 is 13.880 and at input length 288 is 13.774 (difference −0.106). For PEMS04 linear MAGE, the corresponding values are 18.345 and 18.248 (difference −0.097). The earlier log reports that many long-window linear runs reached the maximum 300 training epochs; check each `epochs_run` before attributing a difference to early stopping.

Files: `MAGE_MS/results/long/PEMS{04,08}_T{12,96,288}_{linear,m3,m4,m5}_s{0..4}.json`.

## 2026-10-27 — Spatial correlation pre-check

Source: `MAGE_MS/results/precheck_hops.json` (unadjusted correlations) and `MAGE_MS/results/precheck_hops_nocommon.json` (global/common component removed). Entries below are correlations computed for pairs at the indicated shortest-path hop count; scale factors 1, 2, 4 and 8 correspond to 5-, 10-, 20- and 40-minute sampling.

| Dataset | Processing | Hop distance | Scale ×1 | Scale ×2 | Scale ×4 | Scale ×8 |
|---|---|---|---|---|---|---|
| PEMS04 | Unadjusted | 1 | 0.4050 | 0.4823 | 0.5329 | 0.5670 |
| PEMS04 | Unadjusted | 3 | 0.3352 | 0.3925 | 0.4397 | 0.4758 |
| PEMS08 | Unadjusted | 1 | 0.2713 | 0.3348 | 0.3746 | 0.4035 |
| PEMS08 | Unadjusted | 3 | 0.1998 | 0.2450 | 0.2841 | 0.3163 |
| PEMS04 | Common component removed | 1 | 0.2332 | 0.2852 | 0.3152 | 0.3328 |
| PEMS04 | Common component removed | 3 | 0.0900 | 0.1075 | 0.1245 | 0.1370 |
| PEMS08 | Common component removed | 1 | 0.1320 | 0.1634 | 0.1784 | 0.1865 |
| PEMS08 | Common component removed | 3 | 0.0218 | 0.0262 | 0.0318 | 0.0353 |

The source files contain values for hops 1–8. This table selects hops 1 and 3 for compact presentation; no estimated propagation range is asserted.

## 2026-10-31 — Step 8. Earlier scale-dependent spatial mixing prototype

**Protocol:** PEMS08, 96 input steps → 12 output steps, earlier `main` implementation; four temporal scales (5, 10, 20, 40 minutes). Positive propagation used the earlier lazy operator `P=(I+S₊)/2`, with repeated powers `P^k`. It is not the same operator/protocol as the November `ProposalMAGE` implementation.

**Stored test MAE:**

| Model | Positive-power schedule | Completed seeds | Test MAE mean ± sample SD |
|---|---|---|---|
| `mage` | MAGE code check | 1 | 13.7358 (seed 0) |
| `shared` | (1,1,1,1) | 5 | 13.7866 ± 0.0858 |
| `proposed` | (1,2,4,8) | 5 | 13.7700 ± 0.0986 |
| `uniform` | Global mean | 2 | 13.9618 ± 0.0084 |
| `nomix` | No spatial propagation | 2 | 13.8199 ± 0.0047 |

For the five paired seeds, `proposed − shared` mean test MAE = **−0.0165** and Proposed has a lower MAE in **2/5** paired seeds. The single-seed MAGE code check cannot be interpreted as a five-seed baseline. Reversed and Independent were not present as completed `results/step8/` outputs.

Source: `MAGE_MS/results/step8/PEMS08_T96_*_s*.json`.

---

## 2026-11-03 — V1. ProposalMAGE implementation and preprocessing

**Code and recorded protocol changes**

- Added separate `MAGE_MS/proposal_model.py`, `MAGE_MS/prep_proposal.py`, training/evaluation scripts, and automated tests.
- Task changed to **96 historical inputs → 12 future targets** at five-minute sampling.
- Proposal input channels: traffic flow, time-of-day, and day-of-week; the target is traffic flow. Occupancy and speed are not used as proposal input channels.
- Data boundaries: original timeline split into train/validation/test segments at **60% / 20% / 20%**. A target window belongs only to its designated segment, while its history can include earlier timestamps.
- Flow normalization parameters are fitted from the training segment and reused for validation and test.
- Multiscale temporal streams: 96, 48, 24 and 12 steps at the four time resolutions. PDM uses a moving-average trend/seasonal decomposition with 2 mixing blocks, hidden width 16 and kernel 5.
- Spatial configuration: 3 blocks, 16 learned graph experts, Top-4 routing. The schedule applies to positive spatial operator repetitions; a separate negative operator contribution is also included.
- Predictions from the four streams are combined after per-stream heads in the non-Grouped variants.

| Arm | Positive propagation schedule / operator |
|---|---|
| `shared` | (1,1,1,1), shared learned operator per block |
| `proposed` | (1,2,4,8), shared learned operator per block |
| `reversed` | (8,4,2,1), shared learned operator per block |
| `independent` | Separate learned spatial expert pools by scale |
| `uniform` | Global node mean |

The November code applies direct learned positive operator powers `S₊^k`. The earlier Step-8 prototype applies `(I+S₊)/2` powers. Their MAE tables are kept separate.

Code references: `MAGE_MS/prep_proposal.py`, `MAGE_MS/proposal_model.py`, `MAGE_MS/test_proposal.py`, `MAGE_MS/test_proposal_prep.py`.

Recorded commits: `a235447f1b53a6fab3142ed0ad2a291366671fa8`, `f9bf4ab593bf3186be8190730bd853fada2815b0`.

## 2026-11-05 — V2. Signed graph operator parameter change

- The initial ProposalMAGE version reused matrices `E1` and `E2` for the positive and negative graph operators.
- Commit `4c6854bfa98c2e3830fe82e24c592399e422cf67` separated the matrices for positive and negative operators.
- Unit tests for operator construction were updated.
- **Recorded paired MAE comparison with all other settings held fixed:** none in this project log. No numerical before/after model effect is assigned to this change.

Sources: `MAGE_MS/proposal_model.py`, `MAGE_MS/test_proposal.py`, repository commit history.

## 2026-11-06 — V3. Hyperparameter selection and scheduled matched evaluation

**Selection:** Ten candidate configurations of PEMS04 Shared (seed 0), evaluated using maximum 60 epochs and patience 15. Selection used validation MAE and did not use the test split.

| Selected trial | Learning rate | Weight decay | Dropout | Validation MAE | Epochs run |
|---|---|---|---|---|---|
| 1 | 0.005 | 0.0001 | 0.3 | 18.1776924 | 60 |

- Selected recipe: `MAGE_MS/results/tune_shared/selected_recipe.json`.
- Six evaluated variants: MAGE, Shared, Proposed, Reversed, Independent and Uniform.
- Planned matched run count: **2 datasets × 6 variants × 5 seeds = 60 jobs** (see completed V4 results below).
- Main training limits: 300 epochs, patience 30, batch size 64; the selected learning rate, decay and dropout above.
- The original November 6 progress snapshot contained 6/60 completed jobs; this is a historical snapshot and is superseded by the completed results in V4.

Source: selected-recipe JSON and `MAGE_MS/run_matched_eval.py`.

## 2026-11-07 — V4. Matched six-model evaluation (60 runs)

**Task and sample definition:** PEMS04 and PEMS08, 96→12, same prepared datasets and split, seeds 0–4 per arm. **Metric:** masked test MAE; additional stored metrics include RMSE and MAPE.

| Dataset | Arm | Completed | Test MAE mean ± SD | Test RMSE mean | Test MAPE mean (%) |
|---|---|---|---|---|---|
| PEMS04 | mage | 5/5 | 18.0921 ± 0.0217 | 29.9090 | 12.2586 |
| PEMS04 | shared | 5/5 | 18.1479 ± 0.0559 | 29.9452 | 12.3688 |
| PEMS04 | proposed | 5/5 | 18.1301 ± 0.0704 | 29.9114 | 12.3636 |
| PEMS04 | reversed | 5/5 | 18.1357 ± 0.0499 | 29.9427 | 12.3537 |
| PEMS04 | independent | 5/5 | 18.1135 ± 0.0288 | 29.9009 | 12.3217 |
| PEMS04 | uniform | 5/5 | 17.9985 ± 0.0386 | 29.7470 | 12.3513 |
| PEMS08 | mage | 5/5 | 13.8126 ± 0.0199 | 23.1668 | 9.1297 |
| PEMS08 | shared | 5/5 | 13.9298 ± 0.0242 | 23.3731 | 9.2375 |
| PEMS08 | proposed | 5/5 | 13.9115 ± 0.0309 | 23.3788 | 9.1439 |
| PEMS08 | reversed | 5/5 | 13.9294 ± 0.0306 | 23.4124 | 9.1962 |
| PEMS08 | independent | 5/5 | 13.9131 ± 0.0122 | 23.2783 | 9.2087 |
| PEMS08 | uniform | 5/5 | 13.8273 ± 0.0680 | 23.3020 | 9.0444 |

| Comparison A−B | PEMS04 ΔMAE | PEMS08 ΔMAE |
|---|---|---|
| Proposed − Shared | -0.0177 | -0.0183 |
| Proposed − Reversed | -0.0056 | -0.0179 |
| Proposed − MAGE | +0.0381 | +0.0988 |

Source files: `MAGE_MS/results/eval_proposal96/` (60 per-seed JSON files), `matched_mae_summary.csv`, `matched_mae_per_seed.csv`, and `matched_mae_table.md`.

The original prespecified comparison used a **1% reduction in mean test MAE** with at least **4/5 paired-seed wins** as its decision rule. The report gives Proposed vs MAGE as **1/5** (PEMS04) and **0/5** (PEMS08). These counts are a documented criterion and observed counts, not an independent significance test.

## 2026-11-07 — V5–V6. Follow-up variants and profiling implementation

The earlier implementation record described these additions:

| Variant | Recorded implementation |
|---|---|
| `mage_features` | MAGE with moving averages, first differences and time-of-day features constructed within historical windows |
| `no_spatial` | Four temporal streams without learned cross-node propagation |
| `grouped` | Four PDM-derived representations concatenated/projected before one spatial stream and one output head |
| `proposed_no_pdm` | Independent linear encoders substituted for the PDM temporal modules in the Proposed architecture |

| Implementation | Positive graph operator applications | Negative graph operator applications |
|---|---|---|
| Shared | 12 | 12 |
| Proposed | 45 | 12 |
| Reversed | 45 | 12 |
| Independent | 12 | 12 |
| Uniform | 0 | 0 |
| No-Spatial | 0 | 0 |
| Grouped | 3 | 3 |
| Proposed-No-PDM | 45 | 12 |

The operator counts describe `ProposalMAGE` factorized operator calls in one model forward pass; the native MAGE implementation is not counted by the same method. Separate versions of the original code are documented by the source fingerprints in the result JSON files.

Files introduced or used: `MAGE_MS/profile_models.py`, `MAGE_MS/report_complexity.py`, `MAGE_MS/test_complexity.py`, `MAGE_MS/run_followup_eval.py`.

Recorded implementation commits: `b77d1623c42aec782b78088daa1ffd237025445c`, `3cd4007377b1892f3ed65992d7732feeffc32785`, `dac6bbef9f893557dfaa7b8045c83ba5a89b73e8`, `ffd5d33357182cbab40e6fdbfb72b32cf11710d4`.

The follow-up run plan was **4 variants × 2 datasets × 5 seeds = 40 jobs**. Its completion is recorded in V7.

## 2026-11-07 — V7. Follow-up four-model evaluation (40 runs)

**Evaluation:** PEMS04 and PEMS08, 96→12, 5 seeds (0–4) for each of `mage_features`, `no_spatial`, `grouped`, and `proposed_no_pdm`. The same selected Shared hyperparameter recipe was used; output source fingerprints are stored separately from the initial 60-run set.

| Dataset | Arm | Completed | Masked test MAE mean ± SD |
|---|---|---|---|
| PEMS04 | mage_features | 5/5 | 18.0183 ± 0.0217 |
| PEMS04 | no_spatial | 5/5 | 18.2481 ± 0.0354 |
| PEMS04 | grouped | 5/5 | 17.8589 ± 0.0211 |
| PEMS04 | proposed_no_pdm | 5/5 | 18.0497 ± 0.0300 |
| PEMS08 | mage_features | 5/5 | 13.7602 ± 0.0146 |
| PEMS08 | no_spatial | 5/5 | 13.8168 ± 0.0313 |
| PEMS08 | grouped | 5/5 | 13.6781 ± 0.0369 |
| PEMS08 | proposed_no_pdm | 5/5 | 13.8876 ± 0.0320 |

| Comparison A−B | PEMS04 Δ test MAE | PEMS08 Δ test MAE |
|---|---|---|
| Grouped − MAGE | -0.2332 | -0.1345 |
| Grouped − Proposed | -0.2713 | -0.2334 |
| Proposed-No-PDM − Proposed | -0.0804 | -0.0239 |
| Grouped − MAGE+Features | -0.1594 | -0.0821 |

Relative MAE difference for Grouped versus MAGE, computed from these rounded/stored means: **−1.29% on PEMS04** and **−0.97% on PEMS08**, where the signed percentage is `(Grouped−MAGE)/MAGE × 100`. The 1.5% relative reduction target is a separate prespecified threshold; these observed percentages are listed without a qualitative classification.

The November 7 follow-up report records **40/40 completed**, and the original matched report records **60/60 completed**. Together, these two **matched 96→12 test-evaluation stages** contain 100 training jobs. Earlier historical encoder and long-window studies are **not** included in this subtotal.

Sources: `MAGE_MS/results/eval_followup96/followup_mae_interim.md`, `MAGE_MS/results/eval_followup96/PEMS*.json`, and V4 source files.

### 2026-11-07 — Seed-0 computational profiles for the 10 models

- One seed-0 checkpoint was profiled per model and dataset.
- Counted FLOPs are the operations supported by the selected PyTorch counting methods; the count does **not** include every possible operator.
- Latency is the reported median single-item inference latency (batch size 1). Throughput is samples per second at batch size 64.
- GPU device, CUDA/PyTorch versions, precision flags and warm-up details are stored in the per-profile outputs.

| Dataset | Model | Checkpoint set | Params (M) | Counted GFLOPs | Graph calls (+/−) | Latency B=1 (ms) | Throughput B=64 (samples/s) |
|---|---|---|---|---|---|---|---|
| PEMS04 | mage | original | 2.703 | 0.9035 | not comparable | 2.293 | 7939.5 |
| PEMS04 | shared | original | 2.995 | 4.1505 | 12/12 | 7.107 | 1157.7 |
| PEMS04 | proposed | original | 2.995 | 6.8063 | 45/12 | 9.620 | 915.6 |
| PEMS04 | reversed | original | 2.995 | 6.8063 | 45/12 | 9.655 | 914.0 |
| PEMS04 | independent | original | 8.654 | 4.1505 | 12/12 | 7.117 | 1153.7 |
| PEMS04 | uniform | original | 1.103 | 2.2152 | 0/0 | 3.811 | 1469.5 |
| PEMS04 | mage_features | follow-up | 2.765 | 0.9412 | not comparable | 2.270 | 7714.1 |
| PEMS04 | no_spatial | follow-up | 1.103 | 2.2152 | 0/0 | 3.748 | 1462.5 |
| PEMS04 | grouped | follow-up | 3.056 | 1.6528 | 3/3 | 3.815 | 1746.5 |
| PEMS04 | proposed_no_pdm | follow-up | 2.594 | 6.0576 | 45/12 | 8.180 | 1557.8 |
| PEMS08 | mage | original | 1.844 | 0.5003 | not comparable | 2.187 | 12623.7 |
| PEMS08 | shared | original | 2.136 | 2.2983 | 12/12 | 7.003 | 1989.3 |
| PEMS08 | proposed | original | 2.136 | 3.7689 | 45/12 | 9.413 | 1556.4 |
| PEMS08 | reversed | original | 2.136 | 3.7689 | 45/12 | 9.469 | 1554.9 |
| PEMS08 | independent | original | 5.269 | 2.2983 | 12/12 | 7.029 | 1989.6 |
| PEMS08 | uniform | original | 1.085 | 1.2267 | 0/0 | 3.829 | 2596.3 |
| PEMS08 | mage_features | follow-up | 1.906 | 0.5212 | not comparable | 2.206 | 12367.7 |
| PEMS08 | no_spatial | follow-up | 1.085 | 1.2267 | 0/0 | 3.717 | 2591.9 |
| PEMS08 | grouped | follow-up | 2.197 | 0.9152 | 3/3 | 3.860 | 3034.1 |
| PEMS08 | proposed_no_pdm | follow-up | 1.735 | 3.3543 | 45/12 | 7.994 | 2557.5 |

**Source fingerprints during profiling:** The six original variants were trained under code digest `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3`; the original profile report records **12/12 code-digest mismatches** between training checkpoint and profile source digest `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`. The follow-up profile report records **0** such mismatches. Both reports record that the data/split fingerprint check passed. A source digest mismatch is a recorded discrepancy; functional equivalence has not been established by the digest comparison alone.

Source reports:
- `MAGE_MS/results/complexity_original_v1/model_complexity_table.md` and `model_complexity_summary.json`.
- `MAGE_MS/results/complexity_followup_v1/model_complexity_table.md` and `model_complexity_summary.json`.

These are profiler measurements for saved seed-0 checkpoints; they are **not five-seed mean runtime profiles**.

## 2026-11-08 — V8. Grouped component ablation (validation only; 28 runs)

**New model arms and definitions**

| Arm | Temporal processing | Spatial processing | Output heads |
|---|---|---|---|
| `grouped` | Four PDM-derived streams, fused before spatial block | Learned positive/negative operators | 1 |
| `grouped_no_pdm` | Four independent linear scale encoders, fused before spatial block | Learned positive/negative operators | 1 |
| `grouped_no_spatial` | Four PDM-derived streams, fused before spatial block | No learned cross-node operator; node embeddings/prompts and feed-forward processing remain | 1 |
| `grouped_uniform` | Four PDM-derived streams, fused before spatial block | Global node-average mixing instead of learned graph | 1 |

**Training/evaluation protocol**: PEMS04 and PEMS08; 96 input → 12 output steps; 5-minute base interval; 4 temporal resolutions (5, 10, 20, 40 minutes); 300 maximum epochs, patience 30, batch size 64; learning rate 0.005, weight decay 0.0001, dropout 0.3; TF32=0. The experiment runner sets `validation_only=true`, result protocol `96to12_v1_validation_only`; no test-set error is recorded in these new runs.

- Phase A: four arms × two datasets × seeds 0 and 1 = **16 completed** JSON jobs; source `MAGE_MS/results/ablation_val96/`.
- Phase B: Grouped and Grouped-No-PDM × two datasets × seeds 2, 3 and 4 = **12 completed** JSON jobs; source `MAGE_MS/results/ablation_val96_s234/`.
- The `ablation_val96_5seed/` directory contains the **five-seed summary Markdown**, not another 20 unique new training jobs.
- All 28 individual JSON files report `completed=true`, code digest `34afb5738dfd08069551c6c8efd09dc314b46b5f9731ecb4a54915921e63c9cc`, and recipe digest `16b0eb133418cfc264306a6e1380d713c39ec6cf8ecbd6fc714a0ab37881e31f`.
- Dataset digest: PEMS04 `0926882c1bed456e5dddd2811bb29a4f5eb5ea5ebfe02cee319f8dfd432607fb`; PEMS08 `a537342a659680fedae25b300def0cd82a94fcc3e42f482a609bb1efac685014`.

### V8.1 Five-seed Grouped versus No-PDM — per-seed best validation MAE

| Dataset | Seed | Grouped | Grouped-No-PDM | ΔMAE (No-PDM−Grouped) |
|---|---|---|---|---|
| PEMS04 | 0 | 17.949080 | 18.022705 | +0.073625 |
| PEMS04 | 1 | 17.961552 | 18.049345 | +0.087793 |
| PEMS04 | 2 | 17.983175 | 18.010174 | +0.026999 |
| PEMS04 | 3 | 17.939528 | 18.028364 | +0.088836 |
| PEMS04 | 4 | 17.975574 | 18.011684 | +0.036110 |
| PEMS08 | 0 | 13.925466 | 13.790399 | -0.135067 |
| PEMS08 | 1 | 13.889094 | 13.904367 | +0.015273 |
| PEMS08 | 2 | 13.832620 | 13.925567 | +0.092947 |
| PEMS08 | 3 | 13.938129 | 13.842965 | -0.095164 |
| PEMS08 | 4 | 13.912772 | 13.895152 | -0.017620 |

| Dataset | Grouped, mean ± SD | Grouped-No-PDM, mean ± SD | ΔMAE (No-PDM−Grouped) | No-PDM paired lower-MAE seeds |
|---|---|---|---|---|
| PEMS04 | 17.9618 ± 0.0181 | 18.0245 ± 0.0159 | +0.0627 | 0/5 |
| PEMS08 | 13.8996 ± 0.0416 | 13.8717 ± 0.0547 | -0.0279 | 3/5 |

The calculations in the preceding tables use six-decimal display values transcribed from 28 JSON records. The repository five-seed report uses the original unrounded floats and reports PEMS04 17.9618 ± 0.0181 versus 18.0245 ± 0.0159 (No-PDM relative improvement −0.35%, 0/5 paired lower-MAE seeds) and PEMS08 13.8996 ± 0.0416 versus 13.8717 ± 0.0547 (+0.20%, 3/5 paired lower-MAE seeds). Minor last-digit differences can arise when calculating from six-decimal values.

### V8.2 Two-seed spatial ablations — best validation MAE

| Dataset | Arm | n | Validation MAE mean ± SD | ΔMAE vs 2-seed Grouped | Paired lower-MAE count vs Grouped |
|---|---|---|---|---|---|
| PEMS04 | grouped | 2 | 17.9553 ± 0.0088 | - | - |
| PEMS04 | grouped_no_spatial | 2 | 18.1361 ± 0.0104 | +0.1808 | 0/2 |
| PEMS04 | grouped_uniform | 2 | 18.0745 ± 0.0804 | +0.1192 | 0/2 |
| PEMS08 | grouped | 2 | 13.9073 ± 0.0257 | - | - |
| PEMS08 | grouped_no_spatial | 2 | 13.9705 ± 0.0669 | +0.0632 | 0/2 |
| PEMS08 | grouped_uniform | 2 | 14.0398 ± 0.0166 | +0.1325 | 0/2 |

The signed differences in this table are computed from the means rounded to four decimals as displayed. Grouped uses **only seeds 0–1** here, not its five-seed mean. Differences computed directly from unrounded per-seed JSON floats can differ by 0.0001 after final rounding.

### V8.3 Model parameters recorded in validation JSON

| Arm | PEMS04 parameters | PEMS08 parameters |
|---|---|---|
| `grouped` | 3,056,156 | 2,196,892 |
| `grouped_no_pdm` | 2,655,084 | 1,795,820 |
| `grouped_no_spatial` | 1,163,708 | 1,146,172 |
| `grouped_uniform` | 1,163,708 | 1,146,172 |

The spatial arms differ from Grouped in parameter count as well as the spatial operation. These comparisons do not keep trainable parameter count constant. No hypothesis-test p-values or confidence intervals are recorded for V8.

Files: `MAGE_MS/results/ablation_val96/ablation_best_val_mae_summary.md`, `MAGE_MS/results/ablation_val96_5seed/ablation_best_val_mae_summary.md`, `MAGE_MS/results/ablation_val96/ablation_manifest.json`, `MAGE_MS/results/ablation_val96_s234/ablation_manifest.json` and the per-seed JSONs.

**Validation–test distinction:** Grouped test MAE from V7 is 17.8589 (PEMS04) and 13.6781 (PEMS08); Grouped *best validation* MAE from V8 is 17.9618 and 13.8996, respectively. The V7 and V8 numbers have different evaluation partitions and must not be subtracted as model comparisons.

---

## Reproducibility status and file inventory (as of 2026-11-08)

| Evaluation stage | Datasets | Models / settings | Seeds per setting | Recorded jobs | Metric | Result directory |
|---|---|---|---|---|---|---|
| Early 12→12 encoder study | PEMS04, PEMS08 | 5 encoder variants | 5 | 50 | Test MAE | `MAGE_MS/results/runs/` |
| Long-input study | PEMS04, PEMS08 | 4 encoders × 3 input lengths | 5 | 120 | Test MAE | `MAGE_MS/results/long/` |
| Earlier Step-8 prototype | PEMS08 | MAGE (1 seed), Shared (5), Proposed (5), Uniform (2), No-mix (2) | varying | 15 | Test MAE | `MAGE_MS/results/step8/` |
| Matched ProposalMAGE | PEMS04, PEMS08 | 6 variants | 5 | 60 | Masked test MAE | `MAGE_MS/results/eval_proposal96/` |
| Follow-up models | PEMS04, PEMS08 | 4 variants | 5 | 40 | Masked test MAE | `MAGE_MS/results/eval_followup96/` |
| Grouped validation ablation | PEMS04, PEMS08 | 4 arms × 2 seeds + 2 arms × 3 more seeds | 2 or 5 | 28 | Best validation MAE | `MAGE_MS/results/ablation_val96*/` |

- These counts apply only to the six separately identified sets. They exclude hyperparameter search trials, official-reproduction training, isolated debugging runs and computational profiles. They are not stated as an exhaustive total for the repository.
- The 96→12 matched test-evaluation stages total **60 + 40 = 100** jobs. The subsequent validation ablation contains **28** new jobs. The 96→12 subtotal across these three stages is **128**, counting validation and test runs separately but not counting duplicated summary copies as new jobs.
- Source digests are not interchangeable: original 60-run training digest begins `9c47e939…`, follow-up 40-run digest begins `4d4fd58b…`, and validation ablation training digest begins `34afb573…`.
- Original complexity profiling has **12** checkpoint/profile code-digest mismatches recorded; follow-up profiling reports **0**.
- The record contains observed figures and execution metadata; it does not establish a statistically significant architectural effect, a capacity-controlled comparison, or performance on external datasets.
- The following items are **not recorded as completed in the result directories reviewed**: a new five-seed test evaluation of Grouped-No-PDM, five-seed spatial ablation results for Grouped-No-Spatial/Grouped-Uniform, and a parameter-matched isolation of fusion order or prediction-head count. This statement describes file evidence, not a claim that no work exists outside the reviewed repository.

**Principal code locations:** `MAGE_MS/proposal_model.py`, `MAGE_MS/train_eval.py`, `MAGE_MS/run_matched_eval.py`, `MAGE_MS/run_followup_eval.py`, `MAGE_MS/run_ablation_eval.py`, `MAGE_MS/report_matched_eval.py`, `MAGE_MS/report_grouped_ablation.py`, `MAGE_MS/test_proposal.py`, `MAGE_MS/test_grouped_ablation.py`.

## 2026-11-08 · Unified Grouped/Slim implementation and pilot launch

- New isolated branch `1108_branches_unified`, based on Jiwon-work `b5d4d81`;
  Slim reference `study-update-2026-11-07` at `76a7d67`. Existing worktrees and
  results were preserved.
- Added `grouped_slim` (fusion → three Slim blocks) and `slim_then_grouped`
  (the same three Slim blocks reused over four scales → fusion). Both keep
  PDM, prompt, fusion dimensions and one head. Each has three theta=0.5
  parameters, applied through tanh, and zero graph experts/graph calls.
- Focused CPU suite: 41 tests passed, zero skipped, including actual trained
  checkpoint regression for the original six arms and Grouped. Four legacy
  stream functional checks passed on CPU. An expanded old runner test has a
  pre-existing missing-function/43-versus-46-job mismatch, documented in
  `MAGE_MS/UNIFIED_SLIM_COMPARISON.md`; its old experiment plan was not changed.
- Architecture-only counts (untrained weights): A/B both 1,163,711 parameters
  on PEMS04 and 1,146,175 on PEMS08. B's counted FLOPs are about 1.93× A's;
  these are not accuracy, latency or memory results.
- Pilot completed: **16/16 validation-only jobs**, two datasets × four arms × seeds 0/1,
  one worker per initially idle RTX 3090. Supplemental validation RMSE/MAPE/horizon metrics
  were evaluated from all sixteen saved best checkpoints. No test evaluation was performed.
- The separate 40-job, five-seed test plan was dry-run only. Both new arms and
  a newly trained Grouped baseline remain in that plan.

### Unified pilot results (validation only, two seeds)

| Dataset | Arm | Validation MAE mean ± SD | Validation RMSE | Validation MAPE (%) | Params | Counted GFLOPs | Latency B=1 ms | Peak B=64 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PEMS04 | grouped | 17.9553 ± 0.0088 | 29.9517 | 12.0446 | 3,056,156 | 1.652829 | 3.861 | 1050.9 |
| PEMS04 | grouped_slim | 17.9856 ± 0.0065 | 30.0381 | 12.0372 | 1,163,711 | 1.166187 | 2.493 | 1043.7 |
| PEMS04 | slim_then_grouped | 18.1710 ± 0.0247 | 30.3669 | 12.1252 | 1,163,711 | 2.252643 | 3.911 | 1043.7 |
| PEMS04 | grouped_uniform | 18.0745 ± 0.0804 | 30.0279 | 12.1544 | 1,163,708 | 1.166187 | 2.490 | 1043.7 |
| PEMS08 | grouped | 13.9073 ± 0.0257 | 24.1263 | 11.0331 | 2,196,892 | 0.915247 | 3.780 | 584.1 |
| PEMS08 | grouped_slim | 13.9799 ± 0.0442 | 24.2874 | 11.5479 | 1,146,175 | 0.645772 | 2.455 | 580.1 |
| PEMS08 | slim_then_grouped | 14.0675 ± 0.0275 | 24.2403 | 11.0589 | 1,146,175 | 1.247392 | 3.879 | 580.1 |
| PEMS08 | grouped_uniform | 14.0398 ± 0.0166 | 24.0822 | 11.0468 | 1,146,172 | 0.645772 | 2.481 | 580.1 |

- Costs are saved-checkpoint inference measurements from seed 0 only, using FP32/TF32 off, batches 1/64, 20 warm-ups and 100 repetitions. Counted FLOPs cover registered formulas.
- Full paired A/B, Grouped-relative and resource comparisons: `MAGE_MS/results/unified_slim_val96_report/`. These are exploratory validation results; the formal 40 test jobs remain unexecuted.
- Initial automated publication encountered missing HTTPS credentials and connected-app HTTP 403. The implementation branch was subsequently published from the user's terminal; `origin/1108_branches_unified` was verified at `4e1298a` before preparing the completed result commit.
