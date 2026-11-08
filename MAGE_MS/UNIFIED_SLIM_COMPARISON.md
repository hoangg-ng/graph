# Grouped and Slim: fusion before or after spatial processing

## Scope and source revisions

Work branch: `1108_branches_unified`, in a separate worktree at
`/home/user/Workspace/KIMJIWON/graph-unified-slim`.

- Grouped baseline: `origin/Jiwon-work`, commit `b5d4d818ff1e0673396ec1c74dc59cb32c2f7e55`.
- Slim reference: `origin/study-update-2026-11-07`, commit `76a7d6717282a68840d18451038ac632c8ad8fc4`.
- Original worktrees, their uncommitted edits, data and results are preserved.
  This is an additive implementation, not a merge of the two source branches.

The research question is whether fusing four temporal representations before
spatial processing or after spatial processing gives the better accuracy/cost
tradeoff. Both models retain Grouped's four resolutions (5/10/20/40 minutes),
PDM encoder, sensor/calendar prompt, model dimension, fusion projection and
single prediction head. Historical Slim results used another training recipe
and TF32 setting; they are background evidence, not directly comparable runs.

## Common Slim block

For hidden state `H[b,n,d]`, block `j` computes:

```text
X = LayerNorm1_j(H)
G = mean(X, dim=1, keepdim=True)
c_j = tanh(theta_j), theta_j initially 0.5
Z = H + c_j * G
F_j(H) = Z + SwiGLU_j(LayerNorm2_j(Z))
```

There are exactly three independent scalar parameters `theta_j`. LayerNorm,
SwiGLU and residual order follow Grouped. No expert pool, router, E1/E2,
lambda logits, top-K routing or signed graph propagation is constructed.
Training returns `(prediction, zero_counts[3, experts])`; zero counts leave
the existing balancing bias unchanged. Inference returns only predictions.

## A: grouped_slim

Let `H_m = PDM_m(input)` and `P = Prompt_Pool(input)`.

```text
U = group_projection(concat(H_1, H_2, H_3, H_4)) + P
Y = head(F_3(F_2(F_1(U))))
```

Each block executes once: three Slim means and three FFNs. Fusion precedes
spatial processing; prompt is added to the fused representation.

## B: slim_then_grouped

```text
U_m = F_3(F_2(F_1(H_m + P))), m = 1..4
Y = head(group_projection(concat(U_1, U_2, U_3, U_4)))
```

All four streams reuse the **same three block objects**, including their
LayerNorm, FFN and theta parameters. There are twelve block/FFN executions,
but only three registered Slim coefficients. Fusion occurs after spatial
processing; prompt participates separately in each stream.

A and B have identical parameter shapes/counts, but different forward paths.
Block/FFN executions, prompt placement and nonlinear processing relative to
fusion all change. Differences cannot be attributed to fusion location alone.

## Architecture-only cost checkpoint

The following CPU counts use the selected dimensions (D=128, PDM width=16,
depth=2), actual dataset node counts and batch 1. Weights are **untrained**;
these counts do not establish accuracy or measured inference speed/memory.
Raw metadata: `results/unified_slim_complexity/structure_counts.json`.

| Dataset | Arm | Params | Counted GFLOPs B=1 | Graph calls | Slim mean calls |
|---|---|---:|---:|---:|---:|
| PEMS04 | grouped | 3,056,156 | 1.652829 | 6 | 0 |
| PEMS04 | grouped_slim | 1,163,711 | 1.166187 | 0 | 3 |
| PEMS04 | slim_then_grouped | 1,163,711 | 2.252643 | 0 | 12 |
| PEMS04 | grouped_uniform | 1,163,708 | 1.166187 | 0 | 0 |
| PEMS08 | grouped | 2,196,892 | 0.915247 | 6 | 0 |
| PEMS08 | grouped_slim | 1,146,175 | 0.645772 | 0 | 3 |
| PEMS08 | slim_then_grouped | 1,146,175 | 1.247392 | 0 | 12 |
| PEMS08 | grouped_uniform | 1,146,172 | 0.645772 | 0 | 0 |

Both Slim arms reduce parameters by approximately 61.92% on PEMS04 and 47.83%
on PEMS08 relative to Grouped. B has about 1.93 times A's counted FLOPs.
Grouped-Uniform still executes unweighted node averaging; its Slim mean-call
count of zero does not mean there is no spatial averaging. Registered FLOP
formulas omit some reductions and elementwise work, so identical counted
FLOPs for A/Uniform do not establish identical full operation counts.

## Verification

The focused suite completed **41 tests, zero skips**, including trained
checkpoint regression for the six original arms and Grouped on both datasets.
Four legacy `streams.py` functional checks also passed on CPU. Its standalone
GPU scaling benchmark is deferred while pilot workers occupy all four GPUs;
the benchmark calls CUDA unconditionally and cannot run with GPUs hidden.

- Unit tests check four PDM output shapes, batch 1/2 predictions `[B,12,N,1]`,
  scalar initialization, exact node-mean/tanh/residual arithmetic, finite
  gradients in theta/PDM/fusion/FFNs, zero routing counts, equal A/B parameter
  counts and distinct predictions with identical weights.
- Forward hooks verify A's fusion→3 blocks and B's 12 blocks→fusion, including
  prompt placement and object reuse.
- Prior-arm output tests load the baseline commit's implementation and the
  same weights/input, rather than comparing the current model to itself.
- Trained regression can read the original checkpoints through environment
  variables, without copying or editing the original results.
- `report_unified_slim.py` refuses mixed code/data/configuration and mismatched
  checkpoint profiles. Validation reports use only `best_val_mae`; test RMSE,
  MAPE and 15/30/60-minute metrics remain unavailable until test evaluation.
  `eval_unified_slim_validation.py` can separately compute **validation** RMSE,
  MAPE and horizon errors from the same saved best checkpoint; its MAE must
  agree with the recorded selection metric. Two supplemental tests verify
  validation-only evaluation and matching provenance. Training JSON/weights
  are never overwritten by this evaluator.

An additional old `test_run_proposal.py` fails on the baseline commit because
it imports the absent `paired_mae_results` and expects 46 jobs while the old
runner defines 43. That historical experiment plan is preserved; this failure
is separate from the unified model and its scheduler.

## Pilot and formal protocol

Ubuntu, Conda `jiwon-mage`, PyTorch 2.5.1+cu121/CUDA 12.1 and four RTX 3090
devices were confirmed. Existing `graph/MAGE_MS/data_proposal96` is read
directly; normalization, windows and train/validation/test splits are unchanged.

Selected recipe: lr=0.005, wd=0.0001, dropout=0.3, D=128, PDM width/depth=16/2,
moving average=5, batch=64, max epochs=300, patience=30, TF32=0. The scheduler
records recipe/code/data/configuration provenance and reuses only matching
completed results or compatible resume snapshots.

The pilot consists of 2 datasets × 4 arms × seeds 0,1 = **16 validation-only
jobs**. It was launched with one job per initially idle GPU. Trial selection
and pilot reporting do not evaluate test labels or test MAE. Final pilot
results and trained-checkpoint costs will be recorded after jobs finish;
pending runs are not reported as completed.

The formal plan is 2 datasets × 4 arms × seeds 0..4 = **40 new jobs**, including
a freshly trained Grouped baseline in the same code version. Both Slim arms
remain in the plan; no winner is selected from the pilot. Formal training is
prepared but is not started automatically.

## Reproduction commands

Run from this worktree's `MAGE_MS/`, with the `jiwon-mage` environment active.
Use an existing prepared-data path appropriate for the host:

```bash
DATA_DIR=/home/user/Workspace/KIMJIWON/graph/MAGE_MS/data_proposal96

# Pilot: 16 jobs; dry-run is the default.
python run_unified_slim_eval.py --validation-only --seeds 0 1 --data-dir "$DATA_DIR" --dry-run
python run_unified_slim_eval.py --validation-only --seeds 0 1 --data-dir "$DATA_DIR" \
  --gpus 0 1 2 3 --slots-per-gpu 1 --execute

# Formal: 40 jobs, up to 4 GPUs, one worker per GPU, max 300 epochs each.
# Prepared only; execute deliberately after reviewing the pilot and resources.
python run_unified_slim_eval.py --data-dir "$DATA_DIR" --dry-run
python run_unified_slim_eval.py --data-dir "$DATA_DIR" \
  --gpus 0 1 2 3 --slots-per-gpu 1 --execute

# Profile existing pilot checkpoints, without further training.
python profile_models.py --datasets PEMS04 PEMS08 \
  --arms grouped grouped_slim slim_then_grouped grouped_uniform --seeds 0 \
  --results-dir results/unified_slim_val96 --data-dir "$DATA_DIR" \
  --out-dir results/unified_slim_complexity --device cuda:0 \
  --batch-sizes 1 64 --warmup 20 --repeats 100

python eval_unified_slim_validation.py --results-dir results/unified_slim_val96 \
  --data-dir "$DATA_DIR" --out-dir results/unified_slim_val96_metrics --seeds 0 1

python report_unified_slim.py --validation-only \
  --results-dir results/unified_slim_val96 --profile-dir results/unified_slim_complexity \
  --validation-metrics-dir results/unified_slim_val96_metrics \
  --out-dir results/unified_slim_val96_report

# After formal test evaluation, generate the test report separately.
python report_unified_slim.py --results-dir results/unified_slim_test96 \
  --out-dir results/unified_slim_test96_report
```

The formal test report can attach separate matching profiles with
`--profile-dir`. A report never substitutes historical validation values for
new test results. Existing report outputs require explicit `--overwrite`.

For regression tests using original files outside this worktree:

```bash
MAGE_REGRESSION_RESULTS_DIR=/home/user/Workspace/KIMJIWON/graph/MAGE_MS/results/eval_proposal96 \
MAGE_GROUPED_REGRESSION_RESULTS_DIR=/home/user/Workspace/KIMJIWON/graph/MAGE_MS/results/eval_followup96 \
MAGE_REGRESSION_DATA_DIR="$DATA_DIR" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
python -m unittest -v test_proposal test_proposal_prep test_eval_tools \
  test_complexity test_grouped_ablation test_unified_slim test_train
```

## Interpretation limits and next work

Two-seed validation results are exploratory. A five-seed test comparison,
15/30/60-minute test metrics, and parameter/latency/memory tradeoffs are required
before claiming an accuracy or efficiency improvement. Comparing A and B alone
does not isolate fusion position. Independent per-scale Slim parameters,
hyperparameter retuning, and changes to normalization/data are outside scope.
