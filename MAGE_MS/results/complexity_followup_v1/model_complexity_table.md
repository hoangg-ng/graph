# Model complexity and matched MAE

MAE is the mean ± sample standard deviation over completed seed runs. Complexity fields come from saved-checkpoint profiles; if only seed 0 was profiled, they represent seed 0. Counted FLOPs include only formulas registered by the stated PyTorch FlopCounterMode and are not guaranteed to cover every operator. Latency and GPU memory are hardware/runtime-specific.

Data/split digest validation passed. Checkpoint/profile code-digest mismatches: **0**.

| Dataset | Model | Test MAE (mean ± SD) | Params (M) | Counted GFLOPs | Graph apps (+/−) | Latency B=1 (ms) | Throughput B=64 (samples/s) | Peak GPU memory B=1/64 (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PEMS04 | grouped | 17.8589 ± 0.0211 | 3.056 | 1.6528 | 3/3 | 3.815 | 1746.5 | 36.1/1049.6 |
| PEMS04 | mage_features | 18.0183 ± 0.0217 | 2.765 | 0.9412 | not comparable (MAGE path) | 2.270 | 7714.1 | 26.1/347.6 |
| PEMS04 | no_spatial | 18.2481 ± 0.0354 | 1.103 | 2.2152 | 0/0 | 3.748 | 1462.5 | 28.8/1042.2 |
| PEMS04 | proposed_no_pdm | 18.0497 ± 0.0300 | 2.594 | 6.0576 | 45/12 | 8.180 | 1557.8 | 23.3/333.0 |
| PEMS08 | grouped | 13.6781 ± 0.0369 | 2.197 | 0.9152 | 3/3 | 3.860 | 3034.1 | 26.0/584.3 |
| PEMS08 | mage_features | 13.7602 ± 0.0146 | 1.906 | 0.5212 | not comparable (MAGE path) | 2.206 | 12367.7 | 19.7/209.9 |
| PEMS08 | no_spatial | 13.8168 ± 0.0313 | 1.085 | 1.2267 | 0/0 | 3.717 | 2591.9 | 21.8/580.1 |
| PEMS08 | proposed_no_pdm | 13.8876 ± 0.0320 | 1.735 | 3.3543 | 45/12 | 7.994 | 2557.5 | 18.1/188.6 |

## Paired model comparisons

### PEMS04

No complete pairs available.

### PEMS08

No complete pairs available.

## Measurement environment

Each completed profile records its GPU, PyTorch/CUDA versions, FP32/TF32 setting, batches, warm-up/repeat counts, FLOP method, and checkpoint/profile code-digest match status. See `profile_environment` in `model_complexity_summary.json` for run-level values.

