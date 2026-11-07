# Model complexity and matched MAE

MAE is the mean ± sample standard deviation over completed seed runs. Complexity fields come from saved-checkpoint profiles; if only seed 0 was profiled, they represent seed 0. Counted FLOPs include only formulas registered by the stated PyTorch FlopCounterMode and are not guaranteed to cover every operator. Latency and GPU memory are hardware/runtime-specific.

Data/split digest validation passed. Checkpoint/profile code-digest mismatches: **12**.

Profiles below execute a different source fingerprint from checkpoint training. Output regression evidence is needed to interpret equivalence; a source mismatch does not itself establish a behavior change.

- PEMS04/independent/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS04/mage/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS04/proposed/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS04/reversed/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS04/shared/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS04/uniform/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS08/independent/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS08/mage/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS08/proposed/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS08/reversed/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS08/shared/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`
- PEMS08/uniform/s0: checkpoint `9c47e939b1875807c82f8901b0d3ae2e4541893cdeaa4d922b962a23ca742cc3` → profile `4d4fd58b70ee7116ebe107603bb11e80ee85b45113ac68128decd27f15b2bca0`

| Dataset | Model | Test MAE (mean ± SD) | Params (M) | Counted GFLOPs | Graph apps (+/−) | Latency B=1 (ms) | Throughput B=64 (samples/s) | Peak GPU memory B=1/64 (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PEMS04 | independent | 18.1135 ± 0.0288 | 8.654 | 4.1505 | 12/12 | 7.117 | 1153.7 | 57.8/1070.5 |
| PEMS04 | mage | 18.0921 ± 0.0217 | 2.703 | 0.9035 | not comparable (MAGE path) | 2.293 | 7939.5 | 25.3/310.1 |
| PEMS04 | proposed | 18.1301 ± 0.0704 | 2.995 | 6.8063 | 45/12 | 9.620 | 915.6 | 35.9/1049.4 |
| PEMS04 | reversed | 18.1357 ± 0.0499 | 2.995 | 6.8063 | 45/12 | 9.655 | 914.0 | 35.9/1049.4 |
| PEMS04 | shared | 18.1479 ± 0.0559 | 2.995 | 4.1505 | 12/12 | 7.107 | 1157.7 | 35.9/1049.4 |
| PEMS04 | uniform | 17.9985 ± 0.0386 | 1.103 | 2.2152 | 0/0 | 3.811 | 1469.5 | 28.8/1042.2 |
| PEMS08 | independent | 13.9131 ± 0.0122 | 5.269 | 2.2983 | 12/12 | 7.029 | 1989.6 | 37.7/596.1 |
| PEMS08 | mage | 13.8126 ± 0.0199 | 1.844 | 0.5003 | not comparable (MAGE path) | 2.187 | 12623.7 | 19.2/190.4 |
| PEMS08 | proposed | 13.9115 ± 0.0309 | 2.136 | 3.7689 | 45/12 | 9.413 | 1556.4 | 25.8/584.1 |
| PEMS08 | reversed | 13.9294 ± 0.0306 | 2.136 | 3.7689 | 45/12 | 9.469 | 1554.9 | 25.8/584.1 |
| PEMS08 | shared | 13.9298 ± 0.0242 | 2.136 | 2.2983 | 12/12 | 7.003 | 1989.3 | 25.8/584.1 |
| PEMS08 | uniform | 13.8273 ± 0.0680 | 1.085 | 1.2267 | 0/0 | 3.829 | 2596.3 | 21.8/580.1 |

## Paired model comparisons

### PEMS04

- **Proposed vs Shared:** ΔMAE=-0.0177, Δcounted GFLOPs=2.6558, Δparameters=0.0000 (first minus second; negative is lower).
- **Proposed vs Reversed:** ΔMAE=-0.0056, Δcounted GFLOPs=0.0000, Δparameters=0.0000 (first minus second; negative is lower).
- **Proposed vs MAGE:** ΔMAE=0.0381, Δcounted GFLOPs=5.9028, Δparameters=291698.0000 (first minus second; negative is lower).
- **Independent vs Shared:** ΔMAE=-0.0344, Δcounted GFLOPs=0.0000, Δparameters=5658768.0000 (first minus second; negative is lower).
- **Uniform vs Proposed:** ΔMAE=-0.1317, Δcounted GFLOPs=-4.5910, Δparameters=-1892448.0000 (first minus second; negative is lower).

### PEMS08

- **Proposed vs Shared:** ΔMAE=-0.0183, Δcounted GFLOPs=1.4706, Δparameters=0.0000 (first minus second; negative is lower).
- **Proposed vs Reversed:** ΔMAE=-0.0179, Δcounted GFLOPs=0.0000, Δparameters=0.0000 (first minus second; negative is lower).
- **Proposed vs MAGE:** ΔMAE=0.0988, Δcounted GFLOPs=3.2686, Δparameters=291698.0000 (first minus second; negative is lower).
- **Independent vs Shared:** ΔMAE=-0.0167, Δcounted GFLOPs=0.0000, Δparameters=3133584.0000 (first minus second; negative is lower).
- **Uniform vs Proposed:** ΔMAE=-0.0842, Δcounted GFLOPs=-2.5423, Δparameters=-1050720.0000 (first minus second; negative is lower).

## Measurement environment

Each completed profile records its GPU, PyTorch/CUDA versions, FP32/TF32 setting, batches, warm-up/repeat counts, FLOP method, and checkpoint/profile code-digest match status. See `profile_environment` in `model_complexity_summary.json` for run-level values.

