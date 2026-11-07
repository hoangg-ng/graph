# Matched MAGE versus multiscale spatial mixing evaluation

All methods use identical 96 -> 12 windows, chronological data split, and five seeds (0..4).
An independent MAGE reimplementation with original backbone and linear temporal encoder is the baseline;
it is not the original paper's 12 -> 12 benchmark result.

## Results for each variant (PEMS04, 96 -> 12, masked test MAE)

| Variant | Spatial mixing schedule | Runs | Test MAE (mean +/- SD) |
|---|---|---:|---:|
| Baseline: MAGE | original spatial mixer | 5/5 | 18.092 +/- 0.022 |
| Shared | (1, 1, 1, 1) | 5/5 | 18.148 +/- 0.056 |
| Proposed | (1, 2, 4, 8) | 5/5 | 18.130 +/- 0.070 |
| Reversed | (8, 4, 2, 1) | 5/5 | 18.136 +/- 0.050 |
| Independent | one operator per scale | 5/5 | 18.114 +/- 0.029 |
| Uniform | global mean | 5/5 | 17.998 +/- 0.039 |

### Paired comparison with MAGE

| Comparison | Paired seeds | Candidate wins | Mean MAE improvement vs MAGE |
|---|---:|---:|---:|
| Shared vs MAGE | 5/5 | 0/5 | -0.31% |
| Proposed vs MAGE | 5/5 | 1/5 | -0.21% |
| Reversed vs MAGE | 5/5 | 0/5 | -0.24% |
| Independent vs MAGE | 5/5 | 1/5 | -0.12% |
| Uniform vs MAGE | 5/5 | 5/5 | +0.52% |

Missing runs: **0**. 
A positive percentage indicates lower MAE for the candidate. 
Rows with fewer than five seeds are preliminary, not final five-seed results.


## Results for each variant (PEMS08, 96 -> 12, masked test MAE)

| Variant | Spatial mixing schedule | Runs | Test MAE (mean +/- SD) |
|---|---|---:|---:|
| Baseline: MAGE | original spatial mixer | 5/5 | 13.813 +/- 0.020 |
| Shared | (1, 1, 1, 1) | 5/5 | 13.930 +/- 0.024 |
| Proposed | (1, 2, 4, 8) | 5/5 | 13.911 +/- 0.031 |
| Reversed | (8, 4, 2, 1) | 5/5 | 13.929 +/- 0.031 |
| Independent | one operator per scale | 5/5 | 13.913 +/- 0.012 |
| Uniform | global mean | 5/5 | 13.827 +/- 0.068 |

### Paired comparison with MAGE

| Comparison | Paired seeds | Candidate wins | Mean MAE improvement vs MAGE |
|---|---:|---:|---:|
| Shared vs MAGE | 5/5 | 0/5 | -0.85% |
| Proposed vs MAGE | 5/5 | 0/5 | -0.72% |
| Reversed vs MAGE | 5/5 | 0/5 | -0.85% |
| Independent vs MAGE | 5/5 | 0/5 | -0.73% |
| Uniform vs MAGE | 5/5 | 3/5 | -0.11% |

Missing runs: **0**. 
A positive percentage indicates lower MAE for the candidate. 
Rows with fewer than five seeds are preliminary, not final five-seed results.

