# Grouped / Slim fusion-order comparison

Mode: validation only (best_val_mae). Completed 16/16 runs. All pilot findings are exploratory.

MAE change is candidate minus Grouped; negative is lower. Standard deviations are sample SD. RMSE/MAPE are means of per-run metrics. Missing metrics remain n/a. In validation mode, RMSE/MAPE and horizon errors come only from separate best-checkpoint validation metrics; these are not test results.

| Dataset | Model | Seeds | MAE mean ± SD | RMSE | MAPE | Params | Counted GFLOPs | Latency B=1 ms | Peak B=64 MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PEMS04 | grouped | 2/2 | 17.9553 ± 0.0088 | 29.9517 | 12.0446 | 3056156 | 1.6528 | 3.861 | 1050.9 |
| PEMS04 | grouped_slim | 2/2 | 17.9856 ± 0.0065 | 30.0381 | 12.0372 | 1163711 | 1.1662 | 2.493 | 1043.7 |
| PEMS04 | slim_then_grouped | 2/2 | 18.1710 ± 0.0247 | 30.3669 | 12.1252 | 1163711 | 2.2526 | 3.911 | 1043.7 |
| PEMS04 | grouped_uniform | 2/2 | 18.0745 ± 0.0804 | 30.0279 | 12.1544 | 1163708 | 1.1662 | 2.490 | 1043.7 |
| PEMS08 | grouped | 2/2 | 13.9073 ± 0.0257 | 24.1263 | 11.0331 | 2196892 | 0.9152 | 3.780 | 584.1 |
| PEMS08 | grouped_slim | 2/2 | 13.9799 ± 0.0442 | 24.2874 | 11.5479 | 1146175 | 0.6458 | 2.455 | 580.1 |
| PEMS08 | slim_then_grouped | 2/2 | 14.0675 ± 0.0275 | 24.2403 | 11.0589 | 1146175 | 1.2474 | 3.879 | 580.1 |
| PEMS08 | grouped_uniform | 2/2 | 14.0398 ± 0.0166 | 24.0822 | 11.0468 | 1146172 | 0.6458 | 2.481 | 580.1 |

## Paired and resource comparisons

| Dataset | Model | ΔMAE | ΔMAE % | Wins / pairs | Param reduction % | Δlatency % | Δmemory % | MAE 15/30/60m |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PEMS04 | grouped | n/a | n/a | n/a | -0.00 | 0.00 | 0.00 | 17.4444/17.9903/18.7753 |
| PEMS04 | grouped_slim | 0.0303 | 0.17 | 0/2 | 61.92 | -35.42 | -0.69 | 17.4624/18.0253/18.8223 |
| PEMS04 | slim_then_grouped | 0.2157 | 1.20 | 0/2 | 61.92 | 1.29 | -0.69 | 17.5774/18.2001/19.1037 |
| PEMS04 | grouped_uniform | 0.1192 | 0.66 | 0/2 | 61.92 | -35.51 | -0.69 | 17.5530/18.1050/18.9406 |
| PEMS08 | grouped | n/a | n/a | n/a | -0.00 | 0.00 | 0.00 | 13.1116/13.9416/15.0578 |
| PEMS08 | grouped_slim | 0.0726 | 0.52 | 0/2 | 47.83 | -35.05 | -0.69 | 13.1553/14.0185/15.1603 |
| PEMS08 | slim_then_grouped | 0.1602 | 1.15 | 0/2 | 47.83 | 2.64 | -0.69 | 13.2180/14.1137/15.2866 |
| PEMS08 | grouped_uniform | 0.1326 | 0.95 | 0/2 | 47.83 | -34.37 | -0.69 | 13.2127/14.0395/15.3007 |

## A versus B

- PEMS04: paired A−B MAE -0.1854 ± 0.0312; A wins 2/2.
- PEMS08: paired A−B MAE -0.0876 ± 0.0167; A wins 2/2.

Cross-dataset consistency: A has lower available paired mean MAE on both datasets (descriptive).

## Interpretation and measurement limits

A fuses before spatial processing; B reuses the same three Slim blocks across four streams. A/B have equal parameters, but B executes Slim/FFN 12 times versus A's 3. Prompt position and nonlinear processing also differ. Results cannot identify a causal effect of fusion position alone.

Cost values come only from matching saved-checkpoint profiles, not random weights. The CSV records profile_seed_count. FLOPs count registered formulas only; latency/memory depend on the device/runtime. Peak B=64 memory is inference allocation, not training peak. Cross-dataset directions are descriptive and do not establish statistical significance or a general architectural effect.
