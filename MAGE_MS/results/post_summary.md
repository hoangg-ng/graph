
## PEMS08
regime counts (sample x node): {'free flow': 573085, 'onset': 10510, 'recovery': 7532, 'congested': 4114, 'low volatility': 162993, 'high volatility': 139240}

| model | free flow | onset | recovery | congested | low volatility | high volatility |
|---|---|---|---|---|---|---|
| linear | 13.12 | 31.81 | 29.35 | 28.12 | 4.98 | 24.56 |
| mslin | 13.14 (+0.02, 2/5) | 31.71 (-0.10, 3/5) | 29.33 (-0.02, 3/5) | 28.00 (-0.12, 4/5) | 5.05 (+0.07, 0/5) | 24.53 (-0.02, 3/5) |
| m3 | 13.15 (+0.03, 2/5) | 31.81 (-0.01, 2/5) | 29.67 (+0.32, 0/5) | 27.99 (-0.13, 3/5) | 5.05 (+0.07, 1/5) | 24.62 (+0.06, 2/5) |
| m4 | 13.15 (+0.03, 1/5) | 31.78 (-0.04, 3/5) | 29.44 (+0.09, 1/5) | 28.01 (-0.11, 4/5) | 5.03 (+0.05, 1/5) | 24.57 (+0.01, 2/5) |
| m5 | 13.23 (+0.10, 0/5) | 31.87 (+0.05, 2/5) | 29.61 (+0.26, 0/5) | 28.37 (+0.25, 1/5) | 5.12 (+0.14, 0/5) | 24.66 (+0.11, 0/5) |

routing (mean over seeds), per block: usage entropy / mean per-node entropy / max expert share
linear: b1 0.993 / 0.915 / 0.089 ; b2 0.993 / 0.860 / 0.080 ; b3 0.981 / 0.974 / 0.114
m5: b1 0.993 / 0.922 / 0.094 ; b2 0.980 / 0.878 / 0.090 ; b3 0.987 / 0.978 / 0.094
JSD(global usage) linear vs m5 same seed, per block: [0.011, 0.0225, 0.0214]; linear vs linear other seed: [0.0087, 0.0114, 0.0271]
m4 gates (scale x1, x2, x4) mean [0.151, 0.416, 0.432], sd across nodes [0.0, 0.0, 0.0], normalised entropy mean 0.912 min 0.912
m5 gates (scale x1, x2, x4) mean [0.129, 0.307, 0.564], sd across nodes [0.093, 0.186, 0.191], normalised entropy mean 0.727 min 0.181

## PEMS04
regime counts (sample x node): {'free flow': 964716, 'onset': 23813, 'recovery': 17412, 'congested': 9060, 'low volatility': 254007, 'high volatility': 258281}

| model | free flow | onset | recovery | congested | low volatility | high volatility |
|---|---|---|---|---|---|---|
| linear | 17.06 | 34.64 | 34.01 | 35.96 | 5.71 | 33.51 |
| mslin | 17.11 (+0.06, 1/5) | 34.63 (-0.01, 2/5) | 33.91 (-0.10, 4/5) | 35.91 (-0.05, 3/5) | 5.76 (+0.05, 0/5) | 33.52 (+0.01, 3/5) |
| m3 | 17.02 (-0.04, 5/5) | 34.64 (-0.00, 2/5) | 34.03 (+0.01, 3/5) | 35.96 (+0.00, 3/5) | 5.70 (-0.01, 2/5) | 33.49 (-0.02, 3/5) |
| m4 | 17.01 (-0.04, 4/5) | 34.63 (-0.02, 3/5) | 34.04 (+0.03, 2/5) | 35.92 (-0.04, 3/5) | 5.72 (+0.01, 3/5) | 33.48 (-0.03, 3/5) |
| m5 | 17.05 (-0.01, 3/5) | 34.79 (+0.14, 0/5) | 34.27 (+0.26, 0/5) | 36.30 (+0.34, 0/5) | 5.68 (-0.03, 4/5) | 33.54 (+0.04, 2/5) |

routing (mean over seeds), per block: usage entropy / mean per-node entropy / max expert share
linear: b1 0.973 / 0.923 / 0.100 ; b2 0.987 / 0.894 / 0.089 ; b3 0.979 / 0.975 / 0.106
m5: b1 0.974 / 0.918 / 0.119 ; b2 0.989 / 0.879 / 0.085 ; b3 0.980 / 0.966 / 0.107
JSD(global usage) linear vs m5 same seed, per block: [0.0334, 0.0198, 0.0334]; linear vs linear other seed: [0.0398, 0.0174, 0.0343]
m4 gates (scale x1, x2, x4) mean [0.128, 0.348, 0.524], sd across nodes [0.0, 0.0, 0.0], normalised entropy mean 0.875 min 0.875
m5 gates (scale x1, x2, x4) mean [0.17, 0.332, 0.498], sd across nodes [0.094, 0.168, 0.185], normalised entropy mean 0.821 min 0.247