# Compute Islands v0 timing

Input: `c/tests/compute_islands_v0_balanced_200.csv`; 8000 records.

## Aggregate

- Mean makespan: **10.845 ms/layer**
- Median makespan: **10.892 ms/layer**
- p95 makespan: **13.916 ms/layer**
- Mean dispatch delay: **0.172 ms**
- Mean pre-assignment work: **5.139 ms**
- Mean GPU lane: **3.429 ms**
- Mean CPU lane: **4.836 ms**
- Mean exposed merge: **0.222 ms**
- Mean post-merge tail: **0.043 ms**
- Row classifications: `{"CPU-critical": 2637, "GPU-critical": 4796, "balanced": 535, "merge/synchronization-critical": 32}`

## Per-layer distribution (mean over tokens)

| Layer | Pre ms | CPU ms | GPU ms | Dispatch ms | Imbalance ms | Merge ms | Makespan ms | GPU slack ms | Class |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 6.922 | 4.953 | 4.479 | 0.251 | 0.563 | 0.156 | 12.862 | 0.049 | GPU-critical |
| 1 | 5.282 | 5.201 | 2.685 | 0.134 | 0.320 | 0.285 | 11.274 | 0.030 | CPU-critical |
| 2 | 5.366 | 4.589 | 3.602 | 0.193 | 0.503 | 0.180 | 10.857 | 0.041 | GPU-critical |
| 3 | 3.844 | 5.095 | 4.188 | 0.191 | 0.516 | 0.188 | 9.857 | 0.044 | GPU-critical |
| 4 | 5.424 | 4.784 | 3.420 | 0.177 | 0.445 | 0.201 | 11.061 | 0.040 | GPU-critical |
| 5 | 5.325 | 4.754 | 4.067 | 0.200 | 0.520 | 0.162 | 10.982 | 0.046 | GPU-critical |
| 6 | 5.422 | 5.278 | 2.976 | 0.138 | 0.362 | 0.269 | 11.515 | 0.031 | GPU-critical |
| 7 | 3.672 | 4.723 | 3.218 | 0.164 | 0.411 | 0.228 | 9.232 | 0.037 | GPU-critical |
| 8 | 5.471 | 4.149 | 4.580 | 0.255 | 0.660 | 0.077 | 10.616 | 0.055 | GPU-critical |
| 9 | 5.655 | 4.603 | 4.344 | 0.215 | 0.566 | 0.129 | 11.183 | 0.050 | GPU-critical |
| 10 | 5.697 | 5.109 | 2.789 | 0.131 | 0.354 | 0.302 | 11.641 | 0.030 | CPU-critical |
| 11 | 3.912 | 5.275 | 3.053 | 0.141 | 0.383 | 0.287 | 10.040 | 0.032 | GPU-critical |
| 12 | 5.599 | 4.686 | 3.785 | 0.203 | 0.497 | 0.178 | 11.187 | 0.045 | GPU-critical |
| 13 | 5.712 | 4.546 | 3.398 | 0.188 | 0.469 | 0.199 | 11.143 | 0.041 | GPU-critical |
| 14 | 5.664 | 4.463 | 3.761 | 0.205 | 0.527 | 0.167 | 11.047 | 0.046 | GPU-critical |
| 15 | 3.862 | 5.058 | 2.396 | 0.115 | 0.313 | 0.347 | 9.745 | 0.027 | CPU-critical |
| 16 | 5.431 | 4.713 | 3.305 | 0.168 | 0.428 | 0.233 | 11.005 | 0.039 | GPU-critical |
| 17 | 5.496 | 4.810 | 3.259 | 0.165 | 0.424 | 0.224 | 11.153 | 0.037 | GPU-critical |
| 18 | 5.425 | 8.476 | 7.197 | 0.177 | 0.453 | 0.214 | 14.775 | 0.039 | GPU-critical |
| 19 | 3.836 | 4.688 | 3.982 | 0.194 | 0.529 | 0.168 | 9.435 | 0.046 | GPU-critical |
| 20 | 5.507 | 4.238 | 4.021 | 0.222 | 0.566 | 0.126 | 10.673 | 0.048 | GPU-critical |
| 21 | 5.311 | 3.739 | 4.045 | 0.240 | 0.619 | 0.088 | 10.002 | 0.054 | GPU-critical |
| 22 | 5.451 | 5.094 | 2.802 | 0.135 | 0.364 | 0.285 | 11.371 | 0.032 | GPU-critical |
| 23 | 3.829 | 4.888 | 3.045 | 0.151 | 0.400 | 0.261 | 9.566 | 0.035 | GPU-critical |
| 24 | 5.469 | 4.673 | 3.605 | 0.179 | 0.471 | 0.203 | 11.023 | 0.040 | GPU-critical |
| 25 | 5.555 | 4.482 | 4.010 | 0.206 | 0.541 | 0.139 | 10.940 | 0.047 | GPU-critical |
| 26 | 5.747 | 3.863 | 4.099 | 0.236 | 0.628 | 0.095 | 10.576 | 0.055 | GPU-critical |
| 27 | 3.844 | 5.187 | 1.992 | 0.094 | 0.264 | 0.369 | 9.819 | 0.022 | CPU-critical |
| 28 | 5.665 | 4.790 | 3.570 | 0.188 | 0.454 | 0.210 | 11.337 | 0.040 | GPU-critical |
| 29 | 5.610 | 4.681 | 2.824 | 0.163 | 0.386 | 0.261 | 11.143 | 0.034 | GPU-critical |
| 30 | 5.671 | 4.666 | 2.439 | 0.129 | 0.318 | 0.306 | 11.140 | 0.029 | CPU-critical |
| 31 | 3.926 | 4.860 | 2.488 | 0.128 | 0.327 | 0.333 | 9.622 | 0.029 | CPU-critical |
| 32 | 5.644 | 4.842 | 2.763 | 0.137 | 0.362 | 0.278 | 11.310 | 0.031 | CPU-critical |
| 33 | 5.605 | 4.718 | 3.029 | 0.162 | 0.400 | 0.241 | 11.164 | 0.037 | GPU-critical |
| 34 | 5.499 | 4.829 | 3.653 | 0.180 | 0.460 | 0.197 | 11.194 | 0.041 | GPU-critical |
| 35 | 3.822 | 4.050 | 4.103 | 0.230 | 0.593 | 0.107 | 8.812 | 0.052 | GPU-critical |
| 36 | 5.446 | 5.148 | 2.075 | 0.098 | 0.250 | 0.355 | 11.357 | 0.023 | CPU-critical |
| 37 | 5.692 | 4.975 | 1.765 | 0.092 | 0.229 | 0.354 | 11.403 | 0.020 | CPU-critical |
| 38 | 5.458 | 4.595 | 3.011 | 0.159 | 0.407 | 0.244 | 10.900 | 0.036 | GPU-critical |
| 39 | 3.788 | 5.180 | 3.339 | 0.160 | 0.410 | 0.248 | 9.822 | 0.037 | GPU-critical |

## Largest makespans

- Layer 18: 14.775 ms, GPU-critical
- Layer 0: 12.862 ms, GPU-critical
- Layer 10: 11.641 ms, CPU-critical
- Layer 6: 11.515 ms, GPU-critical
- Layer 37: 11.403 ms, CPU-critical

The GPU completion timestamp is host-clocked immediately after the resident synchronization; this is suitable for the lightweight v0 comparison, but not equivalent to a hardware-synchronized CUPTI timeline.
