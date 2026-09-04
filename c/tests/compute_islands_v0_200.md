# Compute Islands v0 timing

Input: `c/tests/compute_islands_v0_200.csv`; 8000 records.

## Aggregate

- Mean makespan: **7.879 ms/layer**
- Median makespan: **8.354 ms/layer**
- p95 makespan: **11.382 ms/layer**
- Mean dispatch delay: **0.083 ms**
- Mean GPU lane: **0.199 ms**
- Mean CPU lane: **3.132 ms**
- Mean exposed merge: **0.381 ms**
- Row classifications: `{"CPU-critical": 5540, "GPU-critical": 2400, "merge/synchronization-critical": 60}`

## Per-layer distribution (mean over tokens)

| Layer | CPU ms | GPU ms | Dispatch ms | Imbalance ms | Merge ms | Makespan ms | GPU slack ms | Class |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.000 | 0.649 | 0.342 | 0.000 | 0.056 | 6.730 | 0.056 | GPU-critical |
| 1 | 0.000 | 0.624 | 0.295 | 0.000 | 0.051 | 5.337 | 0.051 | GPU-critical |
| 2 | 0.000 | 0.629 | 0.281 | 0.000 | 0.051 | 5.275 | 0.051 | GPU-critical |
| 3 | 0.000 | 0.611 | 0.272 | 0.000 | 0.050 | 3.789 | 0.050 | GPU-critical |
| 4 | 0.000 | 0.617 | 0.272 | 0.000 | 0.050 | 5.218 | 0.050 | GPU-critical |
| 5 | 0.000 | 0.603 | 0.263 | 0.000 | 0.050 | 5.239 | 0.050 | GPU-critical |
| 6 | 0.000 | 0.606 | 0.273 | 0.000 | 0.049 | 5.203 | 0.049 | GPU-critical |
| 7 | 0.000 | 0.595 | 0.255 | 0.000 | 0.050 | 3.740 | 0.050 | GPU-critical |
| 8 | 0.000 | 0.612 | 0.266 | 0.000 | 0.053 | 5.273 | 0.053 | GPU-critical |
| 9 | 0.000 | 0.616 | 0.267 | 0.000 | 0.050 | 5.335 | 0.050 | GPU-critical |
| 10 | 0.000 | 0.618 | 0.269 | 0.000 | 0.050 | 5.265 | 0.050 | GPU-critical |
| 11 | 0.569 | 1.178 | 0.264 | 0.356 | 0.052 | 4.433 | 0.052 | GPU-critical |
| 12 | 4.621 | 0.000 | 0.000 | 0.000 | 0.516 | 9.532 | 0.000 | CPU-critical |
| 13 | 4.704 | 0.000 | 0.000 | 0.000 | 0.521 | 9.737 | 0.000 | CPU-critical |
| 14 | 4.523 | 0.000 | 0.000 | 0.000 | 0.526 | 9.581 | 0.000 | CPU-critical |
| 15 | 4.652 | 0.000 | 0.000 | 0.000 | 0.513 | 8.220 | 0.000 | CPU-critical |
| 16 | 4.512 | 0.000 | 0.000 | 0.000 | 0.526 | 9.481 | 0.000 | CPU-critical |
| 17 | 4.581 | 0.000 | 0.000 | 0.000 | 0.534 | 9.626 | 0.000 | CPU-critical |
| 18 | 4.414 | 0.000 | 0.000 | 0.000 | 0.525 | 9.412 | 0.000 | CPU-critical |
| 19 | 4.569 | 0.000 | 0.000 | 0.000 | 0.549 | 8.208 | 0.000 | CPU-critical |
| 20 | 4.319 | 0.000 | 0.000 | 0.000 | 0.519 | 9.366 | 0.000 | CPU-critical |
| 21 | 4.398 | 0.000 | 0.000 | 0.000 | 0.537 | 9.429 | 0.000 | CPU-critical |
| 22 | 4.618 | 0.000 | 0.000 | 0.000 | 0.521 | 9.608 | 0.000 | CPU-critical |
| 23 | 4.697 | 0.000 | 0.000 | 0.000 | 0.525 | 8.259 | 0.000 | CPU-critical |
| 24 | 4.573 | 0.000 | 0.000 | 0.000 | 0.522 | 9.628 | 0.000 | CPU-critical |
| 25 | 4.641 | 0.000 | 0.000 | 0.000 | 0.538 | 9.590 | 0.000 | CPU-critical |
| 26 | 4.248 | 0.000 | 0.000 | 0.000 | 0.517 | 9.209 | 0.000 | CPU-critical |
| 27 | 4.392 | 0.000 | 0.000 | 0.000 | 0.507 | 7.910 | 0.000 | CPU-critical |
| 28 | 4.338 | 0.000 | 0.000 | 0.000 | 0.513 | 9.291 | 0.000 | CPU-critical |
| 29 | 4.313 | 0.000 | 0.000 | 0.000 | 0.525 | 9.250 | 0.000 | CPU-critical |
| 30 | 4.082 | 0.000 | 0.000 | 0.000 | 0.525 | 9.022 | 0.000 | CPU-critical |
| 31 | 4.169 | 0.000 | 0.000 | 0.000 | 0.525 | 7.716 | 0.000 | CPU-critical |
| 32 | 4.318 | 0.000 | 0.000 | 0.000 | 0.517 | 9.326 | 0.000 | CPU-critical |
| 33 | 4.360 | 0.000 | 0.000 | 0.000 | 0.533 | 9.305 | 0.000 | CPU-critical |
| 34 | 4.517 | 0.000 | 0.000 | 0.000 | 0.516 | 9.525 | 0.000 | CPU-critical |
| 35 | 4.331 | 0.000 | 0.000 | 0.000 | 0.515 | 7.941 | 0.000 | CPU-critical |
| 36 | 4.353 | 0.000 | 0.000 | 0.000 | 0.514 | 9.220 | 0.000 | CPU-critical |
| 37 | 4.405 | 0.000 | 0.000 | 0.000 | 0.513 | 9.442 | 0.000 | CPU-critical |
| 38 | 4.415 | 0.000 | 0.000 | 0.000 | 0.505 | 9.291 | 0.000 | CPU-critical |
| 39 | 4.656 | 0.000 | 0.000 | 0.000 | 0.521 | 8.179 | 0.000 | CPU-critical |

## Largest makespans

- Layer 13: 9.737 ms, CPU-critical
- Layer 24: 9.628 ms, CPU-critical
- Layer 17: 9.626 ms, CPU-critical
- Layer 22: 9.608 ms, CPU-critical
- Layer 25: 9.590 ms, CPU-critical

The GPU completion timestamp is host-clocked immediately after the resident synchronization; this is suitable for the lightweight v0 comparison, but not equivalent to a hardware-synchronized CUPTI timeline.
