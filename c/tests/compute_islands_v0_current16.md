# Compute Islands v0 — layer classification

Input: `c/tests/overlap_diag_dp4a_16.csv.layers.csv`; layers: 40.
This is aggregate lightweight instrumentation. Host and CUDA clocks are not treated as synchronized.

## Aggregate

- Mean layer envelope: **23.618 ms**
- Mean CPU window: **18.798 ms**
- Mean GPU event work: **6.392 ms**
- Mean residual boundary: **4.820 ms**
- Mean CPU/GPU imbalance: **12.406 ms**
- Class counts: `{"merge/synchronization-critical": 40}`

## Largest layer envelopes

| Layer | CPU ms | GPU ms | Envelope ms | Residual ms | Hit rate | Class |
|---:|---:|---:|---:|---:|---:|---|
| 38 | 27.686 | 5.634 | 32.712 | 5.026 | 55.0% | merge/synchronization-critical |
| 36 | 27.212 | 5.281 | 32.306 | 5.094 | 54.2% | merge/synchronization-critical |
| 37 | 26.059 | 5.417 | 30.980 | 4.920 | 55.8% | merge/synchronization-critical |
| 39 | 25.504 | 5.514 | 30.265 | 4.761 | 55.0% | merge/synchronization-critical |
| 32 | 24.903 | 5.994 | 29.614 | 4.711 | 60.0% | merge/synchronization-critical |
| 35 | 24.130 | 5.555 | 28.634 | 4.504 | 59.2% | merge/synchronization-critical |
| 24 | 23.106 | 5.873 | 27.722 | 4.616 | 63.3% | merge/synchronization-critical |
| 1 | 21.225 | 6.433 | 26.948 | 5.724 | 70.8% | merge/synchronization-critical |
| 17 | 22.027 | 5.650 | 26.944 | 4.918 | 60.8% | merge/synchronization-critical |
| 31 | 22.169 | 5.925 | 26.942 | 4.773 | 59.2% | merge/synchronization-critical |

## Interpretation

This report is a makespan diagnostic, not a scheduler. It supports the island model only provisionally: the observed layer envelope is compared with the maximum of the CPU and GPU local windows, while the residual is retained as exposed boundary/measurement territory.
