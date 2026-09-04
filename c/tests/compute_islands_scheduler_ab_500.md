# Compute Islands scheduler placement A/B

Matched production-style A/B on one GTX 1080.  Both runs used the same
`coli` binary, model, prompt, 32-token warmup, 500-token request, 3,642-slot
QTH1 snapshot, DP4A resident path, `QTIER_LFRU=0`, and read-only heat files.
`COLI_TIMERS`, timing CSV, timing dump, and CUPTI were explicitly disabled.

The cleanest complete resource pair is `compute_islands_scheduler_ab_500_v3.json`:

| Run | tok/s | ms/token | GPU util | Power | VRAM | Engine CPU | Engine RSS | Output SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| current | 2.2317 | 448.081 | 6.26% | 44.63 W | 7,683 MiB | 144.25% | 21.11 GiB | `2d8185c5...19bbb7cb` |
| makespan-balanced | 2.2128 | 451.918 | 6.19% | 44.70 W | 7,682 MiB | 143.76% | 21.41 GiB | `37014d33...d76638a1` |

The candidate is **0.85% slower**, well inside observed runtime variance.  GPU
utilization, power, VRAM, and CPU load are effectively unchanged.  A second
complete pair (`v2`) measured 2.1835 versus 2.2199 tok/s, showing the same
high variance rather than a reproducible candidate win.  The first pair is
not used for resource conclusions because its sampler followed the Python
parent process instead of `qwen36`.

## Correctness gate

The 64-token deterministic smoke produced stable but different output text:

```text
current          1669f3c1...f5f35674
makespan-balanced 244ecbed...28ef69a9
```

The first textual difference occurs at character 134.  Therefore exact output
identity is **FAIL for this heat snapshot**.  This is likely the existing
CPU/GPU numerical-path sensitivity exposed by a different expert assignment,
not a simulator error, but it must be resolved or accepted under an explicit
tolerance before this placement can be used in production.

## Decision

Structural/runtime A/B: **PASS** — the matched experiment ran with the
requested diagnostics disabled and collected GPU, VRAM, power, PCIe, CPU, RSS,
private memory, system RAM, temperature, and memory-utilization samples.

Performance: **FAIL / not demonstrated** — no reproducible improvement.

Correctness: **FAIL for exact identity** — candidate output diverged from the
current placement.

Next intervention: do not install this heat policy. First isolate the
placement-dependent numerical divergence with a short per-layer output parity
trace; only a candidate that preserves the accepted output criterion should
return to long performance A/B.
