# M3 execution-DAG step 2: S=1 MSA block scan

Date: 5 September 2026.

## Change

The MiniMax-M3 index-score scan now splits `S=1` work over independent key
blocks. A worker visits keys in increasing order inside its block and writes
only that block's scores for all index heads. The f64 dot routine, max pooling,
`kv_start`, local-block promotion, greedy top-k, lowest-index tie-breaking,
and selected-block order are unchanged. Top-k selection remains serial.

`S>1` retains the serial scanner. The default threshold is 32 blocks (4096
tokens at M3's 128-token block size), selected from the benchmark below.
`COLI_MSA_SCAN_MIN_BLOCKS=<n>` overrides it; `1` is for forcing the test path.
`PROF=1` reports parallel-call, block-job and OpenMP worker-team-slot counts.

## Correctness and reachability

[`test_msa_index_scan.c`](../../c/tests/test_msa_index_scan.c) compares the
production one- and four-worker routes to an independent scalar f64 reference,
then independently reselects blocks. It covers 1, 127, 128, 129, 2047, 2048,
2049 and 8193 tokens; ties; local blocks; truncated first blocks via
`kv_start`; and empty historical blocks.

```text
test_msa_index_scan: ok (8 lengths, kv_start, ties, local blocks; forced parallel)
test_pipe_block: ok
m3-tiny-check: OK (prefill 24/24, decode 20/20)
```

With `COLI_MSA_SCAN_MIN_BLOCKS=1`, the 20-token M3 replay reported `20/20`
parallel S=1 calls, 75 block jobs and 200 worker-team slots on 10 threads.
At the normal threshold it reported `0/20`, proving the short-context guard.

## Scan microbenchmark

[`bench_msa_index_scan.c`](../../c/tests/bench_msa_index_scan.c) times the
production scanner alone at M3-like 8 heads, dimension 128, block size 128.
Separate processes gave stable one- and ten-thread teams. On the Step-0
i5-12600K host, lower is better:

| Tokens | Blocks | 1 thread ms/scan | 10 threads ms/scan | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 1 | 0.012 | 0.067 | 0.18x |
| 1,024 | 8 | 0.090 | 0.093 | 0.97x |
| 2,048 | 16 | 0.183 | 0.162 | 1.13x |
| 4,096 | 32 | 0.360 | 0.268 | 1.34x |
| 8,192 | 64 | 0.719 | 0.464 | 1.55x |
| 65,536 | 512 | 5.838 | 1.490 | 3.92x |

Thirty-two blocks avoids the measured short-context loss while admitting the
first clearly material 4K result. The 65K result is an isolated scanner
measurement, not model-wide token latency.

The short end-to-end control supports that guard: normal threshold delivered
984.30 tok/s (p50 1.0 ms); deliberately forcing one-block parallelism delivered
629.66 tok/s (p50 1.4 ms). No full-M3 long-context token-latency measurement
was possible on this host.

## Gate 2 status

| Gate | Status | Evidence / limitation |
|---|---|---|
| C | PASS, scanner and tiny-M3 scope | Scalar, one-thread and four-thread scores/selects agree at all required boundaries; tiny oracle and PIPE pass with the route forced. |
| M | PASS | The direct test forces four workers; runtime counters show 20 parallel S=1 scans and 75 block jobs. |
| P | PASS for scan plus short-token control | Scan timings span 128–65,536 tokens and the short token control is recorded. Full-M3 long-context latency remains open, so no model-wide speed claim is made. |

The next step may rely on a parallel MSA scan with unchanged selection
semantics. Retain the scalar/reference test when changing scan or threshold.
