# MiniMax-M3 Step 7 — three-drive calibration capture

## Scope

This capture extends the Windows storage evidence to all three available model
copies:

- drive `101`: C:, Samsung SSD 980 PRO, NVMe controller `201`;
- drive `102`: E:, Samsung MZAL4512, NVMe controller `202`;
- drive `103`: D:, Kingston XS1000, USB; controller/upstream identity is not
  inferred.

All three files are the same 4,526,467,992-byte M3 shard. The capture used
16 MiB reads, eight samples per cell, and inflight levels 1/2/4/8. It measured
each drive independently, all three drive pairs, and the three-drive group.
The raw capture is
[`m3-execution-step7-calibration-3drive-2026-09-07.json`](m3-execution-step7-calibration-3drive-2026-09-07.json);
the emitted planner text is
[`m3-execution-step7-calibration-3drive-2026-09-07.planner.txt`](m3-execution-step7-calibration-3drive-2026-09-07.planner.txt).

## Observed result

At inflight 8, the single-drive summaries were:

| Drive | p95 latency | p99 latency | max latency | p50 aggregate rate |
|---|---:|---:|---:|---:|
| C: / 101 | 42.1 ms | 43.0 ms | 43.6 ms | 8.30 GB/s |
| E: / 102 | 59.5 ms | 60.4 ms | 61.0 ms | 7.23 GB/s |
| D: USB / 103 | 3.024 s | 3.025 s | 3.025 s | 5.62 GB/s* |

`*` The USB aggregate-rate p50 is dominated by later warm-cache samples; its
first measured read was 581 ms and its first single-drive aggregate rate was
28.9 MB/s. This is evidence of mixed cache state, not a contradiction: the
capture deliberately records OS cache as `unknown` and does not flush or infer
physical traffic.

The pair and triple groups completed without short reads. The emitter produced
nine singleton-drive profile rows (three load classes for each drive) and no
fabricated controller/upstream profile for D:. Multi-drive rows remain
aggregate evidence and are not copied to each drive.

For a purely illustrative p99/2x sensitivity review, the largest observed
levels within the bound were 2 for C:, 4 for E:, 2 for USB D:, 4 for the C/E
pair, 4 for each NVMe/USB pair, and 8 for the triple. The generated review is
[`m3-execution-step7-calibration-3drive-review-p99-2x-2026-09-07.json`](m3-execution-step7-calibration-3drive-review-p99-2x-2026-09-07.json).
Because cache state is unknown and the multiplier was not a pre-registered
product criterion, these are sensitivity results, not admission limits.

## Interpretation

This proves that the USB copy is a materially different, slower candidate and
should be represented as its own drive resource. It does not prove that D:
shares a controller or upstream link with either NVMe, and it does not make
the observed inflight levels safe queue-depth bounds. Cache classification,
repeated held-out validation, and shared-controller/upstream topology remain
open. No planner admission profile is promoted from this capture.
