# MiniMax-M3 Step 7c — full-model runtime route/read smoke

## Result

The full local MiniMax-M3 snapshot was run twice with the same prompt,
`SEED=4242`, bounded PIPE/parallel settings, and the same C/E replica mirror:
once with shadow disabled and once with shadow enabled. Both engine runs
exited successfully.

The model-aware config generator verified both safetensors replicas and
emitted 7,296 expert identities / 14,592 copy rows. The shadow runtime loaded
all six topology resources and all 14,592 copies:

```text
[M3_SHADOW] enabled: 6 resources, 14592 copies, 2 replica map entries
```

The enabled run produced 453 completed shadow observations. Every observation
completed successfully and used one of the two complete physical paths:

```text
route 101: 306 reads
route 102: 147 reads
bad/incomplete rows: 0
unique layers: 57
unique (layer, expert) pairs: 444
```

The engine trace contained 10,366 events in both runs. Comparing the semantic
multiset `(kind, layer, expert, resource)` while ignoring wall-clock/thread
ordering gave exact equality: 0 events only in shadow-off and 0 only in
shadow-on. Parallel event order is intentionally not compared.

The shadow trace also carried the admission fields required for later
calibration joins: `actual_load`, `consumer_need_ns`, `predicted_ready_ns`,
`issue_ns`, and `ready_ns`. For the 453 completed rows, both predicted and
actual readiness were before the synthetic consumer need on all 453 rows. The
diagnostic `ready_ns - predicted_ready_ns` error had a 51.7 ms median and
74.6 ms p95 (15.0--91.8 ms). These numbers describe this run with synthetic
profiles and a deliberately spacious need time; they are not held-out
calibration bounds.

The reproducible machine-readable report is
[`m3-execution-step7-full-model-runtime-report-2026-09-07.json`](m3-execution-step7-full-model-runtime-report-2026-09-07.json).
It reports the same evidence both by physical drive/path and by controller:
controller 201 has 306 samples and controller 202 has 147 samples, with zero
deadline misses in either group. This is an evidence join, not proof of
shared-controller contention; the two controllers are physically distinct on
this host.

## Scope limits

The generated profiles are explicitly synthetic and were used only to make
the runtime admission path executable. The generator validated the complete
7,296-key inventory and the runtime loaded all 14,592 copy rows; this proves
full-model inventory/configuration loading. The 453 observed rows covered 444
of those expert identities and prove candidate enumeration, physical-path
expansion, lease/start/completion registration, and shadow-on semantic trace
equality for the observed requests. They do not prove that every identity was
observed in a live route. This is not storage calibration, a latency result,
or a Gate-P performance promotion.

Mixed-residency tensors, direct-striped reads, mmap physical traffic,
io_uring, same-controller/upstream contention, sanitizer runs, and held-out
performance remain separate Step-7 requirements.
