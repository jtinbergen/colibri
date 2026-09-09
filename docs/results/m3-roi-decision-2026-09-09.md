# MiniMax-M3 local heterogeneous-DAG ROI decision

Date: 2026-09-09  
Machine: Intel i5-12600K, 32 GB RAM, Windows 11  
Workload: persistent MiniMax-M3 int4, four rotating prompts, 32 generated
tokens, CPU path, no GPU, `PIPE=1`

## Decision

No promotable ROI has been demonstrated for the current local heterogeneous
DAG configuration. The result is negative for the tested strategies and
configuration, not a claim that every future storage or compute-island design
must fail.

The bounded task-DAG remains an opt-in/reference path. It is not an active
admission optimization. The current real cross-layer prefetch policy should not
be enabled for ROI on this machine.

## Evidence

### Current real prefetch, matched effective cap

Baseline and `PILOT_REAL` were compared at the same effective cap of 8
experts/layer and the same 27.8 GB RAM budget:

| metric | baseline | `PILOT_REAL` | change |
|---|---:|---:|---:|
| rotating median throughput | 0.46 tok/s | 0.44 tok/s | -4.3% |
| p95 request | 104.80 s | 112.92 s | +7.8% |
| median hit rate | 43.3% | 54.2% | +10.9 percentage points |
| expert wait | 50.218 s | 50.667 s | approximately unchanged |

The hit-rate improvement did not become serving improvement. The current
prefetch policy is rejected for active use.

### Fixed hot-store residency

The 7 GB hot-store protected 219 experts and left a 228-expert LRU at cap
4/layer. On the rotating workload it produced 0.43 tok/s and 109.47 s p95,
with 38.9% hit rate. This is not better than the matched no-pin result and
shows that a fixed hot set derived from the accumulated history is not robust
for this rotating workload.

### Dual-drive versus primary-only screen

With the same fixed hot-store settings, dual-drive routing produced 0.43
tok/s / 109.47 s p95; primary-only produced 0.42 tok/s / 117.36 s p95. The
observed dual-drive signal is below the 10% acceptance threshold and is not a
counterbalanced physical-I/O experiment because both runs used `DIRECT=0` and
retained page cache. It cannot justify a scalability claim.

## What the profiler proves

The profiler provides hard evidence within its scope that this workload is
expert-I/O-bound: roughly 69--73% of reported phase time is expert I/O, expert
wait is about 50--56 seconds/request, I/O-active wall time is about 50--60
seconds, and aggregate I/O queue depth is generally 1.7--2.1 with a non-zero
QD0 share. It also records fetched bytes, hit rate, cumulative service time,
compute overlap, and resident tiers.

It does not yet prove per-drive saturation or shared-controller behavior. The
mirror probe measures isolated drive rates (approximately 5.07 GB/s primary
and 0.95 GB/s mirror in the latest run), while serving telemetry is aggregate.
Per-drive bytes are now available; per-drive queue depth, per-drive tail
latency, source-choice timing, and shared-parent/controller counters remain
incomplete.

The profiler now exposes runtime source attribution. In the decode smoke with
the same dual-drive 83--84% / 16--17% routing, it reported:

```text
[PROF] storage source: primary 97.622 GB/12260 reads |
       mirror1 21.308 GB/2676 reads | mirror share 17.9% of attributed bytes
[PROF] storage source: primary 97.208 GB/12208 reads |
       mirror1 21.117 GB/2652 reads | mirror share 17.8% of attributed bytes
```

This proves that the mirror is participating in real runtime expert reads,
not merely passing the startup probe. It still does not prove that the mirror
is the limiting resource: the current counters are source-attributed engine
window totals, and buffered reads (`DIRECT=0`) are not identical to physical
device completions.

Runtime source smoke: `m3-profiler-source-smoke32-2026-09-09.txt`.

## Planner policy

The planner may use:

- observed aggregate wait and tail measurements within their recorded scope;
- conservative capacity for unknown/unavailable resources;
- theoretical specifications only as bounds;
- a drive/controller/link hierarchy where siblings share the parent budget;
- a candidate source only when its measured tail and reserved queue budget fit.

It must not add nominal drive bandwidths blindly, infer saturation from an
I/O-bound label, or promote a hit-rate increase into active admission.

## Remaining optional work

The only justified follow-up on this machine is a counterbalanced primary-only
versus dual-drive experiment with per-drive runtime telemetry, ideally with a
controlled cache protocol. A third drive is eligible for consideration only
after topology discovery confirms an independent controller/upstream path and
the scheduler can issue enough independent reads. A third drive on a shared
parent may add contention rather than capacity.

If the counterbalanced storage test remains below 10%, stop local storage/DAG
ROI expansion and redirect hardware effort to larger RAM, measured full
residency, or a genuinely independent compute island. This is a configuration
stop condition, not a claim about all future machines.
