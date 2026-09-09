# MiniMax-M3 Step 7 gate — interim evidence

## Status

**C/M dependency PASS in bounded shadow scope; overall Gate 7 remains NOT
GATED / NOT PROMOTED.** This report records the implemented bounded planner
slice and its limits; it is not approval for active scheduling or a final
Step-7 performance gate.

The dependency table permits Step 8a shadow-only preparation after Step 7 C/M.
It does not permit active planner admission: P, real shared-resource
contention calibration, sanitizer evidence, and the final full-diff review
remain open.

## Evidence that passes

- `make -C c test-shadow-m3 test-shadow-observer-m3 test-shadow-store-m3
  test-shadow-registry-m3 test-shadow-m3-bridge` passes on the Windows MSYS2
  build used for this checkout.
- The focused fixtures cover shared-controller bandwidth, defer versus an
  independent replica, startup-latency asymmetry, three controller groups
  over an optional shared upstream, hard in-flight/byte bounds, missing
  calibration, failure cleanup, and deterministic decision logs.
- `m3_shd_predict` is logically pure: it does not append to the log or alter
  the real active ledger. Hypothetical defer evaluation drops only
  predicted-complete entries from its local calculation. The planner-owned
  ledger/log boundary is protected by a bounded C11 atomic guard for
  concurrent observer calls.
- Candidate evaluation uses a bounded fluid event loop: startup transitions,
  transfer-count changes, and completions advance a fake monotone time and
  recompute per-resource rates. Hard admission uses the current snapshot;
  defer is only a private counterfactual.
- An optional calibrated conversion profile now forms a bounded serial
  virtual stage after I/O completion. It orders equal-ready flows by request
  ID, contributes its own duration/residual, rejects conversion-capacity
  overflow, and returns `UNKNOWN` for missing duration classes. Storage-only
  profiles remain behaviorally unchanged; the deterministic fixture covers
  the 0.2 s I/O plus 0.1 s conversion and 0.3/0.4 s serial-ready results.
- The live observer has an atomic decide-and-record operation: counterfactual
  selection and registration of the actual issued path share one planner
  guard, so concurrent observer calls cannot both decide against the same
  stale occupancy snapshot. The operation can also return one consistent
  post-registration occupancy snapshot for the complete actual path. Focused
  regressions cover the second-copy selection after the first path is
  registered, clock-callback re-entry, automatic IDs, and invalid candidate
  bounds.
- The planner fixture retires the same three active requests through four
  controlled completion-order permutations and compares the resulting
  decision-log bytes; all permutations are equal.
- The topology/profile text loader, explicit request states, bounded queued
  demand ledger, and need/enqueue/request-ID priority ordering are covered by
  parser and queue fixtures. Queue overflow and byte overflow are rejected
  before publication; duplicate resources and cyclic parent topology are also
  rejected before a planner is opened.
- The queue policy now accepts an optional calibrated `queue_age_ns` bound.
  A fake-clock regression confirms that an age-priority request outranks a
  newer request with an earlier consumer need, while aged ties remain ordered
  by enqueue time and request ID. The bound is included in the profile hash.
  Weight-copy rows now also require a real drive resource, and an explicitly
  unavailable matching copy cannot silently fall back to the observed drive;
  both mapping cases have focused regressions.
- Calibration requires explicit units/second, nonzero samples, matching
  topology hash, and a pinned profile version. Missing or contradictory
  present evidence is rejected rather than filled with a default capacity;
  a declared resource with no direct profile remains `UNKNOWN`.
- The 7b-ext calibration slice now emits planner-readable topology text for
  `cpu_island`, `gpu_island`, `memory_dom`, and `link` entries and parses it
  end-to-end. The topology fixture and calibration emit gate pass under
  `-Werror`; missing transfer/GPU and same-controller measurements remain
  `NOT_RUN`. Controller/upstream rows without direct measurements remain
  profile-less and therefore cannot contribute fabricated capacity.
- The profile hash now covers every kind-specific topology payload
  (`isa_caps`, cores, VRAM/P2P, parent links, rates, bandwidth class, and
  eligibility); a regression rejects hash equality after a payload change.
- A follow-up machine-A capture now records inflight 8 and 16 with 16 samples
  per cell. The combined p99/2x review reports observed candidates only;
  cache state remains unknown, so these results are sensitivity evidence and
  do not become planner admission bounds.
- `COLI_EXPERT_STORE=shadow-auto` now wraps the real backend behind a
  transparent forwarding store. The adapter preserves views, leases, GPU
  handles, stats, return values, and destruction; its forwarding contract is
  covered with fake-store and registry tests.
- The live hook records lookup count, failures, release count, and successful
  view-byte totals inside planner-owned telemetry; it does not insert a
  request into the hypothetical ledger without a configured path snapshot.
- The adapter now accepts an explicit snapshot-provider callback. With a
  provider, a successful real lookup creates a planner-only request, records
  a deterministic shadow decision, and removes the temporary queue entry;
  the store's real lease and active occupancy remain untouched. Drive-only
  provider paths are expanded through their declared controller/upstream
  ancestors before prediction, so shared-controller limits cannot be bypassed.
  This seam is covered by the store forwarding fixture.
- `m3_shadow_m3_bridge.{c,h}` provides a pure, tested translation from a
  MiniMax-M3 load/DAG snapshot (forward/generation/consumer node, bytes,
  deadline, and drive) into the planner request and drive-only candidate.
  `m3_shadow_m3_runtime.{c,h}` now carries copied per-job context through the
  real `expert_load()` path under the opt-in `COLI_M3_SHADOW_RUNTIME` build
  define. The context is restored around each PIPE/OpenMP job; it is not used
  as a process-global execution context.
- `make -f Makefile.deepseek-v4 deepseek-v4` links the Windows engine with
  the planner and transparent store adapter objects; no default-path build
  break was introduced.
- `make -C c colibri` remains green with the opt-in runtime seam compiled in;
  omitting the environment configuration leaves the normal engine path
  unchanged.
- The focused ASan/UBSan and TSan commands now cover all five shadow-boundary
  binaries (planner, observer, store wrapper, registry, and M3 bridge) in CI workflow
  `.github/workflows/m3-step7-shadow.yml`; hosted CI evidence is still
  pending for this commit. The local Windows MinGW toolchain cannot run that
  sanitizer build because its `libsanitizer.spec` is absent.
- The pure M3 bridge was recompiled with `-Werror` together with the four
  boundary fixtures; all five local tests pass. It preserves the model,
  forward, generation, consumer-node, deadline, key, byte, drive, and demand
  fields, rejects incomplete snapshots, and now exercises drive-to-
  controller-to-upstream expansion before pure prediction. This remains a
  translation seam, not live `colibri.c` integration.
- The store byte accounting now has an explicit overflow regression: a
  `data_bytes + scale_bytes` overflow saturates to `UINT64_MAX` before the
  aggregate observer counter is updated. The forwarding and lifecycle test
  remains green.
- Duplicate real completion is now covered: the second completion is rejected
  and the active ledger count is unchanged. GCC `-fanalyzer` and Clang
  warning-only syntax checks are clean for the planner, store, and M3 bridge
  sources.

## Still missing — blocking

- The M3-specific observer seam now exists and the synthetic tiny run records
  real `expert_load()` decisions plus bounded correlated completion rows. The
  on/off replay compares semantic engine output and exit status and passes,
  including normalized `COLI_TRACE` lifecycle-event equality. The replay now
  configures two complete equivalent copies on independent
  drive/controller/upstream paths for both observed experts. Its atomic
  decision/start operation records the second copy as the counterfactual when
  the first is occupied, while the actual engine route remains unchanged.
  The replay harness pins the bounded parallel contract and eight workers for
  both shadow-off and shadow-on runs, making that overlap requirement
  reproducible rather than dependent on host scheduling.
  This is stronger production-seam evidence for the tiny case; the runtime
  trace rejects incomplete or mixed six-component mappings from accuracy
  statistics. A separate full-model shadow-off/on route/read smoke is recorded
  below; it uses synthetic profiles and remains C-only evidence.
- The full-model generator validated both safetensors replicas against the
  M3 config-derived 7,296-key inventory and emitted 14,592 copy rows. The
  runtime loaded all rows; the fixed-seed shadow-on run completed 453 rows
  covering 444 identities, and its semantic engine trace matched shadow-off
  exactly (10,366 events per run). The report also aggregates ready-time
  error/deadline evidence by drive/path and controller (201: 306 samples,
  202: 147 samples). This does not prove that every identity was live-observed,
  nor does it supply hardware calibration or P evidence.
- The bounded runtime trace is now consumable by an offline report tool that
  computes ready-time error and actual/predicted/guarded deadline misses per
  drive/resource path. The tiny report has two successful rows and no misses;
  it is evidence for the accounting path, not a Gate-7 accuracy result.
- The replay still supplies an explicit synthetic consumer-need delta because
  its tiny fixture does not enable an M3 DAG executor. On eligible production
  serial M3 DAG paths, the live seam now records one observed monotonic
  `need=now` timestamp when the current consumer is ready except for its
  missing weights. It is not refreshed by later predictions. PIPE/parallel
  paths and any path that cannot establish that readiness remain UNKNOWN.
- The current tiny configuration uses two synthetic complete copies on
  independent drive/controller/upstream paths. It proves production-seam
  candidate enumeration and counterfactual selection while the actual route
  remains unchanged. It is still only a two-expert replay: full-model
  route/lease/read equality, mixed-residency tensors, direct striped reads,
  mmap physical traffic, and io_uring remain explicitly unknown/unobserved.
- The bounded text calibration/topology parser now exists and rejects
  malformed/incomplete rows. A Windows capture now records both available
  NVMe devices at inflight 1/2/4, with per-sample p50/p95/max latency and
  aggregate bytes/s in
  `m3-execution-step7-calibration-2026-09-07.json`; the initial p99 capture is
  `m3-execution-step7-calibration-p99-2026-09-07.json`, followed by a
  16-sample p99 matrix in
  `m3-execution-step7-calibration-p99-16-2026-09-07.json`. Its illustrative
  2x review selects observed full-group levels 2/4/4 for resource 101,
  resource 102, and the pair respectively; these remain sensitivity results,
  not accepted admission bounds. It is still raw calibration evidence: cache
  condition is unknown, same-controller/shared-upstream contention is not
  measured, and the capture does not synthesize planner profile rows or safe
  maximum queue depths. Hardware P remains
  `NOT_RUN`, including the all-resident performance comparison and
  six/seven-drive measurement.
- A separate full-M3 copy-table scale fixture now parses and opens 15,360
  copy rows (60 layers x 128 experts x two replicas) and repeats first,
  middle, and last-key lookup 3,000 times. It remains bounded-capacity
  evidence; it does not replace the full-model smoke's missing mixed-residency,
  direct-striped, mmap, io_uring, or performance evidence.
- The recorded 57-digest baseline remains separate evidence. The new
  `test_m3_shadow_runtime_replay.py` compares the same tiny run with the
  observer disabled/enabled and requires a non-empty live decision log plus a
  correlated runtime-trace row, but it does not replace a full digest/route
  trace replay or hardware calibration.
- Local `test-asan` could not start on this MSYS2-GCC because the toolchain
  lacks `libsanitizer.spec`; this is an environment limitation, not a
  sanitizer pass. Hosted ASan/UBSan/TSan evidence remains required.
- The stronger-model final diff review required by `AGENTS.md` remains
  outstanding. An Astra review attempt returned a usage-limit error before
  producing findings; this report does not substitute for that review.
- The three planner concerns identified by the compact Astra review are now
  addressed locally: admission capacity is checked under the guard, a bounded
  decision-log snapshot API exists alongside the quiescent-only pointer
  accessor, and request IDs have guarded duplicate/exhaustion handling with
  queued-to-active promotion tests. The combined decide/start contention test
  and request-ID-joined live replay now also pass; the narrow Astra follow-up
  found no blocker for those two evidence points. This is still not independent
  Gate-7 approval.
- The simulator now preserves the declared `uint64_t` calibrated rates through
  its duration and shared-rate arithmetic; a 5,000,000,000-byte/s regression
  prevents silent `uint32_t` truncation.
- The calibration-tail refinement is independently reviewed in a compact
  Astra pass: only explicit p95/p99 labels are accepted for non-zero residuals;
  p90/unsupported labels are rejected by fixtures. No blocker was found in
  that scoped review. This does not replace the remaining full Gate-7 review.
- The live `consumer_need` refinement received a second scoped Astra check:
  serial M3 DAG `need=now` capture is accepted, while PIPE/parallel paths stay
  `UNKNOWN` until post-shared-compute readiness is explicit. No blocker was
  found in that narrowed condition.

## Reproduction

```text
make -C c test-shadow-m3 test-shadow-observer-m3 test-shadow-store-m3 \
  test-shadow-registry-m3
```

The standalone results are sufficient evidence for the bounded simulator
slice only. They do not close Gate 7.
