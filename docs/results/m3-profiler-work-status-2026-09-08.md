# M3 profiler work status — 2026-09-08

## Current state

The profiler now has an exploratory shared-contention matrix path. Multi-drive
groups default to a bounded Cartesian sweep of independent per-drive queue
depths, with a common barrier start, aggregate/per-drive throughput, latency
summaries, and raw matrix metadata. The current machine manifest explicitly
marks `shared-upstream-901` as `sweep: matrix`.

The raw capture and planner artifacts remain evidence-only. Active admission is
still disabled and must remain disabled.

The authoritative machine run is now the disjoint-shard capture described in
`m3-machine-A-disjoint-calibration-v2.md`. It uses 18 existing shards per
drive, capture-wide range accounting, and `range_reuse=false`.

## Files changed in this continuation

- `c/tools/m3_shadow_calibrate.py`
  - Added bounded `matrix_cells()` enumeration.
  - Added `measure_matrix()` with per-drive queue-depth vectors and common
    start/latest-completion timing.
  - Added matrix CLI controls and capture metadata.
  - Added matrix-target planner emission for explicit controller/upstream
    resources.
  - Added ordered multi-shard source pools, capture-wide disjoint range
    allocation, preflight slot accounting, and explicit unmeasured topology
    comments.
- `c/tools/m3_shadow_calibration_review.py`
  - Preserves matrix queue-depth, per-drive, and fairness metadata in review
    output.
- `c/tools/m3_shadow_calibration_promote.py`
  - Emits candidate profile rows only from a provenance-matched review; it
    rejects reused-range captures and always marks output inactive.
- `c/machine_profile.py`
  - Recognizes matrix metadata and only reports shared contention evidence when
    a complete matrix cell is present.
  - Retains the manifest topology graph, including unmeasured link/GPU
    declarations, in the profile evidence bundle.
- Tests updated for matrix enumeration, capture, review metadata, and profile
  coverage.
- `docs/m3-profiler-contention-invariants.md`
  - Written measurement/ownership contract.
- `docs/results/m3-machine-A-calibration-manifest-2026-09-08.json`
  - Explicitly marks the shared-upstream group as matrix mode.

## Verification completed

- `python -m unittest c.tools.test_m3_shadow_calibrate`: **20 passed** (including
  barrier-failure, topology, and range-accounting coverage).
- `python -m unittest c.tools.test_m3_shadow_calibration_promote`: **2 passed**.
- `python c/tools/test_m3_shadow_calibration_review.py`: **PASS**.
- Python compilation of calibration/review tools: **PASS**.
- Real-file smoke capture on this machine: **PASS**; produced matrix cells and
  planner rows.
- Full disjoint direct-I/O capture: **PASS**; 46 measurements, 2,448 required
  slots and 2,556 available slots per drive, with `range_reuse=false`.
- C planner parser/open gate: **PASS**; 10 topology resources and zero raw
  profile rows.
- Full-model configuration gate: **PASS**; 7,296 experts and 14,592 copies.
- Reviewed candidate planner/configuration gates: **PASS**; 5 candidate rows,
  still `candidate_only` and inactive.
- `python -m unittest tests.test_machine_profile`: **5 passed**.

## Review and remaining gates

The matrix review now uses isolated-throughput retention, normalized fairness,
raw request-completion tails, and a downward-closed observed envelope. Raw
planner emission is topology-only, so no unreviewed rate is injected into the
planner. The disjoint capture is versioned and source-provenance checked. It remains
unsuitable for admission because hardware-DMA/PCH evidence is unverified, reads are not
OS-overlapped asynchronous I/O, and stronger-model final review plus held-out
validation remain open:

1. Attest/validate the production storage-to-memory DMA path separately from
   `host_memcpy`, and measure PCH/DMI traffic capacity.
2. Add held-out planner validation and repeat parser/replay/sanitizer gates.
3. Obtain the required stronger-model final review before any admission change.
4. Implement and validate an OS-overlapped backend before treating queue depth
   as asynchronous controller capacity.

No active planner or performance claim is approved by this status.

## Continuation update

The profiler has since gained a platform direct-read backend and sustained
worker loops. On this Windows machine, a real full-size run completed with
`win32-no-buffering`, 26 matrix cells, 16 full two-drive cells, 16 samples,
and one operation per worker. The authoritative disjoint shared-upstream
review produced an observed candidate total inflight of 2 with minimum
per-drive retention approximately `0.746`; the review remains
`CANDIDATE_ONLY` because DMA/admission gates are still open.

Artifacts from that run are listed in
`docs/results/m3-machine-A-disjoint-calibration-v2.md`.

## Resume checkpoint (superseded by 2026-09-09 continuation)

The three code fixes identified by the independent stronger-model review have
now been implemented and verified:

1. Bind promotion provenance to the complete capture contents with a
   `capture_sha256`; reject modified measurements or `read_bytes` values.
2. Make review scope explicit and order-independent: reject incompatible
   request/topology captures, and retain the conservative minimum isolated
   rate at the most specific target scope.
3. Preserve the reviewed tail percentile in emitted candidate rows instead of
   hardcoding the p95 label.

The calibrator, reviewer, promoter, planner emitter, and full-model qualifier
now bind and preserve the capture digest; tests and authoritative artifacts
were regenerated.

The final stronger-model review is still required after those fixes. The
external gates remain unchanged: DMA/read-to-ready evidence is
`OBSERVED_NOT_ATTESTED`, the
current backend reports `asynchronous=false`, held-out planner validation is
unproven, and active admission must remain disabled.

## 2026-09-09 continuation

Completed the integrity/review fixes and regenerated the authoritative
artifacts:

- v2 captures now bind `capture_sha256` after every optional field is added;
  review and promotion reject altered request sizes or measurements.
- Review rejects incompatible request/topology scopes, is order-independent,
  retains the conservative minimum isolated rate, and preserves p95/p99 labels.
- Windows endpoint probing uses unprivileged `pnputil` output. The capture now
  records controller 201 as negotiated PCIe Gen4 x4 (about 7.877 GB/s payload
  upper bound), controller 202 as Gen3 x4 (about 3.938 GB/s), and the GTX 1080
  endpoint separately.
- Direct reads are recorded as `OBSERVED_NOT_ATTESTED` read-to-ready evidence;
  this is not a hardware-DMA attestation. PCH/DMI traffic capacity remains
  unmeasured.
- Missing profile/telemetry has an explicit `original_colibri_baseline`
  fallback: conservative `storage_bounds.max_inflight` and `rate=UNKNOWN`,
  never a silently invented bandwidth.

Verification after regeneration: 28 calibration/promoter unit tests passed,
review regression passed, machine-profile tests passed, full-model generation
passed with 7,296 experts and 14,592 copies, and candidate output remains
`CANDIDATE_ONLY`/inactive.

The requested stronger-model final review could not be obtained because the
subagent service reported its usage limit. This remains an explicit review
blocker. Hardware-DMA attestation, PCH/DMI traffic measurement, true
OS-overlapped I/O, and held-out planner validation also remain open; the
original Colibri baseline remains the safe fallback.

## 2026-09-09 topology and fallback continuation

The profiler contract was tightened so coverage distinguishes a missing
measurement from a non-applicable one. The current machine profile now reports:

- drive service and controller service: `EVIDENCE_ONLY`;
- shared controller: `NOT_APPLICABLE` (one drive per controller);
- shared upstream: `EVIDENCE_ONLY`;
- host memory: `EVIDENCE_ONLY`;
- storage-to-ready: `OBSERVED_NOT_ATTESTED`;
- negotiated PCIe topology: `OBSERVED`;
- NUMA inventory: `OBSERVED`, one node;
- GPU transfer: `NOT_APPLICABLE` (all declared GPUs are ineligible);
- PCH/DMI counter: `UNAVAILABLE` on this installation; no capacity is inferred.

`coli plan` now emits the same explicit `original_colibri_baseline` contract
when no profile exists or a supplied profile is invalid: admission is disabled,
unknown capacity remains conservative, and unavailable link/DMA rates remain
`UNKNOWN`. This metadata does not alter the original placement path.

The platform endpoint probe now supports Windows PnP, Linux `lspci`, and
best-effort macOS `system_profiler` bindings. On Windows it also records the
parent device chain and common ancestor; the current capture finds the two
NVMe root ports under common PCI root-complex ancestry while preserving their
distinct negotiated endpoint links. This is topology evidence, not a PCH/DMI
bandwidth claim.

Windows source paths now also carry read-only volume-extent evidence: the C:
volume backing drive 101 maps to physical disk 1, and the E: volume backing
drive 102 maps to physical disk 0. This validates path-to-disk identity without
claiming that disk numbering proves controller or PCH ownership.

The authoritative capture was regenerated after these source changes and its
self-digest/tool-source digest were verified. It now uses Windows
OS-overlapped direct I/O (`win32-overlapped`, `asynchronous=true`). Reviewed
controller candidates are controller 201 at inflight 2 and approximately
5.375 GB/s, and controller 202 at inflight 4 and approximately 3.112 GB/s.
The shared-upstream 901 matrix has no candidate under the p95, retention, and
fairness policy, so its rate remains UNKNOWN. Host-copy evidence is
approximately 8.298 GB/s p50 and 8.408 GB/s p95 for the 256 MiB buffer.
Review output remains `CANDIDATE_ONLY`; active admission is disabled.
The raw rows record a `submit_first_wave_before_reap` protocol with pending-submission
counts and submission skew; exact hardware queue depth remains unattested.
The authoritative matrix uses 8 samples and 2 disjoint operations per worker,
so each sample includes a refill operation while capture-wide range reuse stays
false.

## 2026-09-09 provenance and final-audit continuation

The stronger-model review completed and did not approve active admission. Its
three provenance findings are now closed in the toolchain:

- full-model configurations with profile rows require and preserve the exact
  reviewed-candidate JSON schema and SHA-256 digest;
- legacy v1 captures are evidence-only and cannot be promoted;
- `machine_profile.py` verifies the v2 capture self-digest before reporting
  capture evidence.

The regenerated raw and candidate planner text both pass the C parser/open and
sparse-calibration semantics gates. The candidate full-model configuration
contains four reviewed rows, the review digest, and
`active_admission_allowed = False`; the raw configuration remains
topology-only. Current focused verification is 39 calibration/promoter unit
tests, 8 machine-profile tests, the review regression, Python compilation, and
both planner parser runs.

`coli profile` now accepts `--review` and embeds the hash-bound review summary
and four candidate rows in the machine profile and attached plan metadata. The
rows remain explicitly evidence-only; attaching them does not alter placement
or admission.

The remaining blockers are measurement/admission gates, not missing fallback
behavior: hardware DMA attestation, PCH/DMI capacity measurement (currently
`UNAVAILABLE`), held-out planner validation, and the required performance
evidence for any future admission change. The overlapped backend is now
implemented and exercised, but it does not by itself attest DMA or authorize
admission; until the remaining gates close, the planner must use the explicit
original Colibri baseline contract for unknown paths.
