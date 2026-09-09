# Step 8a — source-admission dispatcher, shadow-only — 2026-09-07

## Verdict

**C/M fixture slice: implemented and locally passing.  Gate 8a is not
promoted and active source selection remains disabled.**

This is the minimum durable active-admission boundary permitted by the current
Step-7 C/M-only transition.  It is installed at the M3 `expert_load()`
pre-I/O seam, but is constructed untrusted and therefore returns `FALLBACK`.
The existing engine source, I/O publication, and output path remain unchanged.
There is no claim of real-hardware P evidence, calibrated active source
selection, prefetch, or residency optimization.

## Current verification — 2026-09-08

The current checkout re-ran the bounded dispatcher and runtime seam with
`EXTRA_CFLAGS=-Werror`; all existing shadow fixtures, the admission fixture,
and the runtime fixture pass.  The tiny shadow-off/on replay was rebuilt with
the documented Python 3.13 interpreter and again produced identical engine
normalized stdout, exit status, and normalized `COLI_TRACE` output.

One fallback-boundary regression was tightened during this verification:
when the dispatcher is opened with `trusted_policy=0`, even an incomplete
candidate path returns `M3_ADM_FALLBACK` without allocating a ledger entry.
This preserves the no-profile/unknown-mapping contract before descriptor
validation can turn the request into a planner rejection.

Teardown is now also ownership-checked: `m3_adm_close()` refuses to free a
dispatcher with live queued, reserved, or issued requests.  The bounded
ledger fixture resets, lease-releases, retires, and then closes a full queue.
Teardown closes admission before that check and requires the owner to
quiesce other API callers.  Dispatcher-local ticket nonces also prevent a
retired request's stale event from matching a later request with the same
external identity.

This is still preparation evidence only.  The follow-up diff requires the
required stronger-model review and hosted sanitizer evidence before any Gate
8a promotion; the production runtime continues to construct admission with
`trusted_policy=0`.

The shared dispatcher source currently also contains the later 8b prefetch /
promotion helpers from the adjacent shadow slice.  This report accepts only
the 8a demand-admission behavior; no prefetch is enabled by the production
runtime or claimed by this gate.

## Implementation

- `c/m3_active_admission.{h,c}` owns a fixed ledger and all mutable
  drive/controller/upstream counters under one guard.  It reserves the full,
  variable-length path as one transaction and retains deferred requests as
  `QUEUED` until `reserve_all` succeeds.
- Request identity binds request ID, generation, destination-lease ID and
  lease generation.  Terminal events also bind a bounded part ID.  Duplicate
  or stale terminal events do not mutate counters.
- The normal lifecycle is `QUEUED -> RESERVED -> ISSUED ->
  COMPLETED/FAILED -> RETIRED`; queued/reserved cancellation and issued
  cancel-pending follow the Step-8 contract.  A partial issue has an explicit
  `close_issue` boundary and holds capacity until issued parts drain.
- I/O capacity and destination lease are separate: resource counters release
  after final terminal I/O; lease release and retirement are a later explicit
  operation.  Reset stops admission and is not drained until every request
  and lease is retired.
- `c/m3_shadow_m3_runtime.c` invokes the dispatcher before the existing
  observer registers the actual load.  It opens the dispatcher with
  `trusted_policy=0`; the probe returns `FALLBACK` and the existing caller
  continues the unchanged load path.  The hook is not itself an I/O
  publication barrier; any future trusted activation needs a separate
  publication/completion adapter.  This keeps synthetic or unknown Step-7
  profiles from changing production behavior.

## Evidence

| Check | Result |
|---|---|
| `test_m3_active_admission` | PASS — shared-resource defer, independent path progress, retry, full bounded ledger, atomic rollback, stale/double terminal, queued/issued cancellation, partial issue, reset-drain, lease uniqueness |
| `test_m3_shadow_m3_runtime` | PASS |
| `test-shadow-m3-runtime-replay` | PASS — shadow off/on normalized stdout, exit code, and normalized `COLI_TRACE` match; runtime log and trace nonempty |
| `git diff --check` | PASS |
| local ASan/UBSan | NOT_RUN — installed MSYS2 linker lacks `libasan` and `libubsan` |
| TSan | NOT_RUN locally — target includes this test for the Linux sanitizer route |

The replay used the existing `m3tiny_i8` fixture and
`docs/results/m3-execution-step7-fixtures/m3_runtime_tiny/config.txt`, with
`PYTHON=/c/Python313/python.exe` under MSYS2.

## Remaining conditions

1. The historical review approved the earlier slice after two repaired P1
   findings.  The current fallback/teardown/ticket follow-up has received an
   independent review; its final ticket-output repair is now covered by the
   focused fixture.  No Gate 8a promotion is claimed.
2. Linux ASan/UBSan/TSan evidence remains required before any concurrency
   safety promotion.
3. Step 7 lacks an admitted trusted calibration/profile, held-out P evidence,
   and shared-controller classification.  Do not enable active selection.
4. Step 8b still must add demand-reserved prefetch and promotion without a
   duplicate load.  Step 8c still needs its complete `ecap`/`ecache` audit
   and shadow-only residency policy.  Neither is implemented by this slice.
