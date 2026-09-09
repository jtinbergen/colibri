# Step 8b — bounded prefetch/promotion, fixture-only — 2026-09-07

## Verdict

**C/M dispatcher fixture slice: implemented and Sol-approved.  Gate 8b is
not promoted; production remains fallback-only.**

No current calibration/profile is trusted to issue a live prefetch, change a
source, or alter residency.  The runtime continues to construct the Step-8
dispatcher with `trusted_policy=0`, so this work cannot alter existing model
I/O or numerical output.

## Implemented contract

- Prefetch and demand share the single dispatcher and its full-path resource
  counters.  There is no prefetch-only bandwidth, controller, upstream, byte,
  or request budget.
- A bounded per-storage demand reserve is validated against eligible resource
  capacity.  Prefetch may enter only when the reserve remains above all
  current reservations and a pure injected predictor reports that projected
  demand wait does not rise.  Missing/worse prediction fails closed.  Retry
  uses the same gate; it cannot bypass it through `reserve_all`.
- An exact immutable coalescing key binds content ID, selected copy, byte
  offset/length, topology/profile, and ordered source path.  Promotion
  attaches a demand consumer to the existing prefetch; it creates no second
  request, reservation, or I/O part.  Queued promotion first reclassifies the
  request as demand and performs ordinary demand admission.
- Consumer leases are individually identified and dispatcher-wide unique.
  Each stores the promoted demand request/generation/lease binding.  Resource
  release, producer-lease release, and every consumer release must occur
  before retirement/buffer reuse.
- Speculative cancellation is allowed only from `QUEUED`/`RESERVED`.
  Issued prefetch is only marked no-longer-wanted and retains capacity until
  terminal completion.

## Evidence

| Check | Result |
|---|---|
| `test_m3_active_admission` | PASS — reserve, bad/missing prediction defer, retry cannot bypass gate, queued/reserved promotion, no duplicate issue, per-consumer release, dispatcher-wide token uniqueness, partial issue, cancellation, reset, shared-resource isolation |
| `test_m3_shadow_m3_runtime` | PASS |
| `test-shadow-m3-runtime-replay` | PASS — shadow off/on engine stdout, exit code, and `COLI_TRACE` match; runtime log/trace nonempty |
| Stronger-model final review | APPROVED after repaired retry-gate and consumer-token findings |
| `git diff --check` | PASS |
| Local ASan/UBSan/TSan | NOT_RUN — this MSYS2 install lacks the sanitizer runtimes; Linux target includes the dispatcher test |

## Remaining gate conditions

1. Obtain Linux sanitizer evidence and real fake-I/O/admission trace capture
   in the supported CI environment.
2. Step 7 still lacks a trusted active calibration, held-out P evidence, and
   shared-controller classification.  Active source choice/prefetch remains
   disabled.
3. Step 8c must still add its planner-side shadow residency actions and
   machine fixtures.  The separate full `ecap`/`ecache` audit is recorded in
   `m3-execution-slice-8c-ecache-audit-2026-09-07.md`.
