# MiniMax-M3 Step 8 closure decision — 2026-09-09

## Decision

Step 8 is **closed for the current desktop host as `NO-GO FOR PROMOTION`**.

This is an operational stop for the current machine, not a claim that the
Step 8 implementation is universally complete or that the research artifacts
should be deleted.

## Scope

Step 8 covered bounded active I/O and residency planning:

- 8a: source selection;
- 8b: bounded prefetch;
- 8c: residency and e-cache planning.

## Evidence and resulting policy

- Runtime profiles show the workload is I/O- and expert-residency-bound, but
  the current machine does not provide a repeatable, tail-safe improvement
  from active source selection or prefetch.
- The real pilot improved observed hit rate but reduced throughput and worsened
  p95 latency.
- Fixed hot-store residency did not produce a promotable gain.
- Dual-drive execution shows that the mirror participates, but the measured
  signal is small, non-counterbalanced, and does not establish a safe active
  admission limit.

Therefore production remains on the conservative Colibri fallback behavior:

- no untrusted profile may control live source selection;
- no untrusted profile may promote prefetch or residency policy;
- theoretical specifications and shadow observations remain advisory;
- the captured telemetry and reports remain available for later comparison.

## Reopen conditions

Reopen Step 8 on a materially different configuration or after new evidence,
such as more usable RAM, an independent storage/controller path, or another
compute island. Reopening requires trusted calibration, held-out workload
evidence, scoped per-resource tail measurements, and an explicit review before
promotion into active admission.
