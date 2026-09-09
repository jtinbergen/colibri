# Packet: Slice A2 — 8a-lite static source rule (machine A)

Deliverable: a static, opt-in policy that prefers C: for demand reads and
treats E: as prefetch-only behind a slack threshold. No fluid sim. No 7c
dependency.

## 1. Scope and invariants

- One behavior change: a new admission flag `COLI_M3_SOURCE_PREF=demand-c`
  (or similar — final name decided by the implementer within the packet).
- Fallback: when the flag is unset or topology is missing the C: replica,
  the existing source-selection path is taken.
- No replacement of the engine’s source selector. The policy is an
  advisory layer in front of `expert_load()`; the existing `disk_bw_probe`
  expression in `c/colibri.c` (call site per base plan codekaart) is
  untouched.
- E: demand reads are rejected with a one-line log, not silently rerouted.
- The rule fires only on configured machines where E: is configured as a
  prefetch-only drive; on other machines the flag is a no-op.

## 2. Tasks

1. Add the flag plumbing: env read at startup, one boolean per machine
   entry in the planner topology, default off.
2. Wrap the existing `expert_load()` source choice with the rule.
3. Emit a `decision_source` trace line on every demand read: `C-pref`,
   `E-prefetch-skip`, or `fallback-existing`.
4. Add a regression that forces the rule and asserts no E: demand read
   reaches the read syscall on a configured machine.
5. Add the inverse test: flag off → existing path is byte-identical
   (shadow on/off equality per existing Step 7 fixtures).

## 3. Evidence

- 57-digest equality with flag on/off.
- Decision-trace log over a 32-token run: counts per rule outcome.
- Paired tok/s + felt wait, 5x alternating, against the A1 winner
  configuration.

Artifact: `docs/results/m3-execution-slice-a2-static-source-<date>.md`.

## 4. Gate A2

- C: identical outputs with flag on/off.
- M: decision log shows the rule fired on demand and skipped E: prefetch
  behind slack; rule failure paths covered by regression.
- P: paired tok/s + felt wait. Decision gate (~1.2 tok/s on A) is the
  trigger to bank A2 and consider 7c-1.

Stop condition: rule shows zero or negative effect → mark A2
`INCONCLUSIVE` and do not proceed to 7c-1 without a different mechanism.
