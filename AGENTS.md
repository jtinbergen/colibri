# Agent routing policy

## Model choice

Classify each task or delegated subtask before editing.

The risk classification of the overall change determines the required review
gates. Low-risk, well-bounded subtasks within a high-risk change may still be
delegated to lower-cost models, provided they do not require reasoning about
the high-risk semantics.

Use a strong model (GPT-5.6 Sol, high reasoning; Astra for an independent
review of especially risky changes) when the task or subtask requires
reasoning about:

- concurrency, synchronization, lock-free code, memory ordering;
- ownership, lifetime, cache/residency correctness, cancellation;
- numerical equivalence, quantization kernels, model-output regressions;
- scheduler/DAG design or changes that can affect latency correctness.

Use a balanced model (GPT-5.6 Terra, medium or high reasoning) for:

- bounded implementation following an accepted design and explicit invariants;
- tests and benchmark harnesses that require implementation or interpretation;
- focused refactors whose semantic effect is understood and bounded;
- documentation derived from verified evidence.

Use a low-cost model (GPT-5.6 Luna) for well-specified work that does not
require independent judgment about high-risk semantics, including:

- repository search, inventory, and call-site discovery;
- mechanical edits with an explicit specification;
- formatting and log extraction;
- running prescribed tests or benchmark matrices and collecting their output;
- other deterministic orchestration with explicit success/failure criteria.

Luna may perform low-risk supporting work inside a high-risk change. For
example, it may inventory users of a synchronization primitive, apply a
specified mechanical edit, run a prescribed concurrency test suite, or
collect benchmark results. It must not independently decide that a
concurrency, ownership, numerical-correctness, or scheduler invariant is
correct.

Test execution and evidence collection do not imply authority to interpret
that evidence as establishing correctness or scalability.

A low-cost model may never be the sole reviewer or final approver of a
concurrency, ownership, numerical-correctness, or scheduler change.

## Delegation requirement

When the runtime supports model-routed subagents, delegate work to the lowest
capability tier that can safely perform it, subject to the review rules below.
The active model must not treat this policy as advisory.

- A Luna-led task must delegate high-risk design, semantic implementation, or
  final review to Sol/high (or Astra when an independent exceptional-risk
  review is required). Luna may prepare bounded evidence and mechanical work,
  but may not approve those changes itself.
- A Terra-led task must delegate high-risk design or final review to Sol/high.
  Terra may implement bounded work only after the invariants are explicit.
- Sol may delegate well-specified inventory, command execution, formatting,
  and other deterministic supporting work to Luna. It remains responsible for
  integrating and interpreting the result.
- Astra is reserved for an independent review of the highest-risk concurrency,
  ownership, numerical-equivalence, or scheduler decisions; it is not the
  default implementation tier.

Give every delegated agent a bounded deliverable, the relevant invariants,
and the required evidence. The coordinating agent must inspect the returned
diff and evidence before accepting it.

If the required higher-capability model or routed-subagent facility is not
available, do not implement or approve the high-risk semantic change. Record
the missing review as a blocker; low-risk evidence collection may continue.

## Delegation packets

Default to a compact, task-specific packet when delegating. Do not inject the
entire accumulated conversation or repository context unless that history is
itself necessary to resolve the delegated question.

Each packet must contain only:

1. the concrete deliverable or review question;
2. the relevant invariants and supported/unsupported scope;
3. exact file paths and, where useful, the specific diff or function region;
4. passing/failing test, trace, benchmark, or reproduction evidence; and
5. the decision the parent needs back (for example: blocking findings only,
   approval of stated invariants, or a bounded patch in named files).

Start subagents without inherited context by default. Include broader context
only when omitting it would make the answer materially unreliable, and state
why in the delegation request. Review agents must distinguish verified facts,
recommendations, blockers, and remaining unproven gates.

## Required review gates

For high-risk changes, require:

1. a written invariant/ownership summary before implementation;
2. a stronger-model review of the final diff;
3. targeted regression tests plus performance evidence;
4. no scalability claim from a microbenchmark alone.
