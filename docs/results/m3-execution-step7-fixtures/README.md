# Step 7 deterministic fixture captures

These fixtures exercise the bounded, observer-only planner with a fake
monotone clock. They do not claim a hardware result. Every service rate is
labelled in bytes/second and every topology carries explicit drive,
controller, and optional upstream budgets.

The executable reproducer is:

```text
make -C c test-shadow-m3 test-shadow-observer-m3 \
  test-shadow-store-m3 test-shadow-registry-m3
```

The expected result is the single `PASS` line from each executable. The
individual cases and their hand-checkable assertions are recorded below; the
test process itself is the authoritative calculation and uses integer
nanosecond arithmetic.

Cases covered:

- shared bandwidth: four drives at 500 MiB/s through one 500 MiB/s controller;
- critical existing read: defer beats a contended candidate;
- independent replica: a free controller beats defer;
- latency asymmetry: earlier startup wins when score ties;
- six/seven-drive shape across three controllers and a shared upstream;
- free-drive/full-controller admission rejection and bounded defer;
- low-latency path under contention versus an independent free path;
- explicit best-effort deadline miss and stable resource-ID tie-break;
- optional serial conversion stage: 0.2 s I/O plus 0.1 s conversion yields
  deterministic 0.3/0.4 s weight-ready times; missing conversion classes stay
  `UNKNOWN`;
- optional `queue_age_ns` policy: a fake-clock regression gives an aged request
  priority over newer demand, with enqueue-time/request-ID tie-breaking;
- missing profile/replica, hard queue and byte bounds, failure cleanup;
- deterministic tie-breaking and byte-identical decision logs.

The all-resident performance comparison and real six/seven-drive contention
measurement remain `NOT_RUN` until the machine and calibration profile exist.
