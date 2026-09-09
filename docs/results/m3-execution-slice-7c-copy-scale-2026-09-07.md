# MiniMax-M3 Step 7c — full copy-table scale evidence

## Result

The bounded copy-table split is implemented and passes a scale fixture for
the configured M3 shape: 60 layers × 128 local experts × 2 replicas =
15,360 copy rows. Resource/profile arrays remain bounded at 128; only the
caller-owned copy table uses the separate `M3_SHD_MAX_COPIES=16384` bound.

The fixture also probes the first, middle, and last expert identities 1,000
times each. Every probe returns both configured replicas.

```text
copy scale: PASS rows=15360 resources=6 profiles=6 parse_ms=43.00
open_ms=65.00 probe_total_ms=40.00 probe_calls=3000 candidates=6000
```

## Scope limits

This is capacity, parsing, opening, and candidate-lookup evidence. It does
not establish full-model route/lease/read equality, mixed-residency behavior,
direct-striped or mmap physical traffic, shared-controller calibration, or
performance promotion. Those remain separate Step-7 evidence requirements.
