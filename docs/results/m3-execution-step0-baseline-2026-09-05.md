# M3 execution-DAG step 0 baseline

Date: 5 September 2026.

This is the baseline record for step 0 of
[the execution plan](../minimax-m3-execution-plan-2026-09-05.md). It changes
no runtime scheduling, quantization, routing, context limit, sampling,
residency policy, or engine code.

## Scope

- Repository: local `colibri-dev`, branch `dev`, HEAD `0efa86f`.
- This is not a clean-HEAD baseline. Existing local changes in `c/colibri.c`,
  `c/iobench.c`, and `c/tools/convert_fp8_to_int4.py`, plus existing untracked
  M3 fixtures and reports, are part of the observed worktree. They were not
  edited or staged by this step.
- The `git diff` content observed after this step has Git blob hash
  `efcf3ec5ef0bd91d955cbe364fb411f89dc8befa`. This identifies the complete
  then-current diff, including uncommitted baseline documentation; it is not
  a committed revision and must be recomputed before reproducing later.
- Tested binary: `c/colibri.exe`, SHA-256
  `2E4D775E0A322D0E43DA32AE6430EF9D54FA8E2759583F6F95B1F93352E9AA53`.
- Test host: Windows 10 Pro 64-bit, Intel Core i5-12600K (10 physical,
  16 logical cores), 32 GiB RAM. The engine selected 10 physical-core OpenMP
  threads. No Naples or Rome machine was available.
- CPU-only correctness scope: the tiny fixture run reported an AVX2 IDOT
  build; its test driver sets `IDOT=0`. It did not enable a CUDA, Metal, or
  Vulkan expert backend.

## Preserved fixture fingerprints

| File | SHA-256 |
|---|---|
| `c/m3tiny/config.json` | `BEDD284C891AF6B6BDC30A37A9DAD5327EE8D3DAB86953BD921AD8DC8AEBCE10` |
| `c/m3tiny/model-00001-of-00001.safetensors` | `0A783A8376E54EA4A7AB3023F53D949BD45A2B646F78F242E0F028E0A6232576` |
| `c/m3tiny_i8/config.json` | `3D4D1F4BC0024ECE42988E6E72FBA1732AA6F6C944048BD996D39C7D409B1D9A` |
| `c/m3tiny_i8/out-00000.safetensors` | `088615D5DC3815B6E51066E2A3C584E1D624BE5E6E1E0852ED49E6007C7C663F` |
| `c/ref_m3.json` | `2FEE36B8E6DD379D6C63774695DF2ACEF26BC764EB440C7B65525E0A4BD17EB1` |
| `c/oracle_logits.npy` | `E80EB5BBD6BBC2FE81F24C089B4A777EA9DD198718191595A6D305D73BC97DEA` |
| `c/tests/test_pipe_block.exe` | `CE3DEE77EB64ABBC25864966F2014FE50FEB43195085B61A5F48267505569A3A` |

`m3-tiny-check` was intentionally not invoked because its generation target
deletes and recreates the shared `m3tiny` and `m3tiny_i8` directories. The
existing driver was invoked directly against the fingerprinted fixture.

## New grouped-int4 reference

[`m3_grouped_int4_reference.json`](../../c/tests/fixtures/m3_grouped_int4_reference.json)
contains two hand-derived grouped-int4 rows, group size 2, and a non-zero
unused high nibble in each final packed byte. The independent reference script
verifies per-group scaling and ignores padding after input dimension 5. It is
a fixture correctness check, not an engine-kernel or full-model test.

| File | SHA-256 |
|---|---|
| `c/tests/fixtures/m3_grouped_int4_reference.json` | `E777FADF79DFEBBFA3FD332B2C285FE8B75B5A4D97AD950768FF54056CBD23A2` |
| `c/tests/test_m3_grouped_int4_reference.py` | `DC7BD5247E3540872CEB5CF66C918A47228392F51349CA6A2573E78E0EF526FA` |

## Commands and results

All commands below ran from `colibri-dev/c` and returned exit code 0.

```powershell
& 'C:\Python313\python.exe' '.\tests\test_m3_tiny.py' `
  --binary '.\colibri.exe' --snap '.\m3tiny_i8' --ref '.\ref_m3.json'
& '.\tests\test_pipe_block.exe'
& 'C:\Python313\python.exe' '.\tests\test_m3_grouped_int4_reference.py'
```

Results:

- M3 teacher forcing: prefill `24/24`; incremental decode `20/20`.
- PIPE regression: `test_pipe_block: ok`.
- Grouped-int4 reference: two output rows checked against `-0.125` and
  `-17.75`.

The grouped-int4 reference script also passed `python -m py_compile`.

Fixed-token replay used the existing M3 oracle. Both commands ran from `c`:

```powershell
$env:SNAP='.\m3tiny_i8'; $env:REF='.\ref_m3.json'; $env:REPLAY='1'
$env:PROF='1'; $env:IDOT='0'; & '.\colibri.exe' 8
```

| Run | Decode | Expert hit | p50 / p90 / p99 forward latency | Attention | Expert compute |
|---|---:|---:|---:|---:|---:|
| A | 20 tokens, 783.80 tok/s | 100.0% | 1.3 / 1.4 / 1.4 ms | 0.012 s | 0.006 s |
| B | 20 tokens, 790.91 tok/s | 100.0% | 1.2 / 1.3 / 1.4 ms | 0.011 s | 0.006 s |

Both runs used CPU backend, `PIPE=1`, `DIRECT=0`, `IDOT=0`, `CTX=4096`,
auto `RAM_GB=19.2`, cache cap 8 per layer, no pinned experts and no disk
reads. They reproduced the same fixed 20-step route. The tiny oracle contains
24 full token IDs and therefore can replay only 20 decode steps; it is a
mechanism smoke test, not the 32/256-token performance baseline required for
the full checkpoint. Its 100% hit rate also means it does not exercise PIPE
I/O overlap or miss scheduling.

## Gate 0 status

| Gate | Status | Evidence / limitation |
|---|---|---|
| C | PASS, fixture scope | Existing tiny oracle and PIPE regression passed; an independent grouped-int4 fixture now exists and is checked. The grouped fixture does not yet call the C grouped-int4 kernel, so it is not evidence for the future reentrant kernel. |
| M | PASS, tiny replay scope | Build, fixture fingerprints, commands, machine configuration and two matching fixed-token replays are recorded. The replay exercises no disk miss. |
| P | NOT_RUN | No 32- or 256-token full-model baseline has been run. The known tiny replay cannot supply enough decode steps; no Naples/Rome machine was available. |

## Completion and remaining baseline work

**Step 0 is done for its required C/M dependency scope.** Step 1 may proceed:
it requires Gate 0 C/M, not the full-model performance gate. Gate 0 P remains
open. A subsequent full-model baseline still needs a dedicated fixed-token
replay fixture with at least 257 valid full-model token IDs and a documented
resident-memory budget. Do not report this tiny result as a full-M3, storage,
Naples or Rome performance baseline.
