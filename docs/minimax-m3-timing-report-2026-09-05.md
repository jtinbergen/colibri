# MiniMax-M3 timing report — 2026-09-05

## Status after the later measurements

This document records the early probes. The later
[comprehensive report](minimax-m3-comprehensive-report-2026-09-05.md)
adds completed 32-token PIPE=1 runs, direct read-latency distributions and
an E:-only run. The historical stalls below remain observations, but they
do not describe the outcome of those later PIPE trials. Completion and
measured overlap are not proof of a speedup over PIPE=0 or a diagnosed fix
for the earlier stalls.

The [multicore analysis and roadmap](minimax-m3-multicore-analysis-2026-09-05.md)
separate these desktop measurements from CPU scheduling findings and
unmeasured Naples/Rome scaling hypotheses.

## Scope

Interactive `hi` probes against the local 241 GB MiniMax-M3 int4 container
using the Colibri v1.10.1 build on Windows 11, 10 physical CPU cores, 32 GB
RAM and an NVIDIA GTX 1080 (sm_61). Runtime instrumentation was enabled with
`PROF=1` and `DISK_SPLIT=1`; the chat UI also reports per-turn wall time,
decode tok/s, expert-hit rate and RSS.

The model was reset between clean probes with `:reset`. Existing expert usage
history was intentionally retained so the normal adaptive warm-cache behavior
could be observed. `hi` is not a fixed-length benchmark: MiniMax sometimes
stops after 2–16 output tokens, so these numbers are directional rather than a
stable throughput benchmark.

## Results

| variant | storage | CUDA | RAM state | startup / resident | cache tier | result | interpretation |
|---|---|---:|---|---:|---|---|---|
| CPU-only baseline | C: only | off (`COLI_CUDA=0`, `--gpu none`) | Chrome open | 4.60 s / 6.1 GB | cap 3/layer; ~0.1 GB pinned | 0.35 tok/s, 24% hit, RSS 13.8 GB, 9 tok, 26 s | valid smoke result; disk misses remain significant |
| CUDA baseline | C: only | on | Chrome open | 5.20 s / 6.1 GB | 45 VRAM experts / 1.43 GB; cap 3/layer | 0.34 tok/s, 32% hit, RSS 15.2 GB, 9 tok, 26 s | CUDA tier did not beat CPU-only on this small SM61 GPU |
| CUDA baseline, repeat | C: only | on | Chrome open | — | usage history retained | 0.37 tok/s, 34% hit, RSS 15.3 GB, 9 tok, 24 s | normal run-to-run variation |
| CUDA + mirror | C: primary + E: mirror | on | Chrome closed; 24.46 GB free | 4.31 s / 6.1 GB | 63 VRAM experts / 2.01 GB; cap 6/layer; ~2.0 GB warm | 0.37 tok/s, 46% hit, RSS 21.0 GB, 16 tok, 44 s | extra RAM is used and hit-rate improves, but throughput is unchanged within noise |
| CUDA + mirror, repeat | C: primary + E: mirror | on | Chrome closed | — | adaptive history retained | 0.14 tok/s, 35% hit, RSS 21.1 GB, 2 tok, 14 s | not comparable: only two output tokens |

## Planner observations

With Chrome closed, the OS reported 24.46 GB physically free. The engine then
reported `RAM_GB=26.2–26.5`, raised the cache from 3 to 6 slots/layer, and
increased the CUDA hot tier from 45 to 63 experts (1.43 → 2.01 GB). This proves
that Colibri used the additional memory.

The planner still classified the run as `disk expert misses`, with an
approximately 8% projected hit-rate before adaptive history. No trusted SSD
probe was available (`ssd_probe_state=absent`), so no measured disk weight was
selected automatically. The E: copy contains all 59 shards, therefore
`COLI_MODEL_MIRROR=E:\minimax_m3_i4` is the correct two-copy configuration.

## Deterministic profiler cross-check

To remove the large variation of interactive `hi`, I repeated the test as a
one-shot run with `--temp 0 --ngen 24 --no-think`, `PROF=1`, `DISK_SPLIT=1`,
`PIPE=0`, and no existing chat state. The model stopped after 7--9 tokens, so
the decode figures and phase shares remain directional. Different output
lengths and retained adaptive history prevent a controlled fixed-token A/B;
the profiles do establish that mirror routing and the expert tier were active.

| variant | auto-tier | decode | expert hit | profiler result |
|---|---:|---:|---:|---|
| CPU, C: only | no | 0.73 tok/s, 9 tok | 59.6% | I/O 53% of decode window; routed GPU 0.000 s |
| CUDA, C: only | yes | 0.75 tok/s, 9 tok | 62.0% | I/O 51%; GPU critical 0.444 s; 96 VRAM experts / 3.06 GB |
| CUDA, C: + E: mirror | yes | 0.65 tok/s, 7 tok | 65.6% | I/O 52%; GPU critical 0.358 s; 102 VRAM experts / 3.25 GB |

The mirror was unequivocally active: Colibri measured C: at 4.81 GB/s and E:
at 0.96 GB/s, then sent 17% of expert bytes to E:. This makes the mirror
slower for this workload: the slower E: path is added to the critical read
set, while `PIPE=0` prevents full read/compute overlap. The raw mirror run
reported 14.1 s expert-disk service, 5.6 s felt wait, and 52% of the decode
window in expert I/O.

## Direct E: read benchmark

I also ran a read-only, uncached benchmark against the largest E: model shard
(`out-00013.safetensors`, 4.53 GB). The test used aligned Windows
`FILE_FLAG_NO_BUFFERING` reads, so these numbers are closer to storage service
than to a page-cache benchmark:

| E: test | result |
|---|---:|
| sequential, one reader, 4 MiB blocks | 2,496 MB/s |
| random, one reader, 1 MiB blocks | 776 MB/s |
| random, four concurrent readers, 1 MiB blocks | 978 MB/s aggregate |

The four-reader result matches Colibri's 0.96 GB/s probe. PowerShell also
confirmed that E: is a volume on physical Disk 0, a 512 GB Samsung
MZAL4512HBLU NVMe; C: is on a separate 1 TB Samsung 980 PRO. The mirror is
therefore a real second physical device, but it is substantially slower for
the small random-read pattern. The tested distribution sent approximately
17% of expert bytes to E: and did not demonstrate a throughput improvement.
This does not rule out a different policy that schedules reads earlier or
accounts for the current queues on both drives.

## Findings

The extra RAM improved residency and expert-hit rate, but did not translate
into a clear tok/s increase. The deterministic cross-check makes the reason
more concrete: the CUDA tier changes 0.73 to 0.75 tok/s, while expert I/O still
accounts for about half of the decode window. The GTX 1080 is participating
when `--auto-tier` is present, but its routed-GPU critical time is only 0.444 s
versus 6.086 s of expert-read wait. That routed-expert metric does not measure
all GPU participation or establish where every dense tensor executes. It
does show that reducing expert-read wait is a relevant optimization target
in this run.

The earlier `--gpu auto` runs without `--auto-tier` were also misleading: they
showed CUDA enabled but placed zero routed experts in VRAM. In that mode the
GPU only held resident dense tensors, and the profile correctly reported
`routed GPU critical 0.000 s`.

The `PIPE=1` mirror probe stalled after prefill with no CPU or disk activity and
was interrupted. A fixed `--ngen 32` probe also stalled before decode output.
Those runs are excluded from the performance table. A future comprehensive
benchmark should use a fixed, non-interactive replay/serve harness with a fixed
token budget and collect the profiler frame there; short interactive `hi`
responses are too variable for precise phase percentages.

## Reproduction settings

```powershell
$env:COLI_MODEL_MIRROR = 'E:\minimax_m3_i4'
$env:COLI_CUDA = '1'       # use '0' and --gpu none for CPU-only
$env:PROF = '1'
$env:DISK_SPLIT = '1'
$env:PIPE = '0'            # PIPE=1 stalled in this run
$env:KVSAVE = '0'
python C:\path\to\colibri-dev\c\coli chat `
  --model C:\Users\jaapj\.lmstudio\models\minimax_m3_i4 `
  --no-attach --temp 0
```
