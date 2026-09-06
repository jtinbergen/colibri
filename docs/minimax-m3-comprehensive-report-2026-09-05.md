# Comprehensive MiniMax-M3 timing- en drive-rapport — 2026-09-05

## Samenvatting

De metingen laten drie verschillende effecten zien:

1. Extra RAM wordt door Colibri benut en verhoogt de residentie/hitrate, maar
   niet genoeg om de vele expert-loads per token te elimineren.
2. CUDA helpt alleen beperkt. Zonder `--auto-tier` stonden er in de one-shot
   run geen routed experts in VRAM. Met `--auto-tier` waren er 96–102 experts
   actief in ongeveer 3,1–3,3 GB VRAM, maar expert-I/O bleef de dominante fase.
3. De C:/E:-mirror is in de gerapporteerde runs een regressie. E: is gezond, maar voor
   Colibri's random-read patroon ongeveer vijf keer trager dan C:. Colibri
   stuurde toch 17% van de expertbytes naar E:.

`PIPE=1` is in de gerapporteerde 32-tokenproeven door de stabiliteits- en
overlapgate gekomen. De profiler meet gelijktijdige PIPE-I/O en
expert-compute. Dit bewijst overlap binnen deze proeven, geen algemene
stabiliteit of versnelling tegenover PIPE=0. De workload blijft I/O-bound;
de geteste mirrorverdeling levert geen throughputwinst op.

De [multicoreanalyse en gezamenlijke roadmap](minimax-m3-multicore-analysis-2026-09-05.md)
verbinden deze metingen met de seriële MSA-indexer en expertuitvoering.
De compute-bevindingen zijn broncodeanalyse; Naples- en Rome-schaalcurves
zijn nog niet gemeten.

## Testomgeving

- Colibri v1.10.1, MiniMax-M3 int4-container, circa 241 GB op disk.
- 59 safetensors-shards, 60 lagen, 128 experts, top-4 routing.
- Intel Core i5-12600K, Windows 11, 32 GB RAM.
- NVIDIA GeForce GTX 1080, 8 GB VRAM, compute capability sm_61.
- C: = fysieke Disk 1, Samsung SSD 980 PRO 1 TB.
- E: = fysieke Disk 0, Samsung MZAL4512HBLU-00BL2, 512 GB NVMe.
- Instrumentatie: `PROF=1`, `DISK_SPLIT=1`, `DIRECT=1`, `KVSAVE=0`.
- De adaptive expert history bleef tussen runs behouden; daardoor zijn de
  hitrates realistisch voor een warme sessie, maar niet volledig onafhankelijk.

## 1. Interactieve `hi`-probes

Deze runs kwamen uit de interactieve chat. `hi` is geen stabiele throughput-
benchmark: de generatie stopte afhankelijk van sampling/EOS na 2–16 tokens.
Gebruik deze tabel daarom als smoke-test en voor gedrag, niet als strikte
ranglijst.

| Variant | Opslag | CUDA/cache | RAM-status | Resultaat | Status |
|---|---|---|---|---:|---|
| CPU-only | C: | CUDA uit, cap 3/layer | Chrome open | 0,35 tok/s; 24% hit; 9 tok; 26 s; RSS 13,8 GB | geldig smoke-resultaat |
| CUDA-baseline | C: | 45 VRAM-experts / 1,43 GB; cap 3/layer | Chrome open | 0,34 tok/s; 32% hit; 9 tok; 26 s; RSS 15,2 GB | geen duidelijke winst |
| CUDA-baseline, herhaling | C: | adaptive history | Chrome open | 0,37 tok/s; 34% hit; 9 tok; 24 s; RSS 15,3 GB | normale variatie |
| CUDA + mirror | C: + E: | 63 VRAM-experts / 2,01 GB; cap 6/layer | Chrome gesloten; 24,46 GB vrij | 0,37 tok/s; 46% hit; 16 tok; 44 s; RSS 21,0 GB | RAM/hitrate beter, snelheid niet |
| CUDA + mirror, herhaling | C: + E: | adaptive history | Chrome gesloten | 0,14 tok/s; 35% hit; 2 tok; 14 s | niet vergelijkbaar |

Chrome sluiten had dus een meetbaar effect op de resource-planning: de cache
ging van ongeveer cap 3 naar cap 6 per laag en de hot CUDA-tier groeide van
45 naar 63 experts. De interactieve tok/s steeg echter niet betrouwbaar,
omdat de outputlengte en disk-miss-patronen sterk varieerden.

## 2. Reproduceerbare one-shot profiler-runs

Voor een betere vergelijking is daarna dezelfde prompt (`hi`) one-shot gedraaid
met `--temp 0 --ngen 24 --no-think`. De engine stopte alsnog na 7–9 tokens,
waardoor de decodeprofielen richtinggevend blijven. De aantallen tokens en
adaptive cachegeschiedenis verschillen; dit is geen gecontroleerde
vaste-token-A/B. Alle runs gebruikten `PIPE=0`.

| Variant | Configuratie | Decode | Hitrate | Expert-I/O | GPU routed | Resident tier |
|---|---|---:|---:|---:|---:|---:|
| CPU | C:, `--gpu none` | 0,73 tok/s; 9 tok | 59,6% | 53% van decode-window; 6,53 s wait | 0,000 s | 35 RAM-experts / 1,1 GB |
| CUDA zonder auto-tier | C:, `--gpu auto` | 0,77 tok/s; 9 tok | 58,9% | 56%; 6,54 s wait | 0,000 s | 0 routed VRAM-experts; 7,26 GB resident dense |
| CUDA met auto-tier | C:, `--gpu auto --auto-tier` | 0,75 tok/s; 9 tok | 62,0% | 51%; 6,09 s wait | 0,444 s | 96 experts / 3,06 GB VRAM |
| CUDA + mirror met auto-tier | C: + E: | 0,65 tok/s; 7 tok | 65,6% | 52%; 5,62 s wait | 0,358 s | 102 experts / 3,25 GB VRAM |

De single-disk CPU- en CUDA-met-auto-tier-runs liggen met 0,73 versus 0,75
tok/s praktisch bij elkaar. De GPU-tier is dus daadwerkelijk actief, maar de
GPU is niet de dominante tijdcomponent: routed GPU-critical time is minder dan
een halve seconde, tegenover ruim vijf à zes seconden expert-read-wachttijd.

Er zijn 228 routed-expertselecties per token (57 actieve MoE-lagen × top-4).
Dat zijn niet noodzakelijk 228 diskreads: residentie kan een selectie zonder
disk-I/O afhandelen en een miss kan meerdere tensorreads vereisen. Gebruik
hitrate, werkelijk gelezen bytes en gevoelde wachttijd om het I/O-effect van
extra RAM vast te stellen.

## 3. Gedrag van mirror en `PIPE`

De mirror was aantoonbaar actief:

- Colibri vond alle 59/59 shards op `E:\minimax_m3_i4`.
- Probe: C: ongeveer 4,81 GB/s, E: ongeveer 0,96 GB/s.
- Automatische verdeling: 83% C: / 17% E:.
- Profiler: in de mirror-run 55,68 GB C: en 11,21 GB E:; opnieuw 17% vanaf E:.

Dit is geen RAID-0-achtige versnelling. Een laag kan pas verder wanneer alle
gevraagde experts beschikbaar zijn. Met `PIPE=0` staat de compute-thread voor
deze loads te wachten; de tragere E:-reads worden daardoor onderdeel van het
kritieke pad. De hogere hitrate maskeert dus niet dat de resterende misses
langs een veel trager pad lopen.

Een historische `PIPE=1`-mirrorprobe is gestart om reads met compute te overlappen, maar
bleef na prefill hangen terwijl CPU- en diskactiviteit stilvielen. Die run is
afgebroken en niet als performance-resultaat gebruikt. Ook een interactieve
fixed-`--ngen 32` probe bleef vóór decode hangen; die is eveneens uitgesloten.

## 4. Directe read-benchmark van E:

Er is niets naar E: geschreven. Tegen de grootste model-shard
`out-00013.safetensors` (4,53 GB) zijn aligned Windows
`FILE_FLAG_NO_BUFFERING` reads uitgevoerd:

| Test | Resultaat |
|---|---:|
| Sequentieel, één reader, 4 MiB blokken | 2.496 MB/s |
| Random, één reader, 1 MiB blokken | 776 MB/s |
| Random, vier gelijktijdige readers, 1 MiB blokken | 978 MB/s aggregate |

De vier-readerwaarde komt vrijwel exact overeen met Colibri's probe van
0,96 GB/s. De lage Task Manager-percentages zijn daarmee niet vreemd: de
expert-reads komen in korte bursts, daarna wacht de engine op routing/compute.
Een snapshot tijdens een wacht- of tussenfase kan dus 0–1% diskgebruik tonen,
ondanks dat de disk de limiterende fase levert.

Windows rapporteerde voor E: `HealthStatus=Healthy` en
`OperationalStatus=OK`; er is in deze controle geen aanwijzing voor een
defecte drive. Het exacte model wordt doorgaans als Samsung PM9B1 512 GB
geclassificeerd; dit is een OEM/mid-range NVMe in plaats van een 980 PRO.

## 5. Conclusie

Meer RAM verhoogde de residentie; de combinatie met CUDA-tier en mirroring
leverde geen overtuigende throughputwinst op in deze proeven. De profielen
tonen dat expert-I/O een groot deel van de decodewindow blijft innemen.
De gerapporteerde resultaten zijn:

1. C: alleen, CUDA + auto-tier: circa 0,75 tok/s in de eerdere one-shot test.
2. C: alleen, CPU + PIPE=1: 0,58-0,62 tok/s in de langere gate-run.
3. C: + E: mirror, CPU + PIPE=1: 0,49-0,50 tok/s in de langere gate-run.

De korte CUDA-run en de langere CPU-runs vormen geen onderlinge ranglijst:
outputlengte, configuratie en residentie verschillen. De geteste mirrorroute
is wel een ongunstig resultaat dat nader onderzoek naar planning rechtvaardigt.

De volgende experimenten moeten afzonderlijk onderzoeken of eerdere reads,
betere bronkeuze, meer residentie of het uitvoeren van reeds beschikbare
experts de gevoelde wachttijd verlagen. E: als vroege prefetch-tier is daarbij
een hypothese; capaciteit als tweede kopie is al aangetoond. Alleen een
hogere overlapfractie bewijst nog geen lagere tokenlatency.

## 6. Addendum: PIPE=1 gate-resultaat

De historische hang is in de nieuwe gecontroleerde runs niet gereproduceerd.
Alle volgende varianten bereikten 32 decode-tokens zonder hang. De profiler
meet hier de echte wall-clock intersection tussen minstens één PIPE-worker in
`expert_load()` en minstens één compute-thread in `expert_ffn()`.

| Variant | Decode | I/O-active | Compute-active | Concurrent | Mirror |
|---|---:|---:|---:|---:|---:|
| C:, PIPE=1, block-wait | 32 tok; 0,58 tok/s | 37,702 s | 10,164 s | 2,819 s; 27,7% van compute | nee |
| C:, PIPE=1, spin-wait | 32 tok; 0,62 tok/s | 36,687 s | 8,703 s | 2,944 s; 33,8% van compute | nee |
| C:+E:, PIPE=1, block-wait | 32 tok; 0,49 tok/s | 40,961 s | 15,343 s | 4,289 s; 28,0% van compute | 18% bytes vanaf E: |
| C:+E:, PIPE=1 + `PILOT_REAL=1` | 32 tok; 0,50 tok/s | 45,958 s | 10,909 s | 4,334 s; 39,7% van compute | 17% bytes vanaf E: |

Conclusie voor de gate: PASS voor voltooiing zonder hang en aantoonbare
I/O/compute-overlap in deze 32-tokenproeven. De historische hang is daarmee
niet verklaard of bewezen opgelost. De overlapfractie meet gelijktijdige
activiteit, niet hoeveel seconden tegenover PIPE=0 zijn bespaard. Voor die
conclusie ontbreekt een gelijkwaardige A/B. De mirror-runs blijven hier
trager; de afzonderlijke read-metingen ondersteunen E: als een ongunstig
pad voor urgente reads onder de geteste belasting.

## 7. Latencyvergelijking C: versus E: en E:-only

Voor een directe latencyvergelijking is `iobench` uitgebreid met timing per
read. Beide drives gebruikten exact dezelfde 1024 pseudo-random offsets in
`out-00013.safetensors`, blokken van 1 MiB en `O_DIRECT`/Windows
`FILE_FLAG_NO_BUFFERING`. QD1 betekent één reader; QD4 betekent vier gelijktijdige
readers. De gerapporteerde latency is de wall-clockduur van de individuele
read-syscall inclusief de storage- en queueingkosten.

| Drive / queue depth | Throughput | Min | Gemiddeld | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| C:, QD1 | 3,41 GB/s | 0,276 ms | 0,307 ms | 0,301 ms | 0,359 ms | 0,413 ms | 0,453 ms |
| E:, QD1 | 0,81 GB/s | 0,453 ms | 1,286 ms | 1,119 ms | 1,781 ms | 1,901 ms | 11,177 ms |
| C:, QD4 | 6,57 GB/s | 0,305 ms | 0,635 ms | 0,633 ms | 0,849 ms | 0,962 ms | 1,198 ms |
| E:, QD4 | 1,00 GB/s | 0,706 ms | 4,173 ms | 4,154 ms | 6,205 ms | 7,541 ms | 16,657 ms |

Dit ondersteunt planning op verwachte aankomsttijd: E: heeft niet alleen lagere bandbreedte,
maar ook een veel slechtere latencycurve zodra reads concurreren. Bij QD4 is
E: ongeveer 6,6x trager op p50, 7,3x op p95 en 7,8x op p99 dan C:. E: is
onder deze belasting een ongunstige bron voor urgente misses wanneer C:
beschikbaar is. Een opportunistische prefetch-route met voldoende verwachte
slack is te onderzoeken. Of die route werkelijk voordeel biedt, hangt ook
af van queueing op C:, voorspelfouten en de kosten van RAM-verdringing.

Omdat de Colibri-documentatie ook random reads van ongeveer 19 MiB beschrijft,
is hetzelfde experiment herhaald met 19 MiB-blokken en 64 reads (ongeveer
1,3 GB totaal). Dit is dichter bij de engine-load dan de 1 MiB-microtest:

| Drive / queue depth | Throughput | Min | Gemiddeld | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| C:, QD1 | 5,54 GB/s | 3,428 ms | 3,587 ms | 3,529 ms | 3,712 ms | 3,787 ms | 5,663 ms |
| E:, QD1 | 0,83 GB/s | 6,650 ms | 24,098 ms | 27,633 ms | 31,485 ms | 42,245 ms | 48,580 ms |
| C:, QD4 | 6,44 GB/s | 7,010 ms | 12,004 ms | 11,850 ms | 12,879 ms | 18,275 ms | 20,782 ms |
| E:, QD4 | 0,91 GB/s | 32,299 ms | 86,969 ms | 86,379 ms | 117,360 ms | 132,982 ms | 134,493 ms |

In deze workload-nabije test ligt E:'s QD1-minimum (6,650 ms) al boven
C:'s QD1-gemiddelde (3,587 ms). Bij QD4 is E:'s minimum (32,299 ms) bijna
2,7 keer C:'s gemiddelde (12,004 ms), terwijl E:'s p99 132,982 ms bedraagt.
Dit ondersteunt een voorkeur voor C: bij urgente reads onder deze
testcondities. Het bewijst geen universele bronkeuze: de planner moet
verwachte aankomst inclusief actuele wachtrijen vergelijken. De 64 reads
per 19 MiB-test geven bovendien slechts een beperkte steekproef voor p99.
Een blok van 19 MiB is geen meting van een volledige M3-expertload; de
daadwerkelijke tensorreadgroottes en quantisatie moeten in de planner terugkomen.

### E:-only end-to-end

Een volledige Colibri-run vanaf `E:\minimax_m3_i4` (geen C:-mirror), met
`--gpu none --cap 2 --ngen 32`, `PIPE=1`, vier workers, `DIRECT=1` en dezelfde
vaste prompt, gaf:

| Metric | E:-only |
|---|---:|
| Model startup | 6,41 s |
| Prefill, 31 tokens | 112,57 s |
| Decode | 32 tokens in 256,09 s; 0,12 tok/s |
| Decode forward p50 / p99 | 8.134 / 9.083 ms |
| Expertbytes | 191,740 GB; 0,75 GB/s over de run |
| Expert read service / gevoelde wait | 707,8 s / 218,1 s |
| Disk-busy | 224,0 s; 87% van de decodewindow |
| PIPE concurrent I/O + compute | 5,983 s; 27,3% van compute-actieve tijd |

De E:-only-run voltooide met PIPE=1 en rapporteerde 0,12 tok/s, tegenover
0,58-0,62 tok/s in de eerdere C:-only PIPE-runs. Die verhouding is ongeveer
een factor vijf, maar is zonder identieke cache- en runcondities geen
geïsoleerde drive-speedup. Binnen de E:-only-run is het I/O-aandeel wel
direct gemeten: circa 85% gevoelde expert-I/O-wachttijd. Dit bevestigt
functionele inzet als opslagbron en motiveert een experiment met vroege
prefetch; de prestatievoordelen van die prefetch zijn nog niet aangetoond.

## Reproductie

```powershell
# Eerdere korte C:-only-probe met routed experts in VRAM; geen vaste-tokenbenchmark
$env:COLI_CUDA = '1'
Remove-Item Env:COLI_MODEL_MIRROR -ErrorAction SilentlyContinue
$env:PROF = '1'
$env:DISK_SPLIT = '1'
$env:PIPE = '0'
$env:KVSAVE = '0'

& 'C:\Python313\python.exe' '.\coli' run `
  --model 'C:\Users\jaapj\.lmstudio\models\minimax_m3_i4' `
  --gpu auto --auto-tier --ngen 24 --temp 0 --no-think 'hi'
```

Voor de mirror voeg je toe:

```powershell
$env:COLI_MODEL_MIRROR = 'E:\minimax_m3_i4'
```
