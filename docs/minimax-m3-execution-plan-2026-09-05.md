# M3 execution-DAG: uitvoeringsplan met briefings en control gates

Datum: 5 september 2026. Status: stappen 0 en 1 afgerond binnen hun vastgelegde tiny-scope; stap 2 afgerond (MSA-scan, CPU-scope).

## Opdracht en startpunt

Verlaag M3 single-sequence decode-latency door MSA, expertcompute, I/O en
residentie op hun afhankelijkheden en beschikbare resources te plannen.
Behoud de modeluitkomst en meet iedere verbetering afzonderlijk. De eerste
implementatie richt zich op de lokale CPU-engine, `S=1`, grouped-int4 met
group size 64, één model en één actieve forward per proces.

Dit document is een overdraagbare werkinstructie. Voer per opdracht één
genummerde stap uit. Lees vooraf de algemene regels en de briefing van die
stap. Lever daarna het gateverslag op. Een volgende stap mag pas beginnen
als zijn vereiste gates zijn geslaagd. Een technische gate is geen verzoek
om opnieuw toestemming aan de gebruiker te vragen voor al opgedragen werk.

De huidige opdracht is het schrijven van dit plan. De beschreven flags,
tracevelden, tests en interfaces zijn voorstellen totdat een stap ze
implementeert. Alleen de expliciet als bestaand genoemde onderdelen zijn
nu beschikbaar. Er zijn met dit document geen engines of benchmarks gestart.

Lees eerst:

- [Multicoreanalyse en aangescherpte roadmap](minimax-m3-multicore-analysis-2026-09-05.md).
- [Meetrapport inclusief PIPE-gate en C:/E:-latency](minimax-m3-comprehensive-report-2026-09-05.md).
- [Eerdere timingproeven en hun beperkingen](minimax-m3-timing-report-2026-09-05.md).

De onderzochte uitgangsversie is lokaal commit `0efa86f` plus bestaande
werkboomwijzigingen. Leg bij uitvoering de dan werkelijke versie vast.
Kopieer historische regelnummers niet blind: zoek onderstaande functies.

## Codekaart en bekende valkuilen

| Onderdeel | Bestaande bron | Wat je vóór een wijziging moet begrijpen |
|---|---|---|
| MSA en GQA | [`colibri.c`](../c/colibri.c): `attention_gqa`, `msa_idx_dot` | Indexer paralleliseert queryrijen alleen bij `S>4`; score/selection ontbreekt in de gedetailleerde attentiontimers |
| Experts | `moe`, `expert_gate_up`, `expert_ffn`, `act_glu` | Gewone CPU-experts worden achtereenvolgens uitgevoerd; shared expert volgt daarna |
| Kernels | [`quant.h`](../c/quant.h), `matmul_i4_grouped_pair` in `colibri.c` | Rijgewijs parallelisme; volgorde binnen een dotproduct en formatdispatch behouden |
| Gedeelde scratch | `g_pq`, `quant_scratch`, thread-local buffers | `expert_ffn` is niet bewezen reentrant; geen concurrente aanroepen zonder audit |
| PIPE | `PipePool`, `pipe_dispatch`, `pipe_ready`, `pipe_wait` | Eén batchgeneratie tegelijk; release/acquire-publicatie; alle loads afhandelen vóór bufferhergebruik |
| Andere I/O | `uring_*`, `PILOT_REAL`, mirror-readcode | `pipe_ready()` meldt bij URING altijd niet-gereed; niet behandelen als gewone completionqueue |
| Residentie | `ESlot`, `ecache`, `ws`, pin-store, `numa_slab_bind` | Slots/slabs worden geswapt; pointeridentiteit alleen is geen stabiele jobidentiteit |
| Profiling/replay | `profile_reset`, `prof_base`, `prof_report`, `run_replay` | Overlappende tijden niet optellen alsof ze exclusieve fasen zijn |
| M3-oracle | [`test_m3_tiny.py`](../c/tests/test_m3_tiny.py), [`oracle_m3.py`](../c/tools/oracle_m3.py) | Bestaande tinyketen gebruikt int8 en `IDOT=0`; geen volledige grouped-int4- of lang-contextgate |
| PIPE-regressie | [`test_pipe_block.c`](../c/tests/test_pipe_block.c) | Test spin/block-waits, publicatie en batchgeneraties; bouw hierop voort |
| Plaatsingssimulatie | [`simulate_scheduler.py`](../c/tests/simulate_scheduler.py), [`placement_balance.py`](../c/tools/placement_balance.py) | Bestaande voorspellingen zijn geen M3-runtimewinst; geometrie en kosten opnieuw vaststellen |
| Publieke runtimegrenzen | [Segment](segment-runtime.md), [Edge](edge-runtime.md) | Bestaande ABI's blijven behouden; dit plan bouwt eerst een interne executor, geen nieuw netwerkprotocol |

`XEXP=1` is geen oplossing voor de eerste scope: het vereist `fmt==2`,
terwijl de lokale M3-experts `fmt==4` gebruiken. De vaste 12 chunks per
expert zijn bovendien geen geschikte algemene taakgrootteregel.

## Regels voor alle uitvoerders

1. Bewaar de bestaande werkboomwijzigingen. Een checkout van alleen HEAD
   bevat de onderzochte lokale patches niet. Maak de baseline reproduceerbaar
   met commit, diff, benodigde nieuwe bestanden en checksums; stage of commit
   geen werk van anderen als onderdeel van je stap.
2. Eén gedragverandering per stap. Verander geen quantisatie, routerkeuze,
   sparse-selectiesemantiek, contextlimiet of sampling om sneller te lijken.
3. Gebruik opt-in gedrag. Feature uit betekent het bestaande pad. Controleer
   eligibility vóór publicatie van taken. Niet-ondersteunde formats, `S>1`,
   speculatie en backends nemen eerst het bestaande pad.
4. CPU-only betekent ook dat CUDA/Metal/Vulkan-expertuitvoering werkelijk uit
   staat. Print de effectieve keuze; leid die niet af uit de binarynaam.
5. Iedere taak heeft een expliciete eigenaar en levensduur voor inputs,
   weights, scratch en output. Evictie, reset en slab-swaps mogen geen buffer
   hergebruiken die I/O of compute nog bezit.
6. Geen blocking diskread in een compute-worker. Geen nieuwe onbeperkte
   queue, workerpool, prefetchbuffer of modelbrede duplicatie van weights.
7. Behoud de oorspronkelijke floating-pointvolgorde binnen outputrijen en
   de oorspronkelijke expert-samenvoegvolgorde. Compute-voltooiingsvolgorde
   mag de reductievolgorde niet bepalen.
8. Een fallback na gedeeltelijke uitvoering is geen simpele tweede aanroep
   van de oude forward. Eerst afronden of draineren, pas daarna hergebruik;
   voeg geen expertoutput of KV-mutatie dubbel toe. Bij onherstelbare fout
   expliciet falen volgens het bestaande contract.
9. Geen GPU-, cluster- of publieke Segment/Edge-uitbreiding in de eerste
   stappen. Ondersteuning alleen claimen voor combinaties die zijn getest.
10. OpenMP kan bestaande workers hergebruiken. Meet joins, werkverdeling en
    wachten; stel niet zonder bewijs dat OS-threads telkens worden aangemaakt.

## Gateprotocol en bewijs

Iedere stap heeft drie aparte beoordelingen:

- **C — correctheid:** dezelfde vereiste uitkomsten en veilige uitvoering.
- **M — mechanisme:** trace/test toont dat de bedoelde route daadwerkelijk
  actief was en het specifieke probleem oplost.
- **P — performance:** lagere latency in een vooraf gekozen regime zonder
  onaanvaardbare regressie in de controles.

Gebruik `PASS`, `FAIL`, `INCONCLUSIVE` of `NOT_RUN`, met scope. Een simulator,
mock, exitcode 0 of hogere CPU-bezetting is op zichzelf geen P-PASS. Een
ontbrekende machine geeft `NOT_RUN: dual Rome`, niet een impliciete PASS.
Voer geen afhankelijke stap uit als de vereiste C/M-gate faalt. Bij onduidelijke
performance mag veilige opt-in infrastructuur blijven voor vervolgonderzoek,
maar het gedrag wordt niet standaard ingeschakeld en de winstclaim ontbreekt.

Voor performancebevorderingen: kies vooraf een primaire scenariofamilie en
controlefamilies. Start met vijf gepaarde A/B-metingen met afwisselende
volgorde, identieke tokens en gedocumenteerde cachecondities. Een voorgestelde
promotiedrempel is minstens 5% minder mediane ms/token in het doelregime,
een gepaard 95%-interval dat nul verbetering uitsluit, en hoogstens 3%
regressie in de controles. Leg deze engineeringgrenzen vóór de run vast;
het zijn geen voorspellingen van haalbare winst. Bij te veel ruis: meer
metingen binnen het afgesproken budget of `INCONCLUSIVE`, geen selectieve runs.

Staartlatency vraagt meer samples dan een 32-tokenprobe. Vermeld altijd het
aantal tokens en runs; noem p99 bij weinig samples exploratief. Controleer
ook piek-RSS, gelezen bytes, werkelijk gevoelde wachttijd en CPU-tijd.

Bewaar artefacten per stap in een nieuwe resultatenmap. Het verslag bevat:

```text
Stap / datum / uitvoerder:
Baseline: commit + patch-/bestandschecksums + binarychecksum
Wijziging: bestanden, functies, featureflag, eligibility
Machine: CPU, cores, OS, NUMA, RAM, drives, compiler en buildflags
Effectieve configuratie: threads, plaatsing, PIPE, backend, quantisatie, context
Input: tokenbestand + checksum; opwarming en residentiebudget
Commando's: letterlijk, met werkdirectory en exitcodes
C: status + bewijsbestanden + aantal gecontroleerde uitkomsten
M: status + activatieteller/trace + aangetoonde gebeurtenisvolgorde
P: status + A/B-tabel + spreiding/interval + regressies
Niet getest / beperkingen:
Besluit: volgende toegestane stap, of concrete herstelopdracht
```

## Volgorde

| Stap | Resultaat | Vereiste eerdere gates |
|---|---|---|
| 0 | Reproduceerbare baseline en geschikte fixtures | Geen |
| 1 | Betrouwbare fase- en dependencytrace | 0 C/M |
| 2 | MSA-parallellisatie | 1 C/M |
| 3 | Interne DAG-contracten en seriële referentie-executor | 1 C/M |
| 4 | Gereedstaande experts uitvoeren tijdens PIPE-loads | 3 C/M |
| 5 | Reentrante grouped-int4-rowkernels | 4 C/M |
| 6 | Parallelle expert-tasks, inclusief shared expert | 5 C/M |
| 6a | Correctheidsherstel en reconciliatie van Gate 6 | Implementatie en reviewbevindingen uit 6; geen eerdere 6 C-PASS vereist |
| 7 | I/O- en residentieplanner in shadow mode | 1, 3 en 6a C/M; echte integratietrace uit 6/6a vóór promotie |
| 8 | Begrensde planner actief maken | 6a en 7 C/M |
| 9 | NUMA-eigenaarschap en lokale teams | 6a C/M; aparte A/B met 8 indien actief |
| 10 | Combinatie-, regressie- en promotiegate | 2, 6a, 8, 9 voor de ondersteunde scope |

Werk standaard in deze volgorde. Stap 2 en de DAG-route hebben verschillende
inhoudelijke dependencies; test hun individuele flags en hun combinatie.
Een gemiste MSA-performancegate blokkeert niet automatisch de expertanalyse.
Aanvulling 6 september: stap 6a is een verplichte herstelgate vóór uitvoering
van stap 7, ook in shadow mode. Dit vervangt bewust de eerdere toestemming
om shadow vanuit alleen 1/3 C/M te starten. Stappen 0–5 worden hierdoor niet
heropend; stappen 7–10 behouden hun nummer. Historische Gate-6-PASS-tekst is
geen toestemming om 6a over te slaan. Het uitschrijven van de plannerbriefing
is geen uitvoering van stap 7.

## Stap 0 — baseline, fixtures en uitvoerbaar benchmarkrecept

**Status: afgerond voor Gate 0 C/M binnen de vastgelegde fixture- en tiny-replay-scope.**
Het gateverslag, de exacte commando's en de expliciete full-modelbeperking staan in
[`results/m3-execution-step0-baseline-2026-09-05.md`](results/m3-execution-step0-baseline-2026-09-05.md).
Gate 0 P blijft `NOT_RUN`; dit blokkeert stap 1 niet, omdat stap 1 volgens de
dependencytabel alleen Gate 0 C/M vereist.

**Briefing voor de uitvoerder**

> Leg de werkelijke M3-uitgangsversie en testomgeving vast. Maak een herhaalbare
> baseline zonder scheduling te wijzigen. Lever een recept dat een volgende
> collega letterlijk kan uitvoeren, met bestaande fixtures beschermd.

Uitvoering:

1. Lees status/diff en controleer de codekaart. Kies een geïsoleerde
   werkkopie die de noodzakelijke lokale wijzigingen bevat. Leg de herkomst
   vast; een schone upstreambuild is geen gelijkwaardige baseline.
2. Controleer de geïnstalleerde toolchain. Bestaande make-targets zijn
   `colibri`, `m3-tiny-check` en `tests/test_pipe_block` met waar nodig `.exe`.
   Bouw en voer ze alleen uit in de gekozen testomgeving.
3. Let op: `m3-tiny-check` roept `m3-tiny-generate` aan, dat de bestaande
   `c/m3tiny` en `c/m3tiny_i8` verwijdert en opnieuw maakt. Voer dit niet
   blind uit in de huidige gedeelde werkboom. Gebruik een gevalideerde
   geïsoleerde fixturemap, of de bestaande driver op bestaande fixtures:

   ```text
   python tests/test_m3_tiny.py --binary <absolute-binary> --snap <absolute-tiny-map> --ref <absolute-ref.json>
   ```

   Dit is een commandotemplate vanuit `c/`; vervang de placeholders eerst.
   De driver controleert de prefill/decode-tellingen. De engine-exitcode
   alleen is onvoldoende voor de REF/TF-gate.
4. Maak een aanvullende grouped-int4-fixture met bekende tensors en een
   onafhankelijke numerieke referentie. De bestaande numpy-oracle leest
   int8; geef hem niet ongemerkt een int4-container. Leg de scope van iedere
   oracle vast voordat je de testresultaten als bewijs gebruikt.
5. Gebruik de bestaande `run_replay`-route met `SNAP`, `REF`, `REPLAY=1` voor
   vaste tokeninput. Verifieer dat tekstprompt-/serve-modi niet voorrang
   krijgen. Check de gerapporteerde decodecount; ongeldige replay-input kan
   terugkeren zonder harde procesfout. Replay fixeert tokeninput, niet
   automatisch hidden states of routes bij numerieke verschillen.
6. Begin met een korte context, 32 tokens als smoke en vervolgens 256
   decode-stappen voor de eerste baseline. Breid doelgericht uit naar 2K,
   8K, 32K en 64K context als het geheugenbudget dat toelaat. Bereken vóór
   allocatie de KV-, indexcache- en scratchbehoefte. Tinytests zijn voor
   correctheid; volledige modelmetingen voor representatieve performance.
7. Leg residentie vast: volledig resident als het past, anders een identiek
   beperkt budget. Geen globale page-cacheflush op een gedeelde machine;
   beschrijf en herhaal de opwarmprocedure.

**Gate 0**

- C: bestaande tiny- en PIPE-tests slagen; grouped-int4-referentie is
  beschikbaar en gecontroleerd; bestaande data zijn behouden.
- M: exacte build, route, tokens, aantallen en configuratie staan in het
  manifest. Ten minste één benchmarkrecept is door een tweede uitvoering
  reproduceerbaar gebleken.
- P: baseline en ruis gerapporteerd; er is nog geen optimalisatiewinst.

Bij ontbrekende full-modelcapaciteit: lever fixturebewijs en markeer de
betreffende performancecases `NOT_RUN`. Verzin geen extrapolatie naar Rome.

## Stap 1 — observability zonder ander uitvoergedrag

**Status: afgerond voor Gate 1 binnen de tiny CPU/PIPE-scope.** Het
implementatie- en gateverslag staat in
[`results/m3-execution-step1-observability-2026-09-05.md`](results/m3-execution-step1-observability-2026-09-05.md).
Full-model-, lange-context-, Naples- en Rome-metingen blijven afzonderlijk
open en zijn geen grond om deze observabilitygate als schaalbaarheidsbewijs te
interpreteren.

**Briefing**

> Voeg aparte MSA-timing en een optionele dependencytrace toe. Verander geen
> taakvolgorde, threads, plaatsing of kernelkeuze. Maak wachten verklaarbaar.

Voeg exclusieve timers toe voor norm/RoPE, indexscan en selectie, naast de
bestaande projectie/core/outputtimers. Werk reset, snapshot en rapportage
allemaal bij. Ontwerp optionele events met monotone timestamps, een
forward-/laag-/expert-/generatie-ID en resource-ID:

```text
route_ready, load_queued, load_start, load_complete, weights_acquired,
compute_ready, compute_start, gate_up_done, activation_done, down_done,
expert_done, reduction_done, consumer_wait_begin/end, weights_released
```

Dit zijn voorgestelde namen. Leg het definitieve schema met versie vast.
Gebruik begrensde per-threadbuffers; niet per tile `fprintf`, een globale
profielmutex of een heapallocatie. Rapporteer overflow; onvolledige traces
zijn geen volledig dependencybewijs. Maak onderscheid tussen I/O-service,
queuewait, felt wait en gelijktijdige activiteit. Optelling van worker-seconden
is geen wall time. Bestaande profilerlabels kunnen bij de nieuwe executor
onvolledig worden en moeten dan expliciet worden aangepast.

**Gate 1**

- C: trace aan/uit geeft dezelfde uitkomsten op de baseline; timertotalen
  zijn consistent, zonder dubbeltelling als exclusieve fasen.
- M: één trace verklaart een wachtende expert, een residentiehit en de
  volledige MSA-tijd. Leg ook vast wanneer een event nog niet bereikbaar is.
- P: overhead met tracing uit en aan gemeten. Gedetailleerde tracing blijft
  uit tijdens uiteindelijke snelheidstests; gebruik lichte counters voor
  routeactivatie. Onverklaarde grote overhead eerst oplossen.

## Stap 2 — MSA over keyblokken paralleliseren

**Briefing**

**Status: afgerond voor Gate 2 binnen de vastgelegde scanner- en tiny-M3-scope.**
Zie [`results/m3-execution-step2-msa-parallel-2026-09-05.md`](results/m3-execution-step2-msa-parallel-2026-09-05.md)
voor de scalaire referentiegate, gedwongen multiworker-test, scanbenchmark en
de expliciete beperking van de full-model-lange-contextclaim.

> Maak de indexscan bij S=1 parallel over keyblokken. Behoud dotproduct,
> pooling, local-blockbeleid, tie-breaking en volgorde van geselecteerde
> blokken exact zoals in de huidige implementatie. Los geen ander
> sparse-attentionvraagstuk in deze stap op.

Verdeel onafhankelijke blokken over workers. Eén worker berekent per blok
de vier heads met dezelfde dotkernel en dezelfde keyvolgorde; schrijf naar
disjuncte scores. Voer top-k aanvankelijk op dezelfde seriële manier uit.
Zorg dat alle relevante Ic-rijen gepubliceerd zijn voordat de scan start.
Kies een gemeten minimumwerkgrootte voor parallellisatie; kleine context
mag op de bestaande route blijven. `S>1` blijft eerst ongewijzigd.

**Gate 2**

- C: identieke scores en geordende selecties tegen de bestaande C-route bij
  dezelfde build, threads 1 en meerdere cores; aparte scalarreferentie met
  vooraf bepaalde floating-pointtolerantie. Test lege/afgekapte blokken waar
  geldig, ties, local blocks, `kv_start`, 127/128/129 en 2047/2048/2049 tokens,
  plus langere context. Tiny-oracle en logits/replayvergelijking blijven groen.
- M: counters tonen dat S=1 werkelijk meerdere blokken op meerdere workers
  uitvoert; tests dwingen het nieuwe pad ook bij kleine fixtures af.
- P: indexscan én totale tokenlatency voor de gekozen contexten rapporteren;
  korte-contextcontrole meenemen. Een snellere scan alleen is geen end-to-endclaim.

## Stap 3 — contracten en een seriële DAG-executor

**Briefing**

> Beschrijf taken, gereedheid en bufferownership voor één M3-laag. Bouw een
> interne seriële referentie-executor met dezelfde uitvoervolgorde als nu.
> Voeg nog geen nieuwe compute-threads of plaatsingsbeleid toe.

Leg in code en een kort contract vast:

- `ExecutionContext`: model/forward/laag/generatie, budgetten en annulering.
- `ExpertTask`: exacte route-index, expert-ID, weight-handle, input/scratch,
  output en afhankelijke taken. Geen rauwe `ESlot*` zonder levensduurcontract.
- Weight-state: `absent → queued → loading → resident`; onafhankelijk daarvan
  compute-state `waiting → ready → running → complete` en fout/annulering.
  Ownership/refcounts beschermen residentie tijdens gebruik.
- Een expert wordt ready bij geldige input én volledige geldige weights én
  exacte routerselectie. De shared expert heeft geen routed-selectie nodig.
- Bufferrelease volgt na de laatste I/O-/computegebruiker, niet alleen na
  `load_complete`. Alle vereiste outputs worden in baselinevolgorde samengevoegd.

Stabiele jobidentiteit bevat minimaal forward, laag, expert en generatie.
Speculatieve predictions krijgen een aparte status. Zij mogen weights
voorbereiden, maar geen modelbijdrage autoriseren.

**Gate 3**

- C: seriële executor reproduceert de oude laagoutputs; test reset, dubbele
  completion, stale generation en foutpad. Shared output telt precies één keer.
- M: iedere start heeft vervulde dependencies; iedere release volgt op de
  laatste gebruiker; geen taken blijven achter na het laag-einde.
- P: kosten van de contractlaag gemeten; geen claim van versnelling.

## Stap 4 — PIPE-aware ready-first, nog zonder parallelle experts

**Briefing**

> Laat de CPU een gereedstaande expert uitvoeren terwijl een eerdere expert
> nog wordt geladen. Behoud één compute-expert tegelijk en de bestaande
> interne OpenMP-kernels. Bereken de shared expert zodra zijn input gereed is.

Gebruik eerst het bestaande pthread-PIPE-pad. Houd één batchgeneratie en
het huidige cachebeleid aan. `pipe_ready` is een selectiehint; respecteer
de bestaande `pipe_wait`/publicatie voordat de slab wordt gebruikt. Als er
geen werk gereed is, wacht op een completion met een gecontroleerd predicate;
voorkom gemiste wakeups. Bewaar afzonderlijke expertoutputs en reduceer
pas in de oorspronkelijke volgorde. Shared compute vóór routed compute
verandert de volgorde van optellen niet.

Behoud de eindvoorwaarde dat alle gepubliceerde reads zijn afgehandeld,
ook jobs die geen compute meer nodig hebben. Cachepromotie en ws-swaps
blijven pas daarna toegestaan. URING, PILOT_REAL en GPU-routes vallen in
deze eerste versie expliciet terug voordat taken worden gepubliceerd.

**Gate 4**

- C: identieke laagoutputs en logits voor alle-hit, alle-miss en gemengde
  cases, spin/block-waits, meerdere generaties en beperkte cachecapaciteit.
- M: een deterministische test houdt load A vast met een latch terwijl B
  gereed is. B/shared starten en eindigen vóór de test A vrijgeeft. Gebruik
  gebeurtenisvolgorde als bewijs, geen fragiele sleep-duurassertie.
- C/M: faultinjectie, annulering en reset veroorzaken geen use-after-free,
  dubbel resultaat of hergebruik vóór drain. Bestaande PIPE-gate blijft groen.
- P: echte PIPE=1 A/B met dezelfde reads, budgetten en tokens; rapporteer
  felt wait en tokenlatency. Geïnjecteerde vertraging bewijst alleen mechanisme.

## Stap 5 — reentrante grouped-int4-rowkernels

**Briefing**

> Maak gate/up-, activation- en downberekening aanroepbaar voor een expliciet
> bereik outputrijen, met uitsluitend expliciete input, weights en scratch.
> Laat de nieuwe rowkernels eerst door de seriële executor aanroepen.

Extraheer de bestaande numerieke loops; schrijf niet tegelijk een nieuwe
SIMD- of quantisatiekernel. Maak uitvoerpaden voor de werkelijk aanwezige
shared-tensorformats; als die nog niet veilig zijn blijft shared buiten de
concurrente fase. Verwijder afhankelijkheid van `g_pq` uit nieuwe taken of
geef de relevante context expliciet door. Audit ook profilerupdates,
activatieconfiguratie en thread-local scratch die per OS-thread wordt opgelost.

Er komt maar één niveau van computeparallellisme: rowtaken mogen niet zelf
een volledig OpenMP-team starten. De bestaande algemene helpers blijven
beschikbaar voor de fallbackroute.

**Gate 5**

- C: full-row versus meerdere tiles geeft identieke resultaten tegen de
  bestaande kernels bij dezelfde build. Test laatste korte tile, group- en
  SIMD-grenzen, nulgewichten, clipping en verschillende expertinputs.
- C: onafhankelijke grouped-int4-referentie plus M3-oracle; expliciet bewijzen
  dat `fmt==4`, group size 64 werkelijk door de nieuwe code ging.
- M: meerdere gelijktijdige kernelcalls met verschillende scratchinputs
  beïnvloeden elkaar niet. Geen writes naar gedeelde `g_pq` op dit pad.
- P: enkelvoudige kerneloverhead gemeten; geen executorwinst claimen.

## Stap 6 — parallelle expert-tasks

**Status gecontroleerd op `b06130b`: failure-publicatierace hersteld;
afsluiting van 6a nog gedeeltelijk open wegens evidencegaten.**
De bevinding op `b9bf5de` is gerepareerd in `04c01fa`; `fa28eaa` corrigeert
de interleavingtest. De blocking Archer/TSan-, ASan/UBSan- en Linux-enginejobs
slagen op `b06130b` in CI-run `34050827680`. Het parallelle mechanisme blijft
onderbouwd; P blijft `NOT_PROMOTED`, met all-resident `NOT_RUN`.
Het actuele C-PASS-record in `results/m3-execution-step6-invariants-2026-09-06.md`
erkent deze reparatie, maar sluit de expliciet uitgebreidere 6a-eisen hieronder
nog niet volledig. Geen nieuwe enginerace vastgesteld; de resterende punten
betreffen testdekking en gate-reconciliatie.

**Briefing**

> Voer gereedstaande (expert, output-row tile)-taken uit met een begrensd
> compute-team. Eén expert mag aan down beginnen zodra zijn eigen volledige
> activation gereed is; andere experts hoeven daarvoor niet klaar te zijn.

Kies de kleinste geschikte executor achter het contract uit stap 3, bijvoorbeeld
een behouden OpenMP-regio met tasks of een begrensde pool. Motiveer de keuze
met meetbare overhead en lifecycle. Bouw geen algemene distributed scheduler.

Gebruik dependencies per expert, private outputs en een vaste reductie.
Tileaantallen volgen matrixgrootte en teamomvang; hardcode geen 12 chunks.
Voorkom oversubscription met I/O-workers en andere OpenMP-fasen. I/O-completions
maken compute gereed maar voeren de matmul niet op de I/O-worker uit.
Gebruik begrensde scratcharenas. Idle workers parkeren of spinnen volgens
een meetbaar beleid; geen onbeperkte busy loop bij disk-wacht of prompt-idle.

**Gate 6**

- C: dezelfde uitkomsten over 1, 2, 4 en meer beschikbare cores; herhaalde
  wisselende voltooiingsvolgorde, shutdown en resourcefouten. Gebruik
  ASan/UBSan waar ondersteund en een passende racecheck; een niet-werkende
  sanitizer met de gekozen runtime is geen geslaagde racegate.
- M: trace toont overlappende expertcompute, shared deelname en down-start
  onafhankelijk van een nog geblokkeerde andere expert. Geen onverwachte
  nested teams of gelijktijdige toegang tot één scratchbuffer.
- P: vergelijk oude engine, stap 4 en stap 6 bij gelijke residentie. Een
  afzonderlijke all-resident test is nodig om I/O-maskering uit te sluiten.
  Zonder voldoende RAM blijft dat geval `NOT_RUN`.

## Stap 6a — correctheidsherstel en gate-reconciliatie

**Status: AFGEROND op `59b4eab`; Gate 6a C/M-PASS vóór stap 7.**

Statusreconciliatie van 6 september, na inspectie van de reparatiediff en
[CI-run 34050827680](https://github.com/jtinbergen/colibri/actions/runs/34050827680)
op `b06130b9c408304a3b698f011301636079b67795`:

- Afgerond: atomische contextflags en consistente accesses; concurrente
  helper-level readiness/failure-regression; injectie van preflightallocatie-
  fouten; assert op geen gedeeltelijke calleroutput bij failure/drain;
  blocking Archer-job. De genoemde CI-run is overall succesvol en Archer,
  ASan/UBSan en Linux-engine zijn afzonderlijk succesvol. `b06130b` bevat de
  eerdere codefixes via zijn ouders; zelf wijzigt het CI-classificatie en docs.
- Afgerond in `6558716`/`59b4eab`: alle acht preflightallocaties, een echte
  `moe()`-dispatcher readiness/failure-interleaving, actieve failure/drain,
  beide completionpermutaties en worker-counttests 1/2/4. De nieuwe
  productiepadtest vraagt onder CI expliciet een team van twee workers zodat
  de interleaving niet achter de PIPE-reservering serialiseert.
- Afgerond: blocking CI-run `34054894622` op `59b4eab`. Archer/TSan,
  ASan/UBSan, Linux-engine, Windows-engine en de volledige overige matrix
  slagen. Hosted TSan/libgomp en Helgrind blijven zichtbare, informatieve
  runtime-diagnostics met `continue-on-error`; zij zijn niet de gezaghebbende
  Archer-classificatie.
- Expliciete beperkingen staan in het 6a-resultatenverslag; niet-injecteerbare
  thread-create-fouten en geen aparte ondersteunde parallelle cancellation-API
  worden niet als PASS geclaimd.
- De actuele Step-6- en 6a-verslagen zijn hiermee gereconcilieerd. Step 7
  blijft inhoudelijk ongewijzigd en is nog niet geïmplementeerd.

**Briefing**

> Sluit de concrete Step-6-reviewbevindingen bij `b9bf5de`. Herstel de
> failure-publicatie, bewijs de ontbrekende relevante lifecyclepaden en maak
> de gezaghebbende racecheck afdwingbaar. Behoud de ondersteunde CPU/S=1/
> grouped-int4-scope, numerieke volgorde en begrensde executor. Bouw geen
> nieuwe scheduler en voeg in dit herstel geen opslagplanner toe.

Werkpakketten, in deze volgorde:

1. **Failure-publicatie:** audit de contextflag-accesses die de parallelle
   executor werkelijk deelt. Bij de reviewversie roept de producer in
   `c/colibri.c:7224` `m3_dag_task_ready()` aan, dat in `c/m3_dag.h:116`
   `ctx->failed` gewoon leest, terwijl een worker via `colibri.c:7253` en
   `m3_dag.h:174` atomisch schrijft. Maak alle mogelijk concurrente accesses
   consistent gesynchroniseerd. Een eerdere acquire-check of latere taskwait
   beschermt die tussenliggende gewone read niet. Zoek functies opnieuw als
   regels verschoven zijn; wijzig geen uitsluitend seriële semantiek zonder
   noodzaak. Niet-aangeroepen cancellationhelpers bewijzen geen productiepad.
2. **Gerichte regression:** forceer via testhooks/latches dat een eerdere
   expert faalt terwijl de dispatcher nog andere experts op readiness kan
   beoordelen. Test de echte producer/context, niet alleen losse rowkernels
   of opeenvolgende lifecyclehelpers. Gebruik geen sleeps als bewijs en voeg
   geen testsynchronisatie toe die de onderzochte concurrente toegang zelf
   wegordent. Laat Archer het pad classificeren. Assert dat geen gedeeltelijke
   bijdrage vóór succesvolle retirement wordt gereduceerd en dat de gekozen
   fout/fallbackroute geen dubbele bijdrage of vroeg bufferhergebruik geeft.
3. **Lifecycle- en resourcebewijs:** inventariseer de bestaande tests en vul
   aantoonbare gaten aan: werkelijke allocatiefouten tijdens preflight met
   cleanup vóór publicatie; failure/drain met nog actieve numerieke taken;
   herhaalde completionvolgordes; ondersteunde annulering; worker-/team-einde
   vóór scratchvrijgave. Een overflowguard is geen allocatiefouttest en een
   OpenMP-region-einde is geen bewijs van expliciete PIPE-worker-shutdown.
   Leg per claim vast welk productiepad de test raakt. Voor thread-create-
   fouten of cancellation/shutdown die de runtime niet injecteerbaar of niet
   ondersteund maakt: rapporteer de concrete beperking en het foutcontract;
   claim geen pass. Een noodzakelijke ondersteunde veiligheidsgarantie mag
   niet stilzwijgend worden geschrapt: ontbrekend bewijs blijft blokkerend
   tenzij de sterkere reviewer een expliciete scopebeperking accepteert.
4. **CI:** maak de Archer/TSan-job gezaghebbend én blocking; verwijder zijn
   `continue-on-error: true`. Houd oude distro-TSan/Helgrind-runtimeconflicten
   afzonderlijk als informatieve diagnostics. Verifieer de job zelf en zijn
   uitgevoerde testsubset; alleen een groene workflowbadge volstaat niet.
5. **Verificatie en verslag:** voer de gerichte DAG-, PIPE- en grouped-int4-
   fused/unfused-tests uit, plus ASan/UBSan en Archer waar ondersteund. Herhaal
   de numerieke worker-countvergelijking 1/2/4 en de beschikbare grotere team-
   omvang op de herstelde versie; ontbrekende hardware is geen pass. Bewaar
   commit, buildflags, commando's, uitkomsten en artefacten in een nieuw
   6a-resultatenverslag. Werk daarna het actuele gateblok van het Step-6-
   invariantenverslag bij; behoud de historische runs als historische evidence.

**Gate 6a**

- C: PASS binnen de ondersteunde scope. Geen mixed atomic/non-atomic
  contextaccess op het parallelle pad; productie-interleaving, alle acht
  allocatiefouten, lifecyclecases, worker-counttests en blocking Archer-run
  zijn groen.
- M: PASS binnen de ondersteunde scope. Tests tonen de bedoelde
  failure/readiness-interleaving, drain vóór reuse, completionpermutaties,
  exact-once lifecycle en begrensd expertparallelisme.
- P: geen nieuwe winstclaim vereist. `NOT_PROMOTED` en all-resident `NOT_RUN`
  mogen blijven; rapporteer relevante regressies zonder ze als C/M te maskeren.
- Afsluiting: finale diff/evidence is vastgelegd in
  [`results/m3-execution-step6a-gate-2026-09-06.md`](results/m3-execution-step6a-gate-2026-09-06.md)
  en CI-run `34054894622`. De actuele 6/6a-gaterecords spreken elkaar niet
  tegen. Step 7 mag nu beginnen; Step 7 zelf blijft `NIET GEÏMPLEMENTEERD /
  NIET GEGATED`.

## Stap 7 — planner in shadow mode

**Status: NIET GEÏMPLEMENTEERD / NIET GEGATED in deze checkout.**
De onderstaande opslagtopologie- en contentiontekst is een uitvoeringscontract,
geen implementatie-evidence. Begin uitvoering pas na afsluiting van 6a C/M.

**Briefing**

> Laat een planner voorstellen doen op echte traces zonder reads, plaatsing
> of compute te wijzigen. Voorspel beschikbaarheid en consumer-wacht, geen
> hitrate als vervangende succesmaat.

Hergebruik bruikbare bestaande traces/simulatorcode, maar verifieer hun
geometrie en maak M3-specifieke aannames expliciet. Verzamel kosten per
requestgrootte, bron, queuedepth en compute-resource. Gebruik getrainde
prompts voor kalibratie en aparte prompts voor evaluatie; geen toekomstige
routerkeuze uit de replay gebruiken als online voorkennis.

### Aanvulling 6 september: drives, controllers en toelating

Dit is een verplicht implementatiecontract, nog geen geïmplementeerde functie.
De eerste meetmachine heeft twee NVMe-drives met mogelijk verschillende
bruikbare concurrency (bijvoorbeeld vier versus één read). De doelopstelling
heeft ongeveer zes à zeven drives verdeeld over drie controllers; aantallen
en indeling mogen veranderen. Hardcode geen aantallen, uniforme queuedepth
of gelijke verdeling over replicas. De genoemde hardwarecapaciteiten zijn
te meten invoer, geen bewezen eigenschappen of standaardinstellingen.

**7a — Expliciete topologie en bounded datamodel**

Lever eerst een gevalideerde configuratie en parser met deze logische velden;
de concrete bestandsindeling mag eenvoudig blijven:

- `resource_id`, `kind` (drive/controller/upstream), `max_inflight`,
  `max_inflight_bytes` en een verwijzing naar het kalibratieprofiel.
- Per drive een lijst unieke `resource_id`s op het gedeelde I/O-pad, inclusief
  de drive zelf. Een upstream-link kan meerdere controllers omvatten. Een
  gedeelde resource staat maar één keer in de configuratie en wordt per read
  maar één keer belast; geen vier kopieën van één controllerbudget.
- Per beschikbare weightkopie: model-/tensoridentiteit, bestand/range,
  byteaantal en fysieke drive. Alleen inhoudelijk equivalente, beschikbare
  kopieën zijn kandidaten. Twee paden naar dezelfde drive zijn geen twee
  onafhankelijke resources. Controllerlidmaatschap betekent niet automatisch
  dat alle reads geserialiseerd moeten worden.
- Per request: unieke request- en forward/generatie-identiteit, consumer-node,
  demand/prefetch, enqueue-tijd, consumer-need, state, gekozen kopie en de
  gereserveerde resources. Definieer configuratiegrenzen voor het aantal
  resources, wachtende requests en bytes; overflow wordt vóór publicatie
  afgewezen. Geen onbegrensde queue of stil afgekapt resourcepad.

Gebruik configureerbare topologie; OS-discovery mag helpen maar is geen
voorwaarde. Ontbrekende of tegenstrijdige mappings maken actieve planning
voor die configuratie ineligible. Shadow rapporteert `UNKNOWN`; het neemt
geen onafhankelijke controllers of oneindige capaciteit aan. Een ongeldige
configuratie laat het bestaande pad vóór plannerpublicatie intact.

**7b — Kalibratie van capaciteit én contention**

Meet per drive de completion-latencyverdeling en throughput voor relevante
requestgroottes en in-flight aantallen, minstens 1/2/4 en verdere waarden
binnen een vooraf begrensd meetbudget. Gebruik echte expert-readpatronen;
noteer cacheconditie, buffered/direct I/O, readbytes, gelijktijdige loads,
CPU-verwerking en sampleaantallen. Scheid I/O-completion van weight-ready
na conversie/verwerking. Kies een bruikbare concurrencygrens uit de gemeten
latencycurve, niet uit een fabrikantmaximum of alleen piekbandbreedte.
Leg vóór de meting vast welke latencytoename en onzekerheid acceptabel zijn.

Meet daarna drives binnen iedere controllergroep tegelijk en vervolgens
groepen over controllers heen. Begin met gestructureerde paren en volledige
groepen; eis geen exponentiële test van alle subsets. Verfijn waar de
voorspelling onvoldoende klopt. Topologie alleen bewijst geen contention;
een gemeten gedeeld knelpunt kan ook boven de controllers liggen.

Bewaar per profiel: topologie/configuratiehash, meetcondities, ondersteunde
grootte-/loadklassen, latency en spreiding, aggregate bandwidth in bytes/s,
afgeleide admissiongrenzen en versie. Een bytebudget is geen bandwidthbudget.
Noem expliciet de tijdseenheid; een capaciteit zoals “500 gigabytes” zonder
`/s` mag niet als bandbreedte worden ingevoerd. Ontbrekende meetklassen blijven
`UNKNOWN`; geen optimistische extrapolatie. Wijzig topologie of I/O-modus
alleen tussen gedrainde generaties en herkalibreer de getroffen profielen.

**7c — Pure voorspeller en deterministische beslissing**

Implementeer eerst een pure functie van een immutable snapshot, request en
kandidaatkopie. Zij doet geen I/O, verandert geen counters en leest geen
toekomstige replay-events. Gebruik één monotone tijdas:

```text
consumer_need = voorspelde vroegste gebruikstijd uit de nu bekende DAG
               (andere dependencies + beschikbaarheid compute-resource)
prediction = predict(snapshot, request, candidate, proposed_dispatch_time)
arrival = prediction.weight_ready_time
slack = consumer_need - arrival - prediction.uncertainty_margin
predicted_wait = max(0, -slack)
```

`consumer_need` wordt niet afgeleid van de eigen voorspelde I/O-completion:
dat zou de te meten stall verbergen. Het is een herberekenbare soft deadline,
geen garantie. Niet-gecommitteerde toekomstige routerkeuzes blijven onbekend.
De voorspeller retourneert ook de gewijzigde completiontijden van al actieve
en gereserveerde requests op gedeelde resources, plus reden/`UNKNOWN`.

Gebruik gemeten load-afhankelijke servicecurves. Tel een expliciete queuewait
alleen op bij een serviceprofiel dat die wachttijd uitsluit; een end-to-end
profiel bevat haar al. Sommeer niet voor ieder resourcepad-element opnieuw
de volledige transfertijd. De drive en zijn controller begrenzen dezelfde
byteflow. De voorspelde totale transfer over een gedeelde resource mag haar
gekalibreerde aggregate capaciteit niet overschrijden. Een enkel gemiddeld
latencygetal maal `(inflight+1)` is geen voldoende completionmodel.

Vaste eerste voorspeller: een conservatieve, event-driven fluid-simulatie.
Dit is een toetsbaar startmodel; slechte held-out nauwkeurigheid blokkeert
activering en vraagt modelreview, geen verborgen heuristiek van de uitvoerder.

- Een serviceprofiel levert vaste startupduur per grootteklasse en per
  resource een aggregate transferrate `C_r(loadklasse, grootteklasse)`.
  Startup is exclusief byte-transfer, queuewait en conversie. Fit deze
  componenten op de meetreeksen van 7b; sla fitresiduen op. Een end-to-end
  latencytabel mag niet rechtstreeks als extra startup worden opgeteld.
- Gebruik voor tussenliggende groottes/load de kleinste gemeten klasse die
  beide naar boven afdekt; geen interpolatie. Bij gemengde groottes geldt per
  resource de laagste toepasselijke aggregate rate. Geen dekkende klasse
  betekent `UNKNOWN`. Valideer deze conservatieve keuze op mixed-size reads.
- Een gestart request reserveert alle resources ook tijdens startup. Na
  startup is het een actieve byteflow. Met `n_r` actieve byteflows door
  resource r krijgt request i rate `min_r(C_r / n_r)` over zijn gehele pad.
  De loadklasse voor `C_r` telt alle issued requests op r, inclusief startup;
  `n_r` telt alleen flows die werkelijk aan de transferfase toe zijn.
  Herverdeel ongebruikte shares niet in deze eerste versie. Zo begrenst ieder
  knelpunt dezelfde flow zonder transfertijd op te tellen of capaciteit te
  vermenigvuldigen. Herbereken rates bij elke dispatch/startup-end/completion.
- Houd voorspelde resterende bytes bij in een apart ledger: trek per verstreken
  eventinterval `rate * dt` af, begrensd op nul; dispatch begint met alle bytes.
  Kopieer dit ledger voor iedere kandidaat. Integreer vooruit tot de volgende
  startup-end of kleinste `remaining_bytes/rate`; handel gelijktijdige events
  in request-ID-volgorde af vóór herberekening. Gebruik dezelfde numerieke
  precisie en tijdafronding in alle fixtures; vergelijk tijden met 1 ns tolerantie.
- Echte completion verwijdert de echte request uit het ledger. Voorspelde nul
  bytes terwijl de echte completion ontbreekt geeft `OVERDUE/UNKNOWN` voor
  betrokken kandidaten; verzin geen vrij resource. Andere onafhankelijke
  groepen blijven planbaar. Gebruik geen latere werkelijke completion uit de
  replay om deze voorspelling achteraf te verbeteren. Deze overdue-regel geldt
  op het werkelijke snapshotmoment. Binnen de gekopieerde toekomstsimulatie
  mogen voorspelde completions uiteraard hypothetische capaciteit vrijgeven;
  zij veranderen nooit de echte counters.
- Weight-ready omvat verwerking: gebruik in de eerste voorspeller één virtuele
  seriële conversieresource, met gekalibreerde duur per formaat/grootte en de
  nu bekende bezetting. Enqueue op voorspelde I/O-completion, ties op request-ID.
  Nul conversieduur mag alleen voor een profiel zonder benodigde verwerking.
  Dit model plant geen echte CPU-taken; ook deze aanname moet held-out kloppen.
- Begrens kandidaten en simulatie-events met de configuratiegrenzen van 7a.
  Budgetuitputting of een niet-positieve rate geeft `UNKNOWN`, geen busy loop.
  Onzekerheidsmarges komen uit vastgelegde kalibratieresiduen en worden niet
  tijdens een kandidaatvergelijking aangepast.

Eerste begrensde policy (geen globale optimizer nodig):

1. Plan alleen bekende demandrequests; prefetch volgt pas in 8b. Sorteer
   wachtende demand op `consumer_need`, daarna enqueue-tijd en request-ID.
   Requests ouder dan een vooraf ingestelde `max_queue_age` gaan eerst in
   enqueue-volgorde. Dit voorkomt policy-starvation bij eindigende reads;
   het is geen garantie bij een defecte of permanent overbelaste bron.
2. Evalueer voor het eerste request alle equivalente bronnen met complete
   profielen. Houd drive én alle gedeelde resources bij. Bereken ook een
   uitstelkandidaat op de eerstvolgende voorspelde resourcecompletion; deze
   reserveert nog niets en wordt bij een echte completion opnieuw beoordeeld.
3. Sluit kandidaten zonder compleet profiel of haalbare harde budgetten op
   hun voorgestelde dispatchmoment uit vóór ranking. Gebruik voor iedere
   overblijvende kandidaat dezelfde scoreverzameling: het nieuwe request plus
   de unie van reeds toegelaten requests die door minstens één kandidaat kunnen
   worden beïnvloed, inclusief conversiequeue-afhankelijkheden. Bepaal die unie
   eenmaal vóór simulatie; verwijder geen requests per kandidaat. Onafhankelijke
   requests buiten die unie hebben geen invloed op de vergelijking. Rangschik op hun som van
   voorspelde consumer-wacht (inclusief dezelfde onzekerheidsmarges).
   Dit is een expliciete lokale proxy, geen bewijs van minimale totale
   DAG-latency. Breek gelijke scores met vroegste weight-ready voor het nieuwe
   request, daarna stabiele drive-ID. Binnen hetzelfde gedeelde knelpunt wint
   daarmee de laagste voorspelde completiontijd als de overige impact gelijk
   is; laagste onbelaste drivelatency is niet het selectiecriterium.
4. Laat alleen nu toe als alle harde resource-/buffergrenzen passen. Past het
   eerste request nergens of wint uitstel, bekijk volgende wachtende requests
   op resources die onafhankelijk zijn van alle kandidaatpaden van dat eerste
   request. Werk nooit buiten de begrensde queue. Na iedere echte toelating
   begint de beoordeling met een nieuw snapshot. Registreer gemiste deadlines
   en plan de beste haalbare kandidaat ook wanneer geen enkele op tijd komt.
5. Herbereken bij enqueue, completion, failure en relevante DAG-readiness.
   Wacht event-driven; geen pollinglus op voorspelde tijdstippen. Uitstel wordt
   niet eindeloos herhaald: voor een age-priority request met nu beschikbare
   capaciteit vervalt de uitstelkandidaat en wint vroegste weight-ready boven
   de somscore. Bij volledig bezette resources wacht het op echte completion.
   Een voorspelde completion is nooit een
   bewijs dat een echte read klaar is of zijn budget vrijgegeven mag worden.

Log per beslissing de snapshot-/profielversie, request/consumer-ID, kandidaten,
resourcepaden, bezetting vóór/na hypothetische toelating, voorspelde ready en
need, onzekerheid, impact op bestaande reads, keuze en afwijs-/uitstelreden.
Koppel later echte dispatch, I/O-completion en weight-ready aan dezelfde ID.
Shadow mag uitsluitend eigen hypothetische counters aanpassen; bestaande
readvolgorde, plaatsing, router en compute blijven gelijk. Houd voorspellingen
naast de werkelijke baseline-observaties, zonder hypothetische completions als
gemeten data te presenteren. Counterfactual winst blijft een voorspelling.

**7d — Verplichte deterministische fixtures vóór echte integratie**

Gebruik een fake monotone clock, vaste servicecurves en een vaste eventqueue;
geen sleeps of timingasserties op een echte drive. Bewaar verwachte keuzes,
completiontijden, counters en besluitredenen als handmatig narekenbare fixtures:

- Twee drives op onafhankelijke resources: A houdt zijn latency bij vier
  reads, B heeft na één een steile toename. A kan vier toelaten; B wordt niet
  gelijkmatig volgepland. Met een bezette A kan B toch de beste kandidaat zijn.
- Vier drives op één controller met een **synthetische** limiet van
  `500 MiB/s`; iedere drive kan die limiet alleen al verzadigen. Vier gelijktijdige
  reads leveren samen geen `2000 MiB/s`. Voor gelijke 100 MiB-reads zonder vaste
  overhead is totale transfertijd voor 400 MiB minstens 0,8 s. Onder gelijke
  overige impact kiest een nieuwe replicated read de laagste voorspelde
  weight-ready tijd die de DAG-deadline haalt; extra spreiding creëert geen
  controllercapaciteit. Voeg een geval toe waarin toelating een kritieke
  bestaande read vertraagt en uitstel daarom wint.
- Zes én zeven drives verdeeld over drie controllers, ongelijke groepsgroottes:
  saturatie van controller 0 beperkt onafhankelijke controllers 1/2 niet.
  Voeg vervolgens een gedeelde upstream-resource toe die groepen wél koppelt.
- Lage onbelaste latency maar hoge actuele contention versus een tragere vrije
  bron; niet-beschikbare replica; geen kandidaat haalt de deadline; stabiele
  tie-break; age-priority onder aanhoudende nieuwe demand; onbekend profiel.
- Admission op een vrij drivebudget maar vol controllerbudget wordt geweigerd.
  Meerdere resourcepaden tellen één gedeelde resource precies één keer.
  Replay met dezelfde snapshots geeft identieke besluiten. Een completion in
  de voorspeller verandert nooit echte readiness of echte occupancy.

**Werkpakketten en stopvoorwaarden voor uitvoerders**

Minimale exacte rekenfixtures voor 7c/7d (alle marges, startup en conversie
zijn nul tenzij vermeld; rates zijn constant; admissionlimieten laten de
genoemde concurrency toe; `t=0` is het snapshot):

| Fixture | Invoer | Verplichte verwachting |
|---|---|---|
| Gedeelde bandwidth | Vier onafhankelijke drives, ieder 500 MiB/s, één controller 500 MiB/s; vier reads van 100 MiB starten tegelijk | Iedere flow krijgt 125 MiB/s; alle vier completen op 0,8 s |
| Kritieke bestaande read | A heeft nog 100 MiB op controller 500 MiB/s, need=0,2 s; nieuwe B heeft 100 MiB op een andere drive van dezelfde controller, need=1 s | B nu: A/B ready=0,4 s, score=0,2 s. B uitstellen tot A: A ready=0,2 s, B=0,4 s, score=0; kies uitstel |
| Onafhankelijke replica | Vorige fixture plus B-replica op vrije tweede controller/drive van 500 MiB/s | A en B ready=0,2 s, score=0; kies tweede controller boven uitstel wegens vroegere B-ready |
| Latency binnen groep | Lege controller 500 MiB/s; één 100 MiB-request, replica X startup=0, Y startup=0,05 s; drives elk 500 MiB/s; need=0,3 s | X ready=0,2 s, Y=0,25 s, beide score=0; kies X |

Leg voor de overige 7d-cases op dezelfde manier concrete curves, need-tijden,
IDs en exacte verwachtingen vast vóór implementatie van de beslisfunctie.

Voer 7a, 7b, 7c en 7d afzonderlijk uit; per pakket bestanden, commando's,
fixtures/resultaten en beperkingen opleveren. 7a/7c/7d mogen eerst met
synthetische profielen worden gebouwd; echte M/P-conclusies vereisen 7b.
Inventarisatie, parserboilerplate, fixture-uitvoering en rapportextractie zijn
geschikt voor Luna. Terra mag bounded implementatie doen volgens dit contract.
Laat de completion-/contentionsemantiek en de uiteindelijke admission/lifetime-
diff door Sol/high of Astra beoordelen volgens `AGENTS.md`. Een zwakker model
mag geen ontbrekende concurrencyregel, hardwarecapaciteit of gate-PASS invullen.
Bij ambiguïteit: lever het concrete falende fixture of ontbrekende contractveld
aan voor review; implementeer geen eigen vervangende schedulerpolicy.

**Gate 7**

- C: shadow aan/uit geeft dezelfde route, reads, plaatsing en output.
- M: alle fixtures van 7d slagen; besluiten gebruiken uitsluitend toen bekende
  informatie. Echte integratietraces tonen drive- én gedeelde-resourcebelasting,
  downstream consumer-need en voorspelde versus echte weight-ready tijden.
- P: rapporteer ready-time-fout, deadline-misses en baselinevergelijking op
  held-out prompts/loadklassen, uitgesplitst naar drive en controllergroep.
  Leg acceptabele voorspellingsfout en onzekerheidsdekking vóór evaluatie vast.
  Niet aanwezige zes-/zevendrivehardware blijft `NOT_RUN`; fixtures bewijzen
  mechanisme, geen hardwarewinst. Zonder zinvolle kalibratie geen actieve planner.

## Stap 8 — begrensde actieve I/O- en residentieplanning

**Briefing**

> Activeer één beleidswijziging tegelijk achter dezelfde readinesscontracten.
> Begin met bronkeuze voor demand reads, voeg daarna begrensde prefetch toe
> en wijzig pas als laatste evictie/plaatsing.

Splits deze stap in drie afzonderlijke gateverslagen:

- **8a bronkeuze:** vaste residentie, geen nieuwe prefetch; vergelijk
  voorspelde completions van equivalente kopieën, niet vaste bytepercentages.
  Gebruik het topologie-/contentioncontract van 7a–7d. Bronkeuze omvat nu
  expliciet toelating en eventueel uitstel op drive én gedeelde resources.
- **8b prefetch:** begrens bytes en in-flight requests; reserveer ruimte voor
  demand. Promoveer bestaande prefetch naar demand zonder dubbele load.
  Prefetch gebruikt dezelfde controller-/upstreambudgetten als demand; een
  apart prefetchbudget creëert geen extra bandwidth. Configureer demandreserve
  per gedeelde resource en laat prefetch alleen toe als voorspelde demand-wacht
  niet stijgt. Annuleer eerst nog niet gestarte prefetch; een al lopende read
  houdt zijn echte capaciteit bezet tot completion, ook na demandpromotie.
- **8c residentie:** optimaliseer verwachte kritieke-padwinst per byte, inclusief
  verplaatsingskosten. Begin binnen bestaande slot-/laaggrenzen. Een globaal
  variabel cachebudget vereist een afzonderlijke audit van alle ecap/ecache-
  allocaties en indexen; introduceer dat niet met alleen een plannerflag.

De huidige PIPE-batchslots zijn geen algemene cross-layer-prefetchopslag.
Gebruik voor vooruitgeladen data afzonderlijke, begrensde ownership en
generaties. Integreer met bestaande pilot/cache-indexering of verklaar die
combinatie expliciet unsupported. Annuleren van een al gestarte read geeft
zijn buffer niet vrij voordat completion vaststaat.

**Admission- en lifetimecontract voor 8a/8b**

- Eén dispatcher bezit de echte resourcecounters en voert alle beslissingen
  uit. Workers melden completion via de bestaande gesynchroniseerde route;
  zij wijzigen geen plannerbudgetten. Geen extra lock-free scheduler bouwen.
- Toelating reserveert drive, alle gedeelde resources en destinationbuffer
  als één dispatchertransactie: alles of niets, vóór publicatie aan I/O.
  Houd in-flight requests én bytes bij; bandwidthcontrole komt daarnaast uit
  het gekalibreerde service-/contentionmodel, niet uit alleen een bytelimiet.
- Statepad: `QUEUED -> RESERVED -> ISSUED -> COMPLETED/FAILED -> RETIRED`.
  Een submitfout mag alleen terugrollen als vaststaat dat niets is uitgegeven;
  bij gedeeltelijke uitgifte eerst alle uitgegeven delen draineren. Het hele
  request blijft daarvoor `ISSUED` met een teller van uitgegeven delen en houdt
  conservatief zijn volledige reservering tot alle delen terminal zijn.
  `QUEUED -> CANCELLED -> RETIRED` heeft geen I/O-reservering vrij te geven.
  `RESERVED -> CANCELLED/FAILED -> RETIRED` geeft de ongebruikte reservering
  precies eenmaal vrij in de dispatcher, zonder op een I/O-event te wachten.
  Na issue is annulering alleen een cancel-pending mark; release gebeurt dan
  precies eenmaal bij de laatste echte terminale I/O-completion. Een per-request
  release-marker voorkomt dubbele cleanup, ook bij submitfout plus reset.
- I/O-resourcebudget en weightbufferlease hebben verschillende eindpunten:
  na I/O-completion kan transfercapaciteit vrij zijn, maar conversie/compute
  mag zijn buffer nog bezitten. Publiceer weight-ready pas na verwerking;
  hergebruik de buffer pas nadat alle consumers zijn geretireerd. Bij failure
  of cancel zonder consumers: geef de buffer vrij zodra geen I/O-deel of
  conversietaak hem nog bezit; publiceer nooit een gedeeltelijk weight-resultaat.
- Duplicate/stale completion wijzigt geen counters of readiness. Bind release
  aan request-ID én generatie, nooit alleen drive, expert-ID of slotpointer.
  Reset/shutdown stopt nieuwe toelating, annuleert queued requests, rolt alle
  unissued reserved requests terug en draint issued reads plus verwerking en
  consumers vóór topologywissel of bufferhergebruik.
- De eerste actieve integratie blijft binnen de bestaande ondersteunde PIPE-
  generatie. Als een bron geen bekende mapping/profiel heeft, kies fallback
  vóór publicatie; draai nooit tegelijk twee onafhankelijke admissionplanners
  voor dezelfde reads. Iedere andere producer op een gedeelde resource moet
  worden meegerekend of expliciet buiten de geteste scope blijven.

**Gate 8a/b/c, telkens opnieuw**

- C: gelijke numerieke output, correct eigenaarschap, budgetgrenzen en
  deduplicatie. Test volle queue, volle cache, foutieve voorspelling, stale
  completion en uitgevallen bron; demand mag niet verhongeren.
- C/M: herhaal 7d tegen de echte admissionadapter met fake I/O-completions.
  Forceer gedeelde-resource-uitputting, submitfout/partiële uitgifte, cancel
  vóór/na issue, dubbele completion, demandpromotie en reset tijdens reads.
  Assert geen overboeking, negatieve counters, dubbele release of vroegtijdig
  bufferhergebruik; na drain zijn alle reserveringen nul. Controleer dat een
  verzadigde groep onafhankelijke groepen niet onbedoeld blokkeert.
- M: heldere beslistraces en geobserveerde completions. Geen evictie van
  in-use weights; geen onzichtbare backlog na reset of prompt-einde.
- P: held-out echte runs tonen lagere tokenlatency of felt wait zonder
  regressie buiten vooraf vastgelegde grenzen. Meer bytes/hits/overlap
  alleen volstaat niet. E:-prefetch blijft experimenteel totdat dit is gemeten.

## Stap 9 — NUMA-ownership

**Briefing**

> Koppel tile-eigenaarschap, workerplaatsing en weightplaatsing op een echte
> NUMA-machine. Vergelijk met de bestaande interleave-route, zonder tegelijk
> de modelprecisie of het aantal resident gemaakte experts te veranderen.

Begin met een vaste partitionering en lokale queues. Kies bewust of
gate/up-intermediates gedeeld, gerepliceerd of overgedragen worden voor down;
tel die kosten mee. Remote work-stealing is begrensd en zichtbaar. Meet
allocatieplaatsing én effectieve threadaffiniteit; de startup-re-exec kan
het launchmasker verruimen. Alleen `COLI_NUMA=1` bewijst geen lokaal ownership.

**Gate 9**

- C: identieke uitkomsten; één-node-fallback correct; lifetime en budgetten
  blijven geldig bij meerdere nodes.
- M: echte NUMA- en affiniteitmetingen tonen de gekozen plaatsing. Een mock
  kan beleid testen maar geen NUMA-performance bewijzen.
- P: één versus twee sockets, dezelfde totale residentie en expliciete
  coreaantallen; vergelijk interleave en ownership. Geen Rome-claim op basis
  van uitsluitend een desktop of Naples. Niet-beschikbare hardware `NOT_RUN`.

## Stap 10 — gecombineerde releasegate

**Briefing**

> Test de afzonderlijke ingrepen en hun combinatie tegen dezelfde baseline.
> Bevorder alleen de daadwerkelijk geteste configuraties; documenteer de
> fallbackmatrix en resterende hypotheses.

Minimale matrix: MSA uit/aan × expert-tasks uit/aan × actieve planner uit/aan,
in korte en lange context en in beperkte en hoge residentie waar haalbaar.
NUMA wordt een aparte as op de doelhost. Test buiten-scope routes zoals
`S>1`, speculatie, GPU en URING expliciet op correcte fallback. Wijzigingen
in gedeelde GLM/M3-kernels vereisen ook de bestaande relevante GLM-tests;
bij ABI-wijzigingen horen de Segment/Edge-gates erbij, niet bij louter docs.

Lever smoke, vaste replay, vrije generatie, langduriger stress en profiling
apart. Resultaten met verschillende numerieke modi, outputlengtes of
residentie worden niet als één snelheidsranglijst gepresenteerd.

**Gate 10**

- C: alle vereiste regressies en gecombineerde lifecyclecases groen.
- M: iedere bedoelde route aantoonbaar actief; fallback en resourcegrenzen
  gedocumenteerd; geen dubbele schedulers of ongecontroleerde nested pools.
- P: vooraf vastgelegde promotiecriteria gehaald met profilerkosten apart.
  Anders opt-in houden of de specifieke ingreep terugnemen zonder ander werk
  te verwijderen. Geen standaardinschakeling op basis van alleen een simulator.

Werk daarna de analyse en het meetrapport bij met nieuwe resultaten,
datum, commit en scope. Behoud eerdere waarnemingen als historische data.
Publicatie, deployment en netwerkuitbreidingen vallen buiten dit plan.

## Direct overdraagbare eerste opdracht

> Voer uitsluitend stap 0 uit van
> `docs/minimax-m3-execution-plan-2026-09-05.md`, inclusief de algemene regels.
> Inspecteer de huidige werkboom en bescherm bestaande fixtures. Leg een
> reproduceerbare baseline vast met tiny-correctheid, een grouped-int4-
> referentie en een vaste-tokenreplay. Gebruik bestaande testdrivers waar
> passend; de engine-exitcode alleen is geen oraclegate. Verander geen
> scheduling, quantisatie of residentiebeleid. Lever het gateverslag met
> letterlijke commando's en bewijsbestanden. Markeer ontbrekende hardware-
> metingen NOT_RUN. Begin niet aan stap 1 voordat gate 0 C/M is geslaagd.
