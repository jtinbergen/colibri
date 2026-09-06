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
| 7 | I/O- en residentieplanner in shadow mode | 1 en 3 C/M; echte integratietrace uit 6 vóór promotie |
| 8 | Begrensde planner actief maken | 6 en 7 C/M |
| 9 | NUMA-eigenaarschap en lokale teams | 6 C/M; aparte A/B met 8 indien actief |
| 10 | Combinatie-, regressie- en promotiegate | 2, 6, 8, 9 voor de ondersteunde scope |

Werk standaard in deze volgorde. Stap 2 en de DAG-route hebben verschillende
inhoudelijke dependencies; test hun individuele flags en hun combinatie.
Een gemiste MSA-performancegate blokkeert niet automatisch de expertanalyse.

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

## Stap 7 — planner in shadow mode

**Briefing**

> Laat een planner voorstellen doen op echte traces zonder reads, plaatsing
> of compute te wijzigen. Voorspel beschikbaarheid en consumer-wacht, geen
> hitrate als vervangende succesmaat.

Hergebruik bruikbare bestaande traces/simulatorcode, maar verifieer hun
geometrie en maak M3-specifieke aannames expliciet. Verzamel kosten per
requestgrootte, bron, queuedepth en compute-resource. Gebruik getrainde
prompts voor kalibratie en aparte prompts voor evaluatie; geen toekomstige
routerkeuze uit de replay gebruiken als online voorkennis.

Voorgesteld besliscontract:

```text
consumer_need = voorspeld tijdstip waarop weights nodig zijn
arrival[source] = nu + geschatte queuewait + read/transfer + benodigde verwerking
slack[source] = consumer_need - arrival[source] - onzekerheidsmarge
```

Voorkom dubbeltelling: een gemeten queued read-latency bevat al queueing;
tel daar niet opnieuw dezelfde queuewait bij op. Kies een bron op verwachte
completion, betrouwbaarheid en budget. Modelleren van prefetch omvat ook
verkeerde predictions, extra bytes, CPU-kosten en verdringing. Een capaciteit
van K expert-slots is geen uniform bytebudget bij verschillende formats.

**Gate 7**

- C: shadow aan/uit geeft dezelfde route, reads, plaatsing en output.
- M: beslissingen zijn reproduceerbaar en bevatten uitsluitend informatie
  die op dat moment beschikbaar was. Synthetische tests omvatten een snelle
  drukke drive, een langzame lege drive en veranderende queuedepth.
- P: voorspellingsfouten en baselinevergelijking op held-out traces; uitsluitend
  als voorspelling rapporteren. Zonder zinvolle kalibratie geen actieve planner.

## Stap 8 — begrensde actieve I/O- en residentieplanning

**Briefing**

> Activeer één beleidswijziging tegelijk achter dezelfde readinesscontracten.
> Begin met bronkeuze voor demand reads, voeg daarna begrensde prefetch toe
> en wijzig pas als laatste evictie/plaatsing.

Splits deze stap in drie afzonderlijke gateverslagen:

- **8a bronkeuze:** vaste residentie, geen nieuwe prefetch; vergelijk
  voorspelde completions van equivalente kopieën, niet vaste bytepercentages.
- **8b prefetch:** begrens bytes en in-flight requests; reserveer ruimte voor
  demand. Promoveer bestaande prefetch naar demand zonder dubbele load.
- **8c residentie:** optimaliseer verwachte kritieke-padwinst per byte, inclusief
  verplaatsingskosten. Begin binnen bestaande slot-/laaggrenzen. Een globaal
  variabel cachebudget vereist een afzonderlijke audit van alle ecap/ecache-
  allocaties en indexen; introduceer dat niet met alleen een plannerflag.

De huidige PIPE-batchslots zijn geen algemene cross-layer-prefetchopslag.
Gebruik voor vooruitgeladen data afzonderlijke, begrensde ownership en
generaties. Integreer met bestaande pilot/cache-indexering of verklaar die
combinatie expliciet unsupported. Annuleren van een al gestarte read geeft
zijn buffer niet vrij voordat completion vaststaat.

**Gate 8a/b/c, telkens opnieuw**

- C: gelijke numerieke output, correct eigenaarschap, budgetgrenzen en
  deduplicatie. Test volle queue, volle cache, foutieve voorspelling, stale
  completion en uitgevallen bron; demand mag niet verhongeren.
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
