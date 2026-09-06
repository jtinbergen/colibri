# MiniMax-M3: multicore- en multisocketanalyse voor EPYC Rome

Datum: 5 september 2026.

## Conclusie

De huidige Colibri-code laat CPU-parallelisme onbenut dat MiniMax-M3
architectonisch wel biedt. Dat blijkt concreet uit de seriële MSA-indexer
tijdens decode en de opeenvolgende uitvoering van onafhankelijke experts.
De zware matrixbewerkingen zijn al parallel; de engine kan dus meerdere
cores en sockets gebruiken. Hoe ver die schaalbaarheid reikt, en hoeveel
winst een andere taakverdeling oplevert, is nog niet op Rome gemeten.

De belangrijkste vermijdbare beperkingen zitten in de implementatie. M3
vereist geen recurrente Gated DeltaNet-state-update. De indexer en de
expertberekeningen bieden mogelijkheden voor verdere parallellisatie zonder
de modelarchitectuur te veranderen.

Er zijn minstens twee afzonderlijke onderzoeksassen: contextlengte en
residentie. Bij korte context en voldoende residentie zijn expertscheduling
en synchronisatie mogelijke schaalbeperkingen. Bij beperkte residentie kan
I/O overheersen, zoals in de desktopmetingen van vandaag. Bij lange context
komt daar de groeiende seriële MSA-scan bij. De omslagpunten zijn niet
gemeten; deze analyse levert geen betrouwbare tok/s-voorspelling op.

De gezamenlijke roadmap voor I/O, residentie en compute staat verderop in
dit document. Een beschikbare-coretelling of een geslaagde overlapmeting
is op zichzelf geen bewijs van evenredige versnelling.

## Scope en bewijsbasis

- Onderzocht: lokale `colibri-dev`-checkout, branch `dev`, commit `0efa86f`
  (`fix: enable MiniMax-M3 safetensors conversion`), inclusief de aanwezige
  ongecommitte wijzigingen in `c/colibri.c`.
- Model: lokale `minimax_m3_i4`-checkpoint, hidden size 6144, 60 lagen,
  64 queryheads, 4 KV-heads en head dimension 128. De eerste drie lagen
  hebben een dense FFN; de overige 57 hebben 128 routed experts, waarvan
  er vier per token worden gekozen, plus één shared expert.
- De conversiemetadata en een safetensors-header bevestigen grouped int4
  met group size 64 voor de onderzochte routed-experttensors.
- Doel: minimale decode-latency voor één conversatie op meerdere Rome-CPU's,
  onder de aanname dat geheugen en disk de benodigde data kunnen leveren.
- Methode: read-only broncode-inspectie en vergelijking met bestaande lokale
  rapporten. Er zijn voor deze analyse geen Rome-benchmarks uitgevoerd en
  geen enginewijzigingen aangebracht.

De op deze datum opgehaalde
[publieke upstreambron](https://raw.githubusercontent.com/JustVugg/colibri/main/c/colibri.c)
bevatte geen `ARCH_M3` of `attention_gqa`. De M3-conclusies hieronder gelden
daarom voor de lokale checkout, niet automatisch voor upstream `main`.
Bronregelnummers verwijzen naar de lokale werkboom ten tijde van de analyse
en kunnen bij latere wijzigingen verschuiven.

## Huidige uitvoering per gegenereerd token

| Fase | Huidige parallellisatie | Gevolg voor schaalbaarheid |
|---|---|---|
| Q/K/V- en indexprojecties | Matrix-outputrijen verdeeld over OpenMP-threads | Duizenden onafhankelijke rijen, maar de projecties worden afzonderlijk uitgevoerd |
| MSA-contextscores en blokselectie | Eén thread bij single-token decode | Seriële latency die met de context groeit |
| Attention score/softmax/value | Over 64 queryheads | Maximaal 64 nuttige workers in deze lus |
| Vier routed experts | Experts achtereenvolgens; matrixrijen binnen een expert parallel | Herhaalde synchronisatie van het volledige team |
| Shared expert | Na de routed experts | Onafhankelijk werk dat zou kunnen overlappen |
| Volgende laag / volgend gegenereerd token | Afhankelijk van eerdere resultaten | Fundamentele keten van data-afhankelijkheden |

Een seriële volgorde van lagen betekent niet dat de berekening binnen een
laag single-threaded is. Elke laag bevat veel parallel werk, maar de volgende
laag kan pas verder wanneer zijn input beschikbaar is. Gewone autoregressieve
decode kan evenmin het volgende onbekende token alvast volledig berekenen.

Zie [de layer-forwardcode](../c/colibri.c#L7411) en
[de matrixkernels](../c/quant.h#L98).

## MSA: één core scant tijdens decode de context

In [`attention_gqa()`](../c/colibri.c#L4451) staat de parallellisatie buiten
de lus over queryrijen:

```c
#pragma omp parallel for schedule(static) if(S>4)
for (int s=0; s<S; s++) {
    // Scan cached keys, score four index heads, select blocks.
}
```

Normale decode voor één sequentie gebruikt `S=1`. Daardoor scant één thread
op iedere sparse laag de beschikbare context voor alle vier indexheads en
voert die thread vervolgens de blokselectie uit. De AVX2-dotproductkernel
versnelt het werk binnen die core, maar verdeelt het niet over meerdere cores.

Bij 65.536 beschikbare contexttokens bedragen alleen de score-dotproducten:

```text
57 lagen × 65.536 keys × 4 heads × 128 dimensies × 2 operaties
≈ 3,8 miljard floating-pointoperaties per gegenereerd token
```

Dit is een berekening uit de lussen, geen gemeten latency. Selectiewerk en
andere attentionbewerkingen zijn hierin niet opgenomen. Meer sockets
versnellen deze fase niet zolang de implementatie zo blijft.

De scan is wel paralleliseerbaar: verschillende keyblokken kunnen hun scores
onafhankelijk berekenen, gevolgd door top-k-selectie. Een recurrente
state-afhankelijkheid dwingt deze scan niet op één core. Een implementatie
moet de scoreberekening en deterministische tie-breaking behouden.

### Blinde vlek in de profiler

Norm/RoPE en de MSA-scan staan na het afsluiten van `t_aproj` en vóór het
starten van `t_acore`. Ze vallen daardoor wel in de totale attentiontijd,
maar niet in die gedetailleerde subtimers. De profilerregel met
`projection/RoPE` is voor dit pad dus geen volledige meting van RoPE.

Een afzonderlijke MSA-timer is nodig om het seriële aandeel goed vast te
stellen. Zie [de projectietimer](../c/colibri.c#L4432),
[de indexer](../c/colibri.c#L4451) en
[de profieluitvoer](../c/colibri.c#L8352).

## Experts: veel parallel werk, maar ongunstige taakverdeling

De [standaard CPU-expertlus](../c/colibri.c#L6420) voltooit een expert voordat
de volgende begint:

```text
expert 0: parallel gate/up → activation → parallel down
expert 1: parallel gate/up → activation → parallel down
expert 2: ...
expert 3: ...
shared expert: ...
```

Top-four routing beperkt de berekening niet tot vier cores: elke expert
bevat duizenden onafhankelijke outputrijen. Het probleem is dat het hele
team herhaaldelijk door afzonderlijke fasen gaat. Dat levert veel joins op,
waarbij tragere workers de voortgang van het team bepalen. De kosten en het
aandeel daarvan in totale tokenlatency zijn nog niet apart gemeten.

Dit betekent niet dat bij iedere matrixbewerking nieuwe OS-threads worden
aangemaakt. OpenMP-runtimes kunnen workerthreads hergebruiken; Colibri stelt
zelf ook een hot-team/wachtbeleid in. De optimalisatie moet vooral de
werkverdeling, parallelle regio's en synchronisatiegrenzen verbeteren.
Alleen OpenMP vervangen door een permanente eigen threadpool bewijst geen
winst. Zie [het startupbeleid](../c/colibri.c#L11100).

De [shared expert](../c/colibri.c#L6661) gebruikt dezelfde post-attentioninput
als de routed tak. Zijn berekening kan daarom in beginsel overlappen met
routering en routed-expertwerk. Het gewone CPU-pad voert hem erna uit.

### Waarom `XEXP=1` dit voor deze checkpoint niet oplost

De bestaande [XEXP-route](../c/colibri.c#L6229) verwerkt meerdere experts in
één parallelle regio. Daarmee erkent de code al expliciet het probleem van
de vele kleine OpenMP-regio's op brede multisocketmachines. Maar:

- De route vereist onder andere `S==1`, volledige residentie van het blok,
  actieve int4-IDOT bij `S=1`, en ungrouped int4 (`fmt==2`). De onderzochte
  checkpoint gebruikt grouped int4 (`fmt==4`) en wordt uitgesloten.
- De vaste 12 chunks per expert leveren voor M3 maximaal 48 taken per
  matrixfase op. Dat vult geen 128 fysieke cores.
- De tussenliggende activatiequantisatie heeft slechts vier
  expertgebonden taken bij top-four routing.
- De shared expert blijft een afzonderlijke fase.

XEXP is dus een nuttig vertrekpunt voor een optimalisatie, maar geen
bestaande schakelaar die deze checkpoint optimaal over twee grote sockets
verdeelt. Eventuele wijzigingen in activatiequantisatie moeten bovendien
apart numeriek worden gevalideerd.

## Vergelijking met Qwen Gated DeltaNet

De lokale [Qwen GDN-code](../c/qwen36.c#L3397) paralleliseert de recurrente
update al over onafhankelijke valueheads. GDN is niet intrinsiek een
single-corebewerking.

De recurrence legt wel een afhankelijkheid tussen tokenposities op. M3
scoort queries tegen gecachte keys, waardoor werk over heads en
contextblokken kan worden verdeeld. Ook tijdens prefill biedt dat andere
mogelijkheden dan het hier geïmplementeerde recurrente GDN-pad.

Bij gewone autoregressieve decode hebben beide modellen nog steeds het
vorige gegenereerde token nodig. M3 verwijdert die afhankelijkheid niet;
het biedt andere mogelijkheden om het werk binnen een laag te verdelen.
De huidige seriële MSA-scan is een implementatiekeuze, geen equivalent van
een verplichte recurrente update.

## Optimalisatierichting voor EPYC Rome

Begin met een baseline en afzonderlijke MSA- en wachttijdmetingen. Voor de
compute-route zijn daarna de volgende ingrepen te onderzoeken; de gemeten
context- en residentieregimes bepalen de prioriteit:

1. **Paralleliseer MSA over contextblokken.** Gebruik voldoende grote
   werkblokken, behoud deterministische selectie en meet deze fase apart.
2. **Breid expertconcurrency uit naar grouped int4.** Plan werk als
   `(expert, output-row tile)`, neem de shared expert mee en behoud teams
   tussen fasen. Gebruik waar mogelijk socketlokale teams en
   afhankelijkheden per expert in plaats van globale barrières.
3. **Laat geheugenplaatsing overeenkomen met werkeigenaarschap.** Plaats
   gewichtstiles bij hun verwerkende workers en wissel kleine activaties en
   outputreducties uit tussen sockets.
4. **Verminder overige teamwijde joins.** Onafhankelijke Q/K/V- en
   indexprojecties kunnen gezamenlijk worden gepland. Norm/RoPE en
   tijdelijke allocaties verdienen aandacht zodra grotere bottlenecks
   kleiner zijn geworden.

De bestaande [`COLI_NUMA=1`](../c/colibri.c#L1452) interleavet grote
gewichtallocaties over Linux-NUMA-nodes. Dat verdeelt geheugenverkeer, maar
legt geen socketlokaal eigenaarschap van de berekening vast. Ook als de
geheugenbandbreedte voldoende is, blijven cross-socketsynchronisatie en
cachecoherentielatency relevant.

AMD beschrijft de invloed van fabricinstellingen op cross-socketlatency en
de verschillende NUMA-instellingen in de
[Rome Workload Tuning Guide](https://docs.amd.com/api/khub/documents/cfE3LuUQtmpHZxHm15xDyA/content).
Een BIOS- of affiniteitinstelling is daarom een meetvariabele, geen
universele oplossing voor de schedulingproblemen in deze engine.

## Benodigde validatie op de doelmachine

Vergelijk één en twee sockets met expliciete fysieke-coreaantallen, tot het
geïnstalleerde maximum. Neem meerdere contextlengtes mee, bijvoorbeeld 2K,
8K, 32K en 64K, met dezelfde tokens, quantisatie, residentie en decode-instellingen.
Meet prefill en single-sequence decode afzonderlijk.

Maak deze baseline vóór optimalisaties. Hardwarevergelijkingen zijn nu al
bruikbaar om beperkingen te lokaliseren; dezelfde vergelijking na een
wijziging laat zien welke beperking is verminderd. Gebruik vaste
tokenreplays om routes vergelijkbaar te houden, plus een afzonderlijke
generatietest voor correctheid en praktisch gedrag. Leg ook compilerflags,
effectieve threadplaatsing, cache-opwarming en achtergrondbelasting vast.

Registreer ten minste milliseconds per token, afzonderlijke MSA-tijd,
expertcompute, attention-coretijd en wachttijd bij barrières. Scheid een
verbetering in totale throughput door batching van een verbetering in de
latency van één conversatie.

Let bij socketgebonden experimenten op de
[startup-re-exec](../c/colibri.c#L11158): die probeert de CPU-affiniteit naar
alle online CPU's terug te zetten. Controleer de effectieve affiniteit na
startup en stel OpenMP-teams en plaatsing expliciet in; vertrouw niet alleen
op het initiële launchmasker.

De eerdere lokale
[timingrapporten](minimax-m3-timing-report-2026-09-05.md) en het
[uitgebreide rapport](minimax-m3-comprehensive-report-2026-09-05.md) betreffen
een Windows-desktop met beperkte residentie en aanzienlijke expert-I/O.
Die metingen onderbouwen geen voorspelling voor volledig resident draaien
op Rome.

## Gezamenlijke planning van I/O, residentie en compute

De drive-metingen en de codeanalyse wijzen naar een gemeenschappelijk
planningsprobleem: data en uitvoerbare taken moeten op tijd bij de juiste
resource komen. Een beslisregel op alleen diskbandbreedte of alleen
expert-hitrate mist de wachttijd op het kritieke pad.

Het [uitgebreide rapport, sectie 7](minimax-m3-comprehensive-report-2026-09-05.md#7-latencyvergelijking-c-versus-e-en-e-only)
rapporteert bij 1 MiB-reads en vier readers p99-latencies van 0,962 ms op C:
en 7,541 ms op E:. Bij 19 MiB zijn dat 18,275 en 132,982 ms. Dit zijn
read-syscallmetingen onder die specifieke belasting, geen gegarandeerde
expertlaadtijden of algemene bovengrenzen. Vooral de 19 MiB-test met 64 reads
geeft een beperkte steekproef voor de staart van de verdeling.

De planningsvraag is daarom: welke beschikbare bron levert de benodigde
expert waarschijnlijk het eerst, gegeven wachtrij, requestgrootte,
residentie en het verwachte moment van gebruik? E: eerder inzetten voor
prefetch is een te testen mogelijkheid. De metingen bewijzen nog niet dat
die strategie end-to-end winst oplevert.

### Eén afhankelijkhedengraaf, uitvoering dicht bij de resources

Een toekomstige planner kan de afhankelijkheden van opslag tot compute
delen met de executor, zonder iedere kleine taak door één centrale queue
te sturen:

```text
voorspelde expert ──→ optionele prefetch ──→ weights beschikbaar
                                             │
exacte routerkeuze + geldige input + weights ──┴─→ gate/up-tiles
                                                   ↓
                                       volledige activation gereed
                                                   ↓
                                             down-tiles
                                                   ↓
                                            expertoutput

geldige input + shared weights ──→ shared expert ──→ shared output

alle vereiste outputs ──→ gewogen samenvoeging + residual ──→ volgende laag
```

Een residentie-hit kan direct aan `weights beschikbaar` voldoen. Een
verkeerde voorspelling mag geen extra expertbijdrage toevoegen. De exacte
router bepaalt welke routed outputs werkelijk nodig zijn.

De planner bepaalt plaatsing, prefetch en verwachte aankomsttijden. De
executor voert daadwerkelijk gereed werk uit en bewaakt dependencies en
de levensduur van buffers. Metingen van beide voeden de volgende beslissingen.
Socketlokale queues en grovere globale plaatsingsbeslissingen zijn een
ontwerprichting om een nieuwe centrale synchronisatiebottleneck te vermijden.

Als A nog wordt geladen, kunnen een gereedstaande B en de shared expert
intussen rekenen. De huidige geordende expertlus kan daarentegen bij de
`pipe_wait()` voor A blijven staan voordat hij aan een latere B toekomt.
Hoe vaak dit voorkomt en hoeveel tijd het kost, moet een trace aantonen.

Niet iedere dependency kan worden verwijderd. Een gewone down-outputrij
gebruikt de volledige intermediate vector van zijn expert. Eerder beginnen
vereist splitsing over de inputdimensie, partial sums en extra reducties.
Begin daarom met onafhankelijkheid tussen experts en outputrijen. Geef
expertoutputs eigen buffers en behoud de oorspronkelijke samenvoegvolgorde
als tokenexactheid vereist is; voltooiingsvolgorde mag die niet stil wijzigen.

Een interne deadline is het voorspelde tijdstip waarop een consumer de data
nodig heeft. Slack is het verschil tussen dat tijdstip en de verwachte
aankomst, inclusief wachtrij en onzekerheidsmarge. Het zijn planningsschattingen,
geen harde garanties. Noodzakelijke reads moeten voorrang kunnen krijgen op
speculatieve prefetch. Prefetch mag geen nog gebruikte weights verdringen of
buffers hergebruiken terwijl I/O of compute ze nog bezit.

## Hardware als onderzoeksplatform

De 12600K-desktop heeft de I/O- en overlapmetingen geleverd. Voor dual
Haswell, Naples 7551 en dual Rome zijn in deze rapporten geen vergelijkbare
M3-schaalcurves beschikbaar. Hun relatieve snelheid kan daarom niet uit de
desktop-tok/s of het aantal cores worden afgeleid.

Een Naples 7551 met 32 cores is een plausibel platform om threadschaling,
taakgrootte en NUMA-plaatsing te onderzoeken. Dat is een onderzoekshypothese,
geen bewijs dat de huidige engine al vóór 32 cores een specifiek plafond
raakt. De seriële MSA-scan schaalt zelf niet mee met het coreaantal; hoe zwaar
dat op deze CPU weegt, vereist een contextafhankelijke meting. Claims over
Naples versus Rome moeten daarnaast de concrete machineconfiguratie en
gemeten kernels meenemen.

Dual Rome is bruikbaar als verder schaaldoel én voor een baseline van de
huidige matrixkernels. Het is niet nodig om eerst alle voorgestelde
optimalisaties af te hebben voordat een tweede socket nuttig kan zijn.
Evenmin bewijst onbenut parallelisme dat 128 fysieke cores na die ingrepen
volledig of efficiënt bezet zullen zijn.

## Vier samenhangende werkpakketten

Het [uitvoeringsplan met briefings en control gates](minimax-m3-execution-plan-2026-09-05.md)
werkt deze pakketten uit in afzonderlijke implementatiestappen, met
correctheidsbewijs, mechanismechecks en performancecriteria per stap.

De gedeelde startvoorwaarde is een reproduceerbare baseline met aparte
prefill/decode, contextlengte, coreaantal, plaatsing en residentie. Onderstaande
statussen beschrijven het bewijs van vandaag, niet al opgeleverde features.

| Werkpakket | Status op 5 september | Acceptatiecriterium |
|---|---|---|
| PIPE=1 en I/O-overlap | De gerapporteerde 32-tokenproeven voltooiden zonder hang en tonen I/O/compute-overlap | Gate geslaagd binnen deze proeven; versnelling tegenover PIPE=0 vereist nog een gelijkwaardige A/B |
| Globale residentie en I/O-planning op verwachte aankomsttijd | Ontwerprichting, ondersteund door drive-latency- en miss-waitmetingen | Minder gevoelde I/O-wachttijd en lagere tokenlatency; kosten van verkeerde prefetch, verdringing en queueing meenemen |
| MSA apart profileren en over contextblokken paralleliseren | Seriële decode-scan en ontbrekende subtimer vastgesteld in code | Correcte scores/selectie behouden; lagere MSA-tijd én end-to-endlatency bij relevante contextlengtes |
| Expert-tasks voor grouped-int4 en NUMA-eigenaarschap | Sequentiële expertlus en ongeschikte XEXP-gate vastgesteld in code | Correcte activaties/output, veilige bufferlevensduur, minder wachten en betere gemeten schaalcurve |

De residentieplanner en executor moeten dezelfde gereedheid en deadlines
kunnen uitwisselen. Dat verplicht niet tot één grote implementatiesprong:
meet eerst per-expert gereedheid, wachttijd en uitvoering, wijzig daarna één
beleidskeuze tegelijk en herhaal dezelfde baseline. Meer overlap, meer
CPU-bezetting en een hogere hitrate tellen alleen als winst wanneer de
gekozen latency- of throughputmaat ook verbetert.

## Als meerdere processors meerdere servers betekent

Er is een extra beperking in
[`cluster_moe_batch()`](../c/colibri.c#L3326): de coordinator stuurt werk naar
één worker en ontvangt diens resultaten voordat hij de volgende worker
afhandelt. Dit is seriële dispatch over workers. Deze route heeft eerst
concurrente dispatch en resultaatverzameling nodig om meerdere servers
tegelijk effectief te gebruiken.

Dat staat los van OpenMP-schaalbaarheid binnen één machine. De huidige
clusterroute is geen bewijs dat M3-decode al efficiënt over meerdere
Rome-servers kan worden verdeeld.
