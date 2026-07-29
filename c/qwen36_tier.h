/* qwen36_tier.h — M-QTIER (R2): VRAM-Experten-Tier für qwen36 auf Basis des
 * Upstream-CUDA-Backends (backend_cuda.cu / coli_cuda_*).
 *
 * Konzept (Colibrì „route → union → place → overlap → learn", RAM→VRAM-Stufe):
 *  - Jeder Experte hat ein Heimat-Device (eid % n_gpus), keine Duplikate (E1/E3).
 *  - Routing-Hitze (Heat) entscheidet, wer VRAM verdient; M2 = Warmup-Füllung
 *    bis Budget, M3 ergänzt LFRU-Swaps + Prefetch.
 *  - Uploads laufen in einem Hintergrund-Thread über Staging-Kopien; das
 *    Decodieren blockiert nie auf einen Upload (E2).
 *  - VRAM-Miss ⇒ Aufrufer rechnet den Experten auf der CPU, überlappend mit
 *    den GPU-Gruppen (qt_issue … CPU … qt_take).
 *
 * Aktivierung: COLI_CUDA=1 [COLI_GPUS=0,1] [CUDA_EXPERT_GB=<G>|auto]
 */
#ifndef QWEN36_TIER_H
#define QWEN36_TIER_H
#include <stdint.h>

/* Init nach Modell-Load. Gibt 1 zurück, wenn der Tier aktiv ist.
 * cap_experts_per_layer muss == n_experts sein (volle RAM-Residenz, Z5);
 * sonst bleibt der Tier aus (Upload-Zeiger könnten sonst evicted werden). */
int  qt_init(int n_layers, int n_experts, int hidden, int inter,
             int cap_experts_per_layer, int topk);
int  qt_ready(void);
void qt_shutdown(void);

/* Pro geroutetem Experten einmal je Token aufrufen (Zeiger auf die RAM-Slots,
 * int4-packed + per-row-Scales). Aktualisiert Heat und stößt ggf. einen
 * Hintergrund-Upload an (nicht blockierend). */
void qt_note(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);

/* Startet die GPU-Gruppen für die residenten der K Experten (asynchron, beide
 * Devices parallel). Rückgabe: Bitmaske der k, die die GPU übernimmt.
 * Danach: Misses auf der CPU rechnen, dann qt_take() mit derselben Maske. */
uint32_t qt_issue(int layer, const int *eids, int K, const float *x);

/* Sammelt die GPU-Ergebnisse ein und akkumuliert val[k]*y_k in out[hidden]. */
void qt_take(uint32_t mask, const float *val, int K, float *out);

/* Telemetriezeile (stderr): Residenz, Hits/Misses, Uploads je Device. */
void qt_stats(void);

#endif
