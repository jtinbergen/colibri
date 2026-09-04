/* qwen36_tier.h -- optional CUDA VRAM expert tier for the qwen36 engine.
 *
 * Applies colibri's placement concept ("route -> place -> overlap -> learn")
 * one level up from the GLM disk tier: experts live in RAM, the *hot* ones
 * are promoted into DEVICE_LOCAL VRAM across one or more GPUs and computed
 * there via the existing CUDA backend (backend_cuda.cu, expert-group API).
 *
 *  - Every expert has one home device (eid % n_gpus), no duplicates.
 *  - Routing heat decides who earns VRAM (LFRU semantics from tier.h, with
 *    hysteresis); a warmstart pre-fills the budget before the first token,
 *    ordered by a persisted heat table (HEAT_FILE) when available.
 *  - Uploads run on a background thread through staging copies; decode never
 *    blocks on placement. A VRAM miss falls back to the CPU int8 path and
 *    overlaps with the in-flight GPU groups.
 *
 * Enable with COLI_CUDA=1 [COLI_GPUS=0,1] [CUDA_EXPERT_GB=<G>|auto]
 * [HEAT_FILE=<path>] [QT_NO_WARMSTART=1]. Compiled only when the build sets
 * -DCOLI_CUDA (CUDA=1); otherwise the inline stubs below keep the engine
 * CPU-only with zero overhead. */
#ifndef QWEN36_TIER_H
#define QWEN36_TIER_H
#include <stdint.h>

#ifdef COLI_CUDA

/* Init after model load. Returns 1 when the tier is active.
 * cap_experts_per_layer must equal n_experts (full RAM residency): the tier
 * stores raw pointers into the expert slots, which must never be evicted. */
/* expert_is_int4: 1 = pesi int4 impacchettati (fmt=4), 0 = int8 (fmt=1). Il
 * chiamante lo determina dalla TAGLIA SU DISCO, non da meta.ebits, che su
 * qualche container mente (cfr. il rilevamento in qwen36.c). */
int  qt_init(int n_layers, int n_experts, int hidden, int inter,
             int cap_experts_per_layer, int topk, int expert_gs,
             int expert_is_int4);
int  qt_ready(void);
int  qt_is_resident(int layer, int eid);
void qt_shutdown(void);

typedef struct {
    int home_gpu, resident_gpu, ram_node, disk_replica, disk_controller;
    uint8_t resident, queued;
} QtExpertLocation;
int qt_expert_location(int layer, int eid, QtExpertLocation *out);

/* Call once per routed expert per token (pointers to the RAM slot: packed
 * int4 + per-row scales). Updates heat and may enqueue a background upload. */
void qt_note(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);

/* Launch the GPU groups for the resident subset of the K selected experts
 * (async, all devices in parallel). Returns a bitmask of the k handled by
 * the GPU. Compute the misses on the CPU, then call qt_take(). */
uint32_t qt_issue(int layer, const int *eids, int K,
                  const float *weights, const float *x);

/* Collect the GPU results and accumulate val[k]*y_k into out[hidden]. */
void qt_take(uint32_t mask, const float *val, int K, float *out);

/* Cumulative resident-take diagnostic counters. Values are microseconds;
 * GPU/reduction are CUDA-event durations, while the remaining fields are
 * host-wall spans. The counters are only populated when COLI_TIMERS=1. */
void qt_resident_timing_totals(double *gpu_us, double *sync_wait_us,
                               double *take_api_us, double *reduce_us,
                               double *d2h_us, double *take_total_us,
                               uint64_t *take_calls, double *qt_take_us,
                               uint64_t *qt_take_calls);
void qt_resident_timing_scope(int decode);
void qt_resident_timing_reset(void);
void qt_resident_timing_call_report(void);
void qt_resident_timing_take_layers(double *take_us, uint64_t *calls, int n);
/* Metrics for the most recent qt_take call, in microseconds. */
void qt_resident_timing_last(double *gpu_us, double *sync_us,
                             double *take_wall_us, int *gpu_valid);
/* Same-host-clock bounds, populated only with COLI_CUDA_TIMELINE=1. Values
 * are monotonic milliseconds. */
void qt_resident_timing_last_host(double *gpu_lower_ms, double *gpu_upper_ms,
                                  double *reduce_lower_ms, double *reduce_upper_ms);
/* Lightweight host-clock boundaries for Compute Islands v0.  These are
 * populated only for an active resident GPU call while island timing is on. */
void qt_resident_timing_last_boundaries(double *gpu_complete_ms,
                                        double *merge_begin_ms);

/* Bounded prefill form. eids/val are flattened in row-major route order;
 * x is [rows,hidden], rows<=8 and routes<=64. The mask has one bit per route
 * entry and qt_take_batch accumulates resident GPU results into out. */
uint64_t qt_issue_batch(int layer, const int *eids, int routes,
                        const float *x, int rows);
void qt_take_batch(uint64_t mask, const float *val, int routes,
                   float *out, int rows);

/* Warmstart: plan the full fill set (heat order, budget reserved), then any
 * number of loader threads may call qt_note_planned per planned expert. */
int  qt_plan_fill(int *layers, int *eids, int max);
void qt_note_planned(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);
int  qt_fill_next(int *layer, int *eid);
void qt_note_block(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);
void qt_fill_wait(void);   /* blocks until the upload queue is drained */

/* One telemetry block on stderr: residency, hits/misses, uploads per device. */
void qt_stats(void);
/* Loader attribution hook. replica is the actual source selected by the
 * storage layer (currently primary=0 for Qwen reads); the tier only records
 * the observation and never chooses a disk. */
void qt_record_storage_read(int replica, uint64_t bytes, uint64_t busy_ns);
void qt_record_storage_read_path(const char *path, uint64_t bytes, uint64_t busy_ns);
void qt_set_expert_storage(int layer, int eid, const char *path);
void qt_set_expert_storage_source(int layer, int eid, const char *path, int replica);
/* Bind a long-lived resident expert arena using the active NUMA policy.
 * Returns 1 only when Linux actually applied interleaving. */
int qt_numa_bind_arena(void *base, size_t bytes);

/* Coarse dense-island prototype. These calls do not alter routing or
 * residency: they register the already-created dense int8 copies and submit
 * the two DeltaNet projections as one island-local batch. */
enum { QT_DENSE_DN_QKV = 1, QT_DENSE_DN_Z = 2, QT_DENSE_DN_OUT = 3 };
enum { QT_DENSE_ATTN_Q = 4, QT_DENSE_ATTN_K = 5, QT_DENSE_ATTN_V = 6, QT_DENSE_ATTN_O = 7,
       QT_DENSE_SHARED_G = 8, QT_DENSE_SHARED_U = 9, QT_DENSE_SHARED_D = 10 };
int qt_dense_init(int n_layers);
int qt_dense_register(int layer,int kind,const int8_t *weights,
                      const float *scales,int I,int O);
int qt_dense_deltanet_proj(int layer,const float *x,float *qkv,float *z);
int qt_dense_register_deltanet_full(int layer,
        const float *conv,const float *b,const float *a,
        const float *dtbias,const float *alog,const float *norm,
        const float *rec,const float *ring,
        int vheads,int kheads,int kdim,int vdim,int convk,int conv_dim,float eps);
int qt_dense_deltanet_full(int layer,const float *x,float *out);
/* Opt-in state-only DeltaNet island. qkv/z/b/a stay on the existing CPU
 * projection path; the backend returns normalized value rows and the caller
 * applies the established CPU out projection. */
int qt_dense_deltanet_state(int layer,const float *qkv,const float *z,
                            const float *b,const float *a,float *norm_out);
int qt_dense_register_lm_head(const int8_t *weights,const float *scales,int I,int O);
int qt_dense_lm_head(const float *x,float *logits,int O);
int qt_dense_attention_proj(int layer,const float *x,float *q,float *k,float *v);
int qt_dense_attention_out(int layer,const float *x,float *out);
int qt_dense_shared(int layer,const float *x,float *out);
void qt_dense_stats(void);

#else /* !COLI_CUDA: inline stubs, engine stays CPU-only */

static inline int  qt_init(int a,int b,int c,int d,int e,int f,int g,int h){(void)h;(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int  qt_ready(void){return 0;}
static inline int  qt_is_resident(int a,int b){(void)a;(void)b;return 0;}
static inline void qt_shutdown(void){}
typedef struct {
    int home_gpu, resident_gpu, ram_node, disk_replica, disk_controller;
    uint8_t resident, queued;
} QtExpertLocation;
static inline int qt_expert_location(int a,int b,QtExpertLocation*c){(void)a;(void)b;(void)c;return 0;}
static inline void qt_note(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline uint32_t qt_issue(int a,const int*b,int c,const float*d,const float*e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline void qt_take(uint32_t a,const float*b,int c,float*d){(void)a;(void)b;(void)c;(void)d;}
static inline void qt_resident_timing_totals(double*a,double*b,double*c,double*d,double*e,double*f,uint64_t*g,double*h,uint64_t*i){
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;
}
static inline void qt_resident_timing_scope(int a){(void)a;}
static inline void qt_resident_timing_reset(void){}
static inline void qt_resident_timing_call_report(void){}
static inline void qt_resident_timing_take_layers(double*a,uint64_t*b,int c){(void)a;(void)b;(void)c;}
static inline void qt_resident_timing_last(double*a,double*b,double*c,int*d){(void)a;(void)b;(void)c;(void)d;}
static inline void qt_resident_timing_last_host(double*a,double*b,double*c,double*d){(void)a;(void)b;(void)c;(void)d;}
static inline void qt_resident_timing_last_boundaries(double*a,double*b){(void)a;(void)b;}
static inline uint64_t qt_issue_batch(int a,const int*b,int c,const float*d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline void qt_take_batch(uint64_t a,const float*b,int c,float*d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;}
static inline int  qt_plan_fill(int*a,int*b,int c){(void)a;(void)b;(void)c;return 0;}
static inline void qt_note_planned(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline int  qt_fill_next(int*a,int*b){(void)a;(void)b;return 0;}
static inline void qt_note_block(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline void qt_fill_wait(void){}
static inline void qt_stats(void){}
static inline void qt_record_storage_read(int a,uint64_t b,uint64_t c){(void)a;(void)b;(void)c;}
static inline void qt_record_storage_read_path(const char*a,uint64_t b,uint64_t c){(void)a;(void)b;(void)c;}
static inline void qt_set_expert_storage(int a,int b,const char*c){(void)a;(void)b;(void)c;}
static inline void qt_set_expert_storage_source(int a,int b,const char*c,int d){(void)a;(void)b;(void)c;(void)d;}
static inline int qt_numa_bind_arena(void*a,size_t b){(void)a;(void)b;return 0;}
enum { QT_DENSE_DN_QKV = 1, QT_DENSE_DN_Z = 2, QT_DENSE_DN_OUT = 3 };
enum { QT_DENSE_ATTN_Q = 4, QT_DENSE_ATTN_K = 5, QT_DENSE_ATTN_V = 6, QT_DENSE_ATTN_O = 7,
       QT_DENSE_SHARED_G = 8, QT_DENSE_SHARED_U = 9, QT_DENSE_SHARED_D = 10 };
static inline int qt_dense_init(int a){(void)a;return 0;}
static inline int qt_dense_register(int a,int b,const int8_t*c,const float*d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
static inline int qt_dense_deltanet_proj(int a,const float*b,float*c,float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline int qt_dense_register_deltanet_full(int a,const float*b,const float*c,const float*d,const float*e,const float*f,const float*g,const float*h,const float*i,int j,int k,int l,int m,int n,int o,float p){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;(void)p;return 0;}
static inline int qt_dense_deltanet_full(int a,const float*b,float*c){(void)a;(void)b;(void)c;return 0;}
static inline int qt_dense_deltanet_state(int a,const float*b,const float*c,const float*d,const float*e,float*f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
static inline int qt_dense_register_lm_head(const int8_t*a,const float*b,int c,int d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline int qt_dense_lm_head(const float*a,float*b,int c){(void)a;(void)b;(void)c;return 0;}
static inline int qt_dense_attention_proj(int a,const float*b,float*c,float*d,float*e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int qt_dense_attention_out(int a,const float*b,float*c){(void)a;(void)b;(void)c;return 0;}
static inline int qt_dense_shared(int a,const float*b,float*c){(void)a;(void)b;(void)c;return 0;}
static inline void qt_dense_stats(void){}

#endif /* COLI_CUDA */
#endif /* QWEN36_TIER_H */
