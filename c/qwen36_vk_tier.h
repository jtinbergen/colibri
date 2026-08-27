/* qwen36_vk_tier.h -- optional Vulkan VRAM expert tier for the qwen36 engine.
 *
 * Reuses the project-wide backend_vulkan.c (already merged via PR #418: "Vulkan
 * backend: expert tier + dense + MLA attention on any Vulkan 1.2 GPU"). On Intel
 * Mesa iGPUs the safe-shader picker selects qmatmul_safe.spv automatically, so
 * the bug-A Mesa subgroup lane-ID cycle does not bite this tier either.
 *
 * Shape mirrors qwen36_tier.h (CUDA tier) exactly so the engine can have one or
 * the other, never both at once on the same GPU:
 *  - init per-model; cap_experts_per_layer matches CUDA's contract.
 *  - vt_ready() gates the engine-side dispatch in the moe() hot loop.
 *  - vt_expert_upload / vt_expert_upload_int4 -- first call uploads; later calls
 *    hit the backend's arena cache and are nearly free.
 *  - vt_moe_run -- synchronous batched dispatch of all K routed experts in one
 *    submit via coli_vk_expert_group; the caller adds the val[kk] weights and
 *    runs the shared expert on the CPU in parallel.
 *
 * Enable with VK=1 build flag (which already gates backend_vulkan.c). When built
 * without VK=1 the inline stubs below keep the engine CPU-only with zero per-token
 * overhead. */
#ifndef QWEN36_VK_TIER_H
#define QWEN36_VK_TIER_H
#include <stdint.h>

#ifdef COLI_VULKAN
#include "backend_vulkan.h"

/* Init after model load. Returns 1 when the tier is active.
 * Same shape as qt_init for symmetry (cap_experts_per_layer must equal n_experts
 * so the backend's arena is sized once). */
int  vt_init(int n_layers, int n_experts, int hidden, int inter,
             int cap_experts_per_layer, int topk, int expert_gs);

/* Tier live? Called from the engine's per-token hot loop. */
int  vt_ready(void);

/* Per-expert upload -- re-runs are no-ops (backend caches via the arena). */
int  vt_expert_upload     (int layer, int slot_idx, int eid,
                            const uint8_t *g, const uint8_t *u, const uint8_t *d,
                            const float *gs, const float *us, const float *ds);
int  vt_expert_upload_int4(int layer, int slot_idx, int eid,
                            const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
                            const float *gs, const float *us, const float *ds);

/* Run the K routed experts for one token in one batched dispatch.
 * idxs[k] is the slot_idx (0 .. cap-1) of expert k; val[k] is the router weight.
 * The expert outputs are accumulated into out[0..hidden-1] with the val[k]
 * weighting already applied (the GPU reduces inside the shader). Returns 1 if the
 * full K-batch ran on GPU; 0 means the engine should fall back to CPU. */
int  vt_moe_run(int layer, const int *idxs, int K,
                const float *val, const float *xs, float *out);

void vt_shutdown(void);

#else /* !COLI_VULKAN: inline stubs, engine stays CPU-only */

static inline int  vt_init(int a,int b,int c,int d,int e,int f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int  vt_ready(void){return 0;}
static inline int  vt_expert_upload    (int a,int b,int c,const uint8_t*d,const uint8_t*e,const uint8_t*f,const float*g,const float*h,const float*i){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;return 0;}
static inline int  vt_expert_upload_int4(int a,int b,int c,const uint8_t*d,const uint8_t*e,const uint8_t*f,const float*g,const float*h,const float*i){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;return 0;}
static inline int  vt_moe_run(int a,const int*b,int c,const float*d,const float*e,float*f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
static inline void vt_shutdown(void){}

#endif /* COLI_VULKAN */
#endif /* QWEN36_VK_TIER_H */
