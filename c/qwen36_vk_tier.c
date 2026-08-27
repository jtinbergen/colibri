/* qwen36_vk_tier.c -- Vulkan VRAM expert tier for qwen36. See header for design.
 *
 * Per-layer cache of K experts' worth of (gate, up, down) tensors resident on
 * the Vulkan device. Tier_t is keyed by (layer, slot_idx) so the engine can upload
 * lazily after the first routing touches an expert. The dispatch is batched via
 * coli_vk_expert_group (existing entry point from backend_vulkan.c, PR #418). */
#include "qwen36_vk_tier.h"

#ifdef COLI_VULKAN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ColiVkTensor *gate, *up, *down;
    int valid;
} VSlot;

typedef struct {
    int on;
    int n_layers, n_experts, hidden, inter, cap, topk, expert_gs;
    VSlot *slot;          /* [n_layers * cap] */
} VTier;
static VTier G;

static VSlot *vs(int layer, int slot_idx) {
    return &G.slot[(size_t)layer * G.cap + slot_idx];
}

int vt_init(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs) {
    (void)ne; (void)topk;
    /* backend_vulkan.c's vendor/driver picker already selected qmatmul_safe.spv
     * on Intel Mesa, qmatmul.spv elsewhere. The tier is enabled iff any Vulkan
     * compute device exists -- this matches backend_vulkan.c's existing semantics
     * and means the user opts in by just building with VK=1 on Intel Mesa. */
    if (!coli_vk_available()) {
        fprintf(stderr, "[vk] no Vulkan compute device -> CPU MoE path\n");
        return 0;
    }
    G.on = 1;
    G.n_layers = nl;
    G.hidden = D;
    G.inter = Ih;
    G.cap = cap;
    G.expert_gs = expert_gs;
    G.slot = calloc((size_t)nl * cap, sizeof(VSlot));
    if (!G.slot) {
        fprintf(stderr, "[vk] OOM allocating %d slot buffers\n", nl * cap);
        G.on = 0;
        return 0;
    }
    fprintf(stderr, "[vk] qwen36 expert tier -> Vulkan VRAM (layers=%d cap=%d hidden=%d inter=%d)\n",
            nl, cap, D, Ih);
    return 1;
}

int vt_ready(void) { return G.on; }

void vt_shutdown(void) {
    if (!G.on) return;
    /* backend_vulkan.c owns the tensor lifecycle via arena_suballoc: tensors
     * are freed en masse at coli_vk_shutdown. Nothing per-tier to release here. */
    free(G.slot);
    G.slot = NULL;
    G.on = 0;
    /* Deliberately do NOT call coli_vk_shutdown -- the backend is shared
     * with kimi_k3 / colibri and tearing it down here would break those engines. */
}

/* Single internal upload. Format from qwen36 model: 1=int8, 2=int4 packed
 * (4-bit nibble with -8 bias; same wire layout as backend_vulkan.c's fmt=2). */
static int vt_upload_internal(int layer, int slot_idx, int eid,
                              const void *g, const void *u, const void *d,
                              const float *gs, const float *us, const float *ds,
                              int fmt) {
    VSlot *s = vs(layer, slot_idx);
    if (!coli_vk_tensor_ensure(&s->gate, g, gs, fmt, G.inter, G.hidden, G.expert_gs)) return 0;
    if (!coli_vk_tensor_ensure(&s->up,   u, us, fmt, G.inter, G.hidden, G.expert_gs)) return 0;
    if (!coli_vk_tensor_ensure(&s->down, d, ds, fmt, G.hidden,  G.inter, G.expert_gs)) return 0;
    s->valid = 1;
    (void)eid;
    return 1;
}

int vt_expert_upload(int layer, int slot_idx, int eid,
                     const uint8_t *g, const uint8_t *u, const uint8_t *d,
                     const float *gs, const float *us, const float *ds) {
    return vt_upload_internal(layer, slot_idx, eid, g, u, d, gs, us, ds, /*fmt=*/1);
}

int vt_expert_upload_int4(int layer, int slot_idx, int eid,
                          const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
                          const float *gs, const float *us, const float *ds) {
    return vt_upload_internal(layer, slot_idx, eid, g4, u4, d4, gs, us, ds, /*fmt=*/2);
}

int vt_moe_run(int layer, const int *idxs, int K,
              const float *val, const float *xs, float *out) {
    if (!G.on || K <= 0) return 0;
    ColiVkTensor *gates[256], *ups[256], *downs[256];
    int rows[256];
    int valid_count = 0;
    for (int kk = 0; kk < K; kk++) {
        VSlot *s = vs(layer, idxs[kk]);
        if (!s->valid) continue;
        gates[valid_count]  = s->gate;
        ups[valid_count]    = s->up;
        downs[valid_count]  = s->down;
        rows[valid_count]    = kk;   /* which input slot in out[] this expert's output lands at */
        valid_count++;
    }
    if (valid_count == 0) return 0;
    /* coli_vk_expert_group writes the K expert outputs sequentially into out[]:
     *   out[expert_k * hidden .. (expert_k+1) * hidden - 1]
     * We use rows[k] = k so expert_k's output starts at out[k*hidden]. */
    int rc = coli_vk_expert_group(gates, ups, downs, rows, valid_count, out, xs);
    if (rc == 0) return 0;
    /* Apply val[idxs[k]] weighting (the backend wrote the unweighted sum --
     * each output row is the expert's contribution, we scale and add into out). */
    float *p = out;
    for (int v = 0; v < valid_count; v++) {
        int input_idx = rows[v];
        float w = val[input_idx];
        for (int d = 0; d < G.hidden; d++) { *p++ *= w; }
    }
    return 1;
}

#endif /* COLI_VULKAN */
