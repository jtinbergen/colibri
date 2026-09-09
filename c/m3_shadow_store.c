#include "m3_shadow_store.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    ColiExpertStore *inner;
    m3_shadow_plan_t *planner;
    m3_shd_snapshot_provider_fn provider;
    void *provider_opaque;
} m3_shd_store_state_t;

static m3_shd_store_state_t *state_of(ColiExpertStore *store) {
    return store ? (m3_shd_store_state_t *)store->state : NULL;
}

static uint64_t view_bytes(const ColiExpertView *view) {
    if (!view) return 0;
    uint64_t total = 0;
    const ColiTensorView *parts[] = {&view->gate, &view->down, &view->up};
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); ++i) {
        uint64_t data_bytes = (uint64_t)parts[i]->data_bytes;
        uint64_t scale_bytes = (uint64_t)parts[i]->scale_bytes;
        if (UINT64_MAX - data_bytes < scale_bytes) return UINT64_MAX;
        uint64_t part = data_bytes + scale_bytes;
        if (UINT64_MAX - total < part) return UINT64_MAX;
        total += part;
    }
    return total;
}

static int shadow_lookup(ColiExpertStore *store, ColiExpertKey key,
                         ColiExpertView *view) {
    m3_shd_store_state_t *state = state_of(store);
    if (!state || !state->inner || !state->inner->ops
        || !state->inner->ops->lookup) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    m3_shd_candidate_t candidates[M3_SHD_MAX_RESOURCES];
    size_t candidate_count = M3_SHD_MAX_RESOURCES;
    uint64_t consumer_need_ns = 0;
    int snapshot_ok = state->provider && state->planner
        && state->provider(state->provider_opaque, key, candidates,
                           &candidate_count, &consumer_need_ns) == 0;
    if (snapshot_ok && (candidate_count == 0
                        || candidate_count > M3_SHD_MAX_RESOURCES)) {
        snapshot_ok = 0;
    }
    if (snapshot_ok) {
        for (size_t i = 0; i < candidate_count; ++i) {
            if (m3_shd_expand_candidate(state->planner, &candidates[i]))
                candidates[i].available = 0;
        }
    }
    m3_shd_observe_lookup_begin(state->planner, &key, 0);
    int result = state->inner->ops->lookup(state->inner, key, view);
    uint64_t bytes = result ? 0 : view_bytes(view);
    m3_shd_observe_lookup_end(state->planner, &key, result, bytes);
    if (!result && snapshot_ok && bytes) {
        m3_shd_request_t request;
        memset(&request, 0, sizeof(request));
        request.key = key;
        request.bytes = bytes;
        request.consumer_need_ns = consumer_need_ns;
        request.demand = 1;
        uint64_t request_id = 0;
        /* Queue only to obtain a stable request ID.  Removing it immediately
         * keeps this observation shadow-only; no real resource is reserved. */
        if (m3_shd_enqueue(state->planner, &request, &request_id)) {
            request.request_id = request_id;
            m3_shd_decision_record_t decision;
            m3_shd_decide(state->planner, &request, candidates,
                          candidate_count, &decision);
            m3_shd_remove_queued(state->planner, request_id);
        }
    }
    return result;
}

static void shadow_release(ColiExpertStore *store, ColiExpertView *view) {
    m3_shd_store_state_t *state = state_of(store);
    if (!state || !state->inner || !state->inner->ops
        || !state->inner->ops->release) {
        if (view) memset(view, 0, sizeof(*view));
        return;
    }
    ColiExpertKey key = view ? view->key : (ColiExpertKey){0, 0};
    state->inner->ops->release(state->inner, view);
    m3_shd_observe_release(state->planner, &key);
}

static int shadow_prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                           size_t count) {
    m3_shd_store_state_t *state = state_of(store);
    if (!state || !state->inner || !state->inner->ops
        || !state->inner->ops->prefetch) return 0;
    return state->inner->ops->prefetch(state->inner, keys, count);
}

static void shadow_stats(const ColiExpertStore *store,
                         ColiExpertStoreStats *stats) {
    const m3_shd_store_state_t *state =
        store ? (const m3_shd_store_state_t *)store->state : NULL;
    if (!state || !state->inner || !state->inner->ops
        || !state->inner->ops->stats) return;
    state->inner->ops->stats(state->inner, stats);
}

static void shadow_destroy(ColiExpertStore *store) {
    m3_shd_store_state_t *state = state_of(store);
    if (!state) {
        free(store);
        return;
    }
    ColiExpertStore *inner = state->inner;
    m3_shadow_plan_t *planner = state->planner;
    free(state);
    free(store);
    if (planner) m3_shd_close(planner);
    if (inner && inner->ops && inner->ops->destroy)
        inner->ops->destroy(inner);
}

static const ColiExpertStoreOps shadow_ops = {
    shadow_lookup, shadow_release, shadow_prefetch, shadow_stats, shadow_destroy
};

ColiExpertStore *m3_shd_store_wrap(ColiExpertStore *inner,
                                   m3_shadow_plan_t *planner) {
    return m3_shd_store_wrap_with_provider(inner, planner, NULL, NULL);
}

ColiExpertStore *m3_shd_store_wrap_with_provider(
    ColiExpertStore *inner, m3_shadow_plan_t *planner,
    m3_shd_snapshot_provider_fn provider, void *provider_opaque) {
    if (!inner || !inner->ops) return NULL;
    ColiExpertStore *store = (ColiExpertStore *)calloc(1, sizeof(*store));
    m3_shd_store_state_t *state =
        (m3_shd_store_state_t *)calloc(1, sizeof(*state));
    if (!store || !state) {
        free(state);
        free(store);
        return NULL;
    }
    state->inner = inner;
    state->planner = planner;
    state->provider = provider;
    state->provider_opaque = provider_opaque;
    store->ops = &shadow_ops;
    store->state = state;
    store->gpu = inner->gpu;
    return store;
}

int m3_shd_store_is_wrapper(const ColiExpertStore *store) {
    return store && store->ops == &shadow_ops;
}
