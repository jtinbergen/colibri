#include "../m3_shadow_store.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int lookups;
    int releases;
    int prefetches;
    int destroyed;
} fake_state_t;

static int fake_lookup(ColiExpertStore *store, ColiExpertKey key,
                       ColiExpertView *view) {
    fake_state_t *s = (fake_state_t *)store->state;
    s->lookups++;
    memset(view, 0, sizeof(*view));
    view->key = key;
    if (key.layer == 99) {
        view->gate.data_bytes = SIZE_MAX;
        view->gate.scale_bytes = 1;
    } else {
        view->gate.data_bytes = 512;
    }
    view->lease = s;
    return 0;
}

static void fake_release(ColiExpertStore *store, ColiExpertView *view) {
    fake_state_t *s = (fake_state_t *)store->state;
    s->releases++;
    memset(view, 0, sizeof(*view));
}

static int fake_prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                         size_t count) {
    (void)keys;
    fake_state_t *s = (fake_state_t *)store->state;
    s->prefetches += (int)count;
    return (int)count;
}

static void fake_stats(const ColiExpertStore *store,
                       ColiExpertStoreStats *stats) {
    const fake_state_t *s = (const fake_state_t *)store->state;
    memset(stats, 0, sizeof(*stats));
    stats->requests = (uint64_t)s->lookups;
    stats->hits = (uint64_t)s->releases;
}

static void fake_destroy(ColiExpertStore *store) {
    fake_state_t *s = (fake_state_t *)store->state;
    s->destroyed++;
    free(s);
    free(store);
}

static int fake_provider(void *opaque, ColiExpertKey key,
                         m3_shd_candidate_t *candidates,
                         size_t *candidate_count, uint64_t *consumer_need_ns) {
    (void)opaque; (void)key;
    if (!candidates || !candidate_count || *candidate_count < 1
        || !consumer_need_ns) return -1;
    candidates[0] = (m3_shd_candidate_t){1, 1, {1}, 1};
    *candidate_count = 1;
    *consumer_need_ns = UINT64_C(1000000000);
    return 0;
}

static ColiExpertStore *make_fake_store(const ColiExpertStoreOps *ops) {
    ColiExpertStore *store = (ColiExpertStore *)calloc(1, sizeof(*store));
    fake_state_t *state = (fake_state_t *)calloc(1, sizeof(*state));
    assert(store && state);
    store->ops = ops;
    store->state = state;
    store->gpu = (void *)(uintptr_t)0x1234;
    return store;
}

int main(void) {
    static const ColiExpertStoreOps ops = {
        fake_lookup, fake_release, fake_prefetch, fake_stats, fake_destroy
    };
    ColiExpertStore *inner = make_fake_store(&ops);
    ColiExpertStore *wrapped = m3_shd_store_wrap(inner, NULL);
    assert(wrapped && m3_shd_store_is_wrapper(wrapped));
    assert(wrapped->gpu == inner->gpu);

    ColiExpertView view;
    assert(coli_expert_lookup(wrapped, (ColiExpertKey){2, 3}, &view) == 0);
    assert(view.key.layer == 2 && view.key.expert == 3);
    coli_expert_release(wrapped, &view);
    assert(wrapped->ops->prefetch(wrapped, NULL, 4) == 4);
    ColiExpertStoreStats stats;
    wrapped->ops->stats(wrapped, &stats);
    assert(stats.requests == 1 && stats.hits == 1);
    wrapped->ops->destroy(wrapped);

    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 10, 4, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 10, M3_SHD_KIND_CONTROLLER, 0, 4, 8192, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[2];
    memset(profiles, 0, sizeof(profiles));
    for (size_t i = 0; i < 2; ++i) {
        profiles[i].resource_id = resources[i].resource_id;
        profiles[i].topology_hash = 1;
        profiles[i].units_per_second_explicit = 1;
        profiles[i].version = M3_SHD_CALIB_VERSION;
        profiles[i].sample_count = 1;
        for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s)
            for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l)
                profiles[i].rate_bytes_per_s[s][l] = 1000000;
    }
    m3_shd_topology_profile_t topology = {1, 2, resources, 0, NULL};
    m3_shd_calibration_set_t calibration = {1, &topology, 2, profiles,
                                            M3_SHD_CALIB_VERSION, {0}, 0};
    m3_shadow_plan_t *planner = m3_shd_open(&topology, &calibration,
                                            (m3_shd_clock_t){0, NULL},
                                            NULL, 0);
    assert(planner);
    m3_shd_candidate_t expanded = {1, 1, {1}, 1};
    assert(m3_shd_expand_candidate(planner, &expanded) == 0);
    assert(expanded.n_resources == 2 && expanded.resources[1] == 10);
    inner = make_fake_store(&ops);
    wrapped = m3_shd_store_wrap_with_provider(inner, planner,
                                               fake_provider, NULL);
    assert(wrapped);
    assert(coli_expert_lookup(wrapped, (ColiExpertKey){3, 4}, &view) == 0);
    coli_expert_release(wrapped, &view);
    size_t log_bytes = 0;
    assert(m3_shd_decision_log(planner, &log_bytes) && log_bytes > 0);
    assert(m3_shd_active_count(planner) == 0);
    assert(m3_shd_queued_count(planner) == 0);
    wrapped->ops->destroy(wrapped);

    inner = make_fake_store(&ops);
    wrapped = m3_shd_store_wrap(inner, planner = m3_shd_open(
        &topology, &calibration, (m3_shd_clock_t){0, NULL}, NULL, 0));
    assert(wrapped && planner);
    assert(coli_expert_lookup(wrapped, (ColiExpertKey){99, 4}, &view) == 0);
    m3_shd_observer_stats_t observer;
    m3_shd_observer_stats(planner, &observer);
    assert(observer.lookup_bytes == UINT64_MAX);
    coli_expert_release(wrapped, &view);
    wrapped->ops->destroy(wrapped);
    puts("m3 shadow store forwarding: PASS");
    return 0;
}
