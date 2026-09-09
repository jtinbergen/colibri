#include "../m3_shadow_plan.h"

#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

static uint64_t now(void *opaque) { return *(uint64_t *)opaque; }

int main(void) {
    const m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 2, UINT64_C(1) << 30, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profile;
    memset(&profile, 0, sizeof(profile));
    profile.resource_id = 1;
    profile.topology_hash = 9;
    profile.units_per_second_explicit = 1;
    profile.version = M3_SHD_CALIB_VERSION;
    profile.sample_count = 1;
    for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s)
        for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l)
            profile.rate_bytes_per_s[s][l] = UINT32_C(100000000);
    const m3_shd_topology_profile_t topology = {
        .topology_hash = 9,
        .n_resources = 1,
        .resources = resources
    };
    const m3_shd_calibration_set_t calibration = {
        9, &topology, 1, &profile, M3_SHD_CALIB_VERSION, {0}, 0
    };
    uint64_t t = 0;
    m3_shadow_plan_t *p = m3_shd_open(&topology, &calibration,
                                      (m3_shd_clock_t){now, &t}, NULL, 0);
    assert(p);
    size_t before = m3_shd_active_count(p);
    ColiExpertKey key = {4, 2};
    m3_shd_observe_lookup_begin(p, &key, 1000);
    m3_shd_observe_lookup_end(p, &key, 0, 1234);
    m3_shd_observe_release(p, &key);
    assert(m3_shd_active_count(p) == before);
    m3_shd_observer_stats_t observer_stats;
    m3_shd_observer_stats(p, &observer_stats);
    assert(observer_stats.lookup_begins == 1);
    assert(observer_stats.lookup_failures == 0);
    assert(observer_stats.lookup_bytes == 1234);
    assert(observer_stats.releases == 1);
    m3_shd_request_t request = {
        .request_id = 7,
        .key = {4, 2},
        .bytes = 1000,
        .consumer_need_ns = UINT64_C(1000000000),
        .demand = 1
    };
    m3_shd_candidate_t candidate = {1, 1, {1}, 1};
    _Atomic int failures = 0;
#pragma omp parallel for
    for (int i = 0; i < 32; ++i) {
        m3_shd_decision_record_t record;
        if (m3_shd_predict(p, &request, &candidate, 1, &record) != M3_SHD_OK)
            atomic_fetch_add_explicit(&failures, 1, memory_order_relaxed);
    }
    assert(atomic_load_explicit(&failures, memory_order_relaxed) == 0);
    assert(m3_shd_active_count(p) == before);
    m3_shd_close(p);
    puts("m3 shadow observer boundary: PASS");
    return 0;
}
