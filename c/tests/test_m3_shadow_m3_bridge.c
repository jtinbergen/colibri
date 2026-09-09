#include "../m3_shadow_m3_bridge.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_translation(void) {
    m3_shd_m3_load_snapshot_t snapshot = {
        .model_id = 9,
        .forward_id = 17,
        .generation = 3,
        .consumer_node = 42,
        .request_id = 8,
        .consumer_need_ns = UINT64_C(900000000),
        .key = {4, 11},
        .bytes = UINT64_C(100000000),
        .drive_resource_id = 7,
        .demand = 1
    };
    m3_shd_request_t request;
    m3_shd_candidate_t candidate;
    assert(m3_shd_m3_make_request(&snapshot, &request, &candidate) == 0);
    assert(request.request_id == 8 && request.model_id == 9
           && request.key.layer == 4
           && request.key.expert == 11);
    assert(request.bytes == UINT64_C(100000000)
           && request.consumer_need_ns == UINT64_C(900000000));
    assert(request.forward_id == 17 && request.generation == 3
           && request.consumer_node == 42 && request.demand == 1);
    assert(candidate.candidate_id == 7 && candidate.n_resources == 1
           && candidate.resources[0] == 7 && candidate.available);
}

static void test_rejects_incomplete_snapshot(void) {
    m3_shd_m3_load_snapshot_t snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    m3_shd_request_t request;
    m3_shd_candidate_t candidate;
    assert(m3_shd_m3_make_request(&snapshot, &request, &candidate) != 0);
    snapshot.demand = 1;
    assert(m3_shd_m3_make_request(&snapshot, &request, &candidate) != 0);
    snapshot.model_id = 1;
    assert(m3_shd_m3_make_request(&snapshot, &request, &candidate) != 0);
}

static void test_planner_boundary_expands_path(void) {
    const uint64_t topology_hash = UINT64_C(91);
    const m3_shd_resource_t resources[] = {
        { 7, M3_SHD_KIND_DRIVE, 70, 4, UINT64_C(400) * 1024 * 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 70, M3_SHD_KIND_CONTROLLER, 700, 4, UINT64_C(400) * 1024 * 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 700, M3_SHD_KIND_UPSTREAM, 0, 8, UINT64_C(800) * 1024 * 1024, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[3];
    memset(profiles, 0, sizeof(profiles));
    for (size_t i = 0; i < 3; ++i) {
        profiles[i].resource_id = resources[i].resource_id;
        profiles[i].topology_hash = topology_hash;
        profiles[i].units_per_second_explicit = 1;
        profiles[i].version = M3_SHD_CALIB_VERSION;
        profiles[i].sample_count = 4;
        for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s)
            for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l)
                profiles[i].rate_bytes_per_s[s][l] = UINT64_C(500) * 1024 * 1024;
    }
    const m3_shd_topology_profile_t topology = {
        topology_hash, 3, resources, 0, NULL
    };
    const m3_shd_calibration_set_t calibration = {
        topology_hash, &topology, 3, profiles, M3_SHD_CALIB_VERSION, {0}, 0
    };
    m3_shadow_plan_t *planner = m3_shd_open(&topology, &calibration,
                                            (m3_shd_clock_t){0, NULL},
                                            NULL, 0);
    assert(planner);

    const m3_shd_m3_load_snapshot_t snapshot = {
        .model_id = 12,
        .forward_id = 34,
        .generation = 2,
        .consumer_node = 56,
        .request_id = 78,
        .consumer_need_ns = UINT64_C(1000000000),
        .key = {3, 5},
        .bytes = UINT64_C(100) * 1024 * 1024,
        .drive_resource_id = 7,
        .demand = 1
    };
    m3_shd_request_t request;
    m3_shd_candidate_t candidate;
    m3_shd_decision_record_t decision;
    assert(m3_shd_m3_make_request(&snapshot, &request, &candidate) == 0);
    assert(m3_shd_expand_candidate(planner, &candidate) == 0);
    assert(candidate.n_resources == 3);
    assert(candidate.resources[0] == 7 && candidate.resources[1] == 70
           && candidate.resources[2] == 700);
    assert(m3_shd_predict(planner, &request, &candidate, 1, &decision)
           == M3_SHD_OK);
    assert(decision.chosen_resource_id == 7);
    m3_shd_close(planner);
}

int main(void) {
    test_translation();
    test_rejects_incomplete_snapshot();
    test_planner_boundary_expands_path();
    puts("m3 shadow M3 bridge: PASS");
    return 0;
}
