#include "../m3_shadow_plan.h"

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#define H UINT64_C(0x7d5a11)
#define MB (UINT64_C(1024) * 1024)
#define IO_BYTES (UINT64_C(100) * MB)
#define RATE (UINT32_C(500) * 1024u * 1024u)

static uint64_t fake_now(void *opaque) { return *(uint64_t *)opaque; }

typedef struct {
    m3_shadow_plan_t *plan;
    uint64_t now;
    unsigned calls;
} reentrant_clock_t;

static uint64_t reentrant_now(void *opaque) {
    reentrant_clock_t *clock = opaque;
    clock->calls++;
    if (clock->plan) (void)m3_shd_active_count(clock->plan);
    return clock->now;
}

static void profile_init(m3_shd_resource_profile_t *p,
                         uint64_t id, uint32_t rate, uint32_t startup) {
    memset(p, 0, sizeof(*p));
    p->resource_id = id;
    p->topology_hash = H;
    p->units_per_second_explicit = 1;
    p->version = M3_SHD_CALIB_VERSION;
    p->sample_count = 4;
    for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s) {
        for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l) {
            p->rate_bytes_per_s[s][l] = rate;
            p->startup_ns[s][l] = startup;
        }
    }
}

static void init_profile_set(m3_shd_resource_profile_t *profiles,
                              const m3_shd_resource_t *resources,
                              size_t n) {
    for (size_t i = 0; i < n; ++i)
        profile_init(&profiles[i], resources[i].resource_id, RATE, 0);
}

static m3_shadow_plan_t *open_plan_with_clock(
    const m3_shd_resource_t *resources,
    m3_shd_resource_profile_t *profiles, size_t n, m3_shd_clock_t clock);

static m3_shadow_plan_t *open_plan(const m3_shd_resource_t *resources,
                                   m3_shd_resource_profile_t *profiles,
                                   size_t n, uint64_t *now) {
    return open_plan_with_clock(resources, profiles, n,
                                (m3_shd_clock_t){fake_now, now});
}

static m3_shadow_plan_t *open_plan_with_clock(
    const m3_shd_resource_t *resources,
    m3_shd_resource_profile_t *profiles, size_t n, m3_shd_clock_t clock) {
    static m3_shd_topology_profile_t topology;
    static m3_shd_calibration_set_t calibration;
    topology.topology_hash = H;
    topology.n_resources = n;
    topology.resources = resources;
    calibration.topology_hash = H;
    calibration.topology = &topology;
    calibration.n_profiles = n;
    calibration.profiles = profiles;
    calibration.version = M3_SHD_CALIB_VERSION;
    char err[128] = {0};
    m3_shadow_plan_t *p = m3_shd_open(&topology, &calibration,
                                      clock, err, sizeof(err));
    return p;
}

static m3_shd_request_t request(uint64_t id, uint64_t need) {
    m3_shd_request_t r;
    memset(&r, 0, sizeof(r));
    r.request_id = id;
    r.key.layer = 1;
    r.key.expert = (int)id;
    r.bytes = IO_BYTES;
    r.consumer_need_ns = need;
    r.demand = 1;
    return r;
}

static m3_shd_candidate_t path(uint64_t id, uint64_t drive, uint64_t controller) {
    m3_shd_candidate_t c;
    memset(&c, 0, sizeof(c));
    c.candidate_id = id;
    c.available = 1;
    c.n_resources = 2;
    c.resources[0] = drive;
    c.resources[1] = controller;
    return c;
}

static m3_shd_candidate_t path3(uint64_t id, uint64_t drive,
                                uint64_t controller, uint64_t upstream) {
    m3_shd_candidate_t c = path(id, drive, controller);
    c.n_resources = 3;
    c.resources[2] = upstream;
    return c;
}

static void test_shared_bandwidth(void) {
    m3_shd_resource_t resources[5] = {
        { 1, M3_SHD_KIND_DRIVE, 100, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 100, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 3, M3_SHD_KIND_DRIVE, 100, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 4, M3_SHD_KIND_DRIVE, 100, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 100, M3_SHD_KIND_CONTROLLER, 0, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[5];
    init_profile_set(profiles, resources, 5);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 5, &now);
    for (int i = 0; i < 3; ++i) {
        m3_shd_request_t r = request((uint64_t)i + 1, UINT64_C(1000000000));
        m3_shd_candidate_t c = path((uint64_t)i + 1, (uint64_t)i + 1, 100);
        assert(m3_shd_submit(p, &r, &c, NULL));
    }
    m3_shd_request_t fourth = request(4, UINT64_C(2000000000));
    m3_shd_candidate_t c4 = path(4, 4, 100);
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &fourth, &c4, 1, &rec) == M3_SHD_OK);
    assert(rec.ready_ns == UINT64_C(800000000));
    assert(m3_shd_active_count(p) == 3);
    m3_shd_close(p);
}

static void test_defer_and_independent_replica(void) {
    m3_shd_resource_t resources[6] = {
        { 1, M3_SHD_KIND_DRIVE, 100, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 100, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 3, M3_SHD_KIND_DRIVE, 200, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 100, M3_SHD_KIND_CONTROLLER, 0, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 200, M3_SHD_KIND_CONTROLLER, 0, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 300, M3_SHD_KIND_UPSTREAM, 0, 16, 16 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[6];
    init_profile_set(profiles, resources, 6);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 6, &now);
    m3_shd_request_t a = request(1, UINT64_C(200000000));
    m3_shd_candidate_t pa = path(1, 1, 100);
    assert(m3_shd_submit(p, &a, &pa, NULL));
    m3_shd_request_t b = request(2, UINT64_C(1000000000));
    m3_shd_candidate_t busy = path(2, 2, 100);
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &b, &busy, 1, &rec) == M3_SHD_DEFER);
    assert(rec.ready_ns == UINT64_C(400000000));

    m3_shd_candidate_t candidates[2] = {busy, path(3, 3, 200)};
    assert(m3_shd_predict(p, &b, candidates, 2, &rec) == M3_SHD_OK);
    assert(rec.candidate_index == 1);
    assert(rec.ready_ns == UINT64_C(200000000));
    assert(m3_shd_real_occupancy_bytes(p) == IO_BYTES);
    m3_shd_close(p);
}

static void test_controller_budget_and_contention_ranking(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 100, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 100, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 3, M3_SHD_KIND_DRIVE, 200, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 100, M3_SHD_KIND_CONTROLLER, 0, 1, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 200, M3_SHD_KIND_CONTROLLER, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[5];
    init_profile_set(profiles, resources, 5);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 5, &now);
    m3_shd_request_t active = request(1, UINT64_C(5000000000));
    m3_shd_candidate_t active_path = path(1, 1, 100);
    assert(m3_shd_submit(p, &active, &active_path, NULL));

    /* The drive budget is free, but the shared controller budget is full. */
    m3_shd_request_t next = request(2, UINT64_C(1000000000));
    m3_shd_candidate_t blocked = path(2, 2, 100);
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &next, &blocked, 1, &rec) == M3_SHD_DEFER);
    assert(rec.candidate_index == UINT32_MAX
           && rec.reason == M3_SHD_REASON_DEFER);

    m3_shd_candidate_t candidates[2] = {
        blocked, path(3, 3, 200)
    };
    assert(m3_shd_predict(p, &next, candidates, 2, &rec) == M3_SHD_OK);
    assert(rec.candidate_index == 1 && rec.chosen_resource_id == 3);
    assert(rec.ready_ns == UINT64_C(200000000));
    m3_shd_close(p);

    /* A low-latency path under current contention loses to a free path. */
    resources[3].max_inflight = 4;
    profiles[0].rate_bytes_per_s[M3_SHD_SIZE_LARGE][M3_SHD_LOAD_LOW] =
        UINT64_C(100) * MB;
    p = open_plan(resources, profiles, 5, &now);
    assert(m3_shd_submit(p, &active, &active_path, NULL));
    next = request(3, UINT64_C(5000000000));
    m3_shd_candidate_t contended = path(2, 2, 100);
    m3_shd_candidate_t free_path = path(3, 3, 200);
    m3_shd_candidate_t ranked[2] = {contended, free_path};
    assert(m3_shd_predict(p, &next, ranked, 2, &rec) == M3_SHD_OK);
    assert(rec.candidate_index == 1 && rec.chosen_resource_id == 3);
    assert(rec.ready_ns == UINT64_C(200000000));
    m3_shd_close(p);
}

static void test_deadline_miss_and_stable_tie_break(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[2];
    init_profile_set(profiles, resources, 2);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 2, &now);
    m3_shd_request_t miss = request(1, UINT64_C(1));
    m3_shd_candidate_t candidate = {1, 1, {1}, 1};
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &miss, &candidate, 1, &rec) == M3_SHD_OK);
    assert(rec.ready_ns == UINT64_C(200000000));
    assert(rec.score_ns == UINT64_C(199999999));

    /* Equal score and ready time must choose the lowest stable drive ID,
     * even when that candidate is listed second. */
    m3_shd_request_t tied = request(2, UINT64_C(1000000000));
    m3_shd_candidate_t candidates[2] = {
        {20, 1, {2}, 1}, {10, 1, {1}, 1}
    };
    assert(m3_shd_predict(p, &tied, candidates, 2, &rec) == M3_SHD_OK);
    assert(rec.candidate_index == 1 && rec.chosen_resource_id == 1);
    m3_shd_close(p);
}

static void test_latency_and_honesty(void) {
    m3_shd_resource_t resources[3] = {
        { 1, M3_SHD_KIND_DRIVE, 100, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 100, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 100, M3_SHD_KIND_CONTROLLER, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[3];
    init_profile_set(profiles, resources, 3);
    profiles[0].residual_ns[M3_SHD_SIZE_LARGE][M3_SHD_LOAD_FREE] =
        UINT64_C(50000000);
    profiles[0].residual_percentile_ppm[M3_SHD_SIZE_LARGE]
        [M3_SHD_LOAD_FREE] = 950000;
    for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s)
        for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l)
            profiles[1].startup_ns[s][l] = UINT32_C(50000000);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 3, &now);
    m3_shd_request_t r = request(1, UINT64_C(300000000));
    m3_shd_candidate_t candidates[2] = {path(1, 1, 100), path(2, 2, 100)};
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &r, candidates, 2, &rec) == M3_SHD_OK);
    assert(rec.candidate_index == 0);
    assert(rec.ready_ns == UINT64_C(200000000));
    assert(rec.uncertainty_ns == UINT64_C(50000000));
    m3_shd_close(p);

    profiles[1].units_per_second_explicit = 0;
    assert(open_plan(resources, profiles, 3, &now) == NULL);
}

static void test_serial_conversion_stage(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[2];
    init_profile_set(profiles, resources, 2);
    m3_shd_topology_profile_t topology = {
        H, 2, resources, 0, NULL
    };
    m3_shd_calibration_set_t calibration = {
        H, &topology, 2, profiles, M3_SHD_CALIB_VERSION, {0}, 0
    };
    calibration.conversion.enabled = 1;
    calibration.conversion.version = M3_SHD_CALIB_VERSION;
    calibration.conversion.sample_count = 4;
    calibration.conversion.topology_hash = H;
    calibration.conversion.max_inflight = 4;
    calibration.conversion.max_inflight_bytes = 4 * IO_BYTES;
    for (size_t size = 0; size < M3_SHD_SIZE_CLASS_COUNT; ++size)
        for (size_t load = 0; load < M3_SHD_LOAD_CLASS_COUNT; ++load)
            calibration.conversion.duration_ns[size][load] =
                UINT64_C(100000000);

    uint64_t now = 0;
    char err[128] = {0};
    m3_shadow_plan_t *p = m3_shd_open(
        &topology, &calibration, (m3_shd_clock_t){fake_now, &now},
        err, sizeof(err));
    assert(p);
    m3_shd_request_t first = request(1, UINT64_C(1000000000));
    m3_shd_candidate_t first_path = {1, 1, {1}, 1};
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &first, &first_path, 1, &rec) == M3_SHD_OK);
    assert(rec.ready_ns == UINT64_C(300000000));
    assert(m3_shd_submit(p, &first, &first_path, NULL));

    m3_shd_request_t second = request(2, UINT64_C(1000000000));
    m3_shd_candidate_t second_path = {2, 1, {2}, 1};
    assert(m3_shd_predict(p, &second, &second_path, 1, &rec) == M3_SHD_OK);
    /* Both I/O transfers finish at 0.2 s; request 1 owns the serial
     * conversion slot first and request 2 is ready at 0.4 s. */
    assert(rec.ready_ns == UINT64_C(400000000));
    m3_shd_close(p);

    /* An unmeasured conversion load class is UNKNOWN, never an implicit
     * zero-duration stage. */
    calibration.conversion.duration_ns[M3_SHD_SIZE_LARGE]
        [M3_SHD_LOAD_LOW] = 0;
    p = m3_shd_open(&topology, &calibration,
                    (m3_shd_clock_t){fake_now, &now}, err, sizeof(err));
    assert(p);
    assert(m3_shd_submit(p, &first, &first_path, NULL));
    now = UINT64_C(400000000);
    assert(m3_shd_predict(p, &second, &second_path, 1, &rec)
           == M3_SHD_RESULT_UNKNOWN_BOUND);
    m3_shd_close(p);
}

static void test_uint64_rate_precision(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[1];
    init_profile_set(profiles, resources, 1);
    for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s)
        for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l)
            profiles[0].rate_bytes_per_s[s][l] = UINT64_C(5000000000);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 1, &now);
    m3_shd_request_t r = request(1, UINT64_C(1000000000));
    m3_shd_candidate_t c = {1, 1, {1}, 1};
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &r, &c, 1, &rec) == M3_SHD_OK);
    assert(rec.ready_ns == UINT64_C(20971520));
    m3_shd_close(p);
}

static void test_deterministic_log(void) {
    m3_shd_resource_t resources[2] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[2];
    init_profile_set(profiles, resources, 2);
    uint64_t n1 = 0, n2 = 0;
    m3_shadow_plan_t *p1 = open_plan(resources, profiles, 2, &n1);
    m3_shadow_plan_t *p2 = open_plan(resources, profiles, 2, &n2);
    m3_shd_request_t r = request(7, UINT64_C(500000000));
    m3_shd_candidate_t candidates[2] = {{1, 1, {1}, 1}, {2, 1, {2}, 1}};
    m3_shd_decision_record_t a, b;
    assert(m3_shd_decide(p1, &r, candidates, 2, &a) == M3_SHD_OK);
    assert(m3_shd_decide(p2, &r, candidates, 2, &b) == M3_SHD_OK);
    size_t na, nb;
    const char *la = m3_shd_decision_log(p1, &na);
    const char *lb = m3_shd_decision_log(p2, &nb);
    assert(na == nb && memcmp(la, lb, na) == 0);
    m3_shd_close(p1);
    m3_shd_close(p2);
}

static void test_three_controller_topology_and_bounds(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 100, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 100, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 3, M3_SHD_KIND_DRIVE, 200, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 4, M3_SHD_KIND_DRIVE, 200, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 5, M3_SHD_KIND_DRIVE, 300, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 6, M3_SHD_KIND_DRIVE, 300, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 7, M3_SHD_KIND_DRIVE, 300, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 100, M3_SHD_KIND_CONTROLLER, 900, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 200, M3_SHD_KIND_CONTROLLER, 900, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 300, M3_SHD_KIND_CONTROLLER, 900, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 900, M3_SHD_KIND_UPSTREAM, 0, 8, 8 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[sizeof(resources) / sizeof(resources[0])];
    init_profile_set(profiles, resources, sizeof(resources) / sizeof(resources[0]));
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles,
                                     sizeof(resources) / sizeof(resources[0]), &now);
    m3_shd_request_t a = request(1, UINT64_C(1000000000));
    m3_shd_candidate_t pa = path3(1, 1, 100, 900);
    assert(m3_shd_submit(p, &a, &pa, NULL));
    m3_shd_request_t b = request(2, UINT64_C(1000000000));
    m3_shd_candidate_t pb = path3(2, 3, 200, 900);
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &b, &pb, 1, &rec) == M3_SHD_OK);
    m3_shd_close(p);

    /* A synthetic upstream cap couples the otherwise independent groups. */
    resources[10].max_inflight = 1;
    p = open_plan(resources, profiles,
                  sizeof(resources) / sizeof(resources[0]), &now);
    assert(m3_shd_submit(p, &a, &pa, NULL));
    assert(m3_shd_predict(p, &b, &pb, 1, &rec) == M3_SHD_DEFER);
    assert(!m3_shd_submit(p, &b, &pb, NULL));
    assert(m3_shd_active_count(p) == 1);
    m3_shd_close(p);
}

static void test_missing_replica_tie_and_real_ledger(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 0, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[2];
    init_profile_set(profiles, resources, 2);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 2, &now);
    m3_shd_request_t a = request(1, UINT64_C(1000000000));
    m3_shd_candidate_t pa = {1, 1, {1}, 1};
    assert(m3_shd_submit(p, &a, &pa, NULL));
    m3_shd_request_t b = request(2, UINT64_C(1));
    m3_shd_candidate_t same = {2, 1, {1}, 1};
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &b, &same, 1, &rec) == M3_SHD_DEFER);
    assert(!m3_shd_submit(p, &b, &same, NULL));
    m3_shd_candidate_t missing = {3, 1, {99}, 1};
    assert(m3_shd_predict(p, &b, &missing, 1, &rec)
           == M3_SHD_RESULT_UNKNOWN_BOUND);
    assert(m3_shd_real_occupancy_bytes(p) == IO_BYTES);

    /* This candidate is available and has an equal score; stable ID wins. */
    m3_shd_candidate_t tie[2] = {{20, 1, {2}, 1}, {21, 1, {1}, 1}};
    assert(m3_shd_predict(p, &b, tie, 1, &rec) == M3_SHD_OK);
    assert(rec.chosen_resource_id == 2);
    assert(m3_shd_record_real_failure(p, 1));
    assert(m3_shd_active_count(p) == 0);
    assert(m3_shd_submit(p, &a, &pa, NULL));
    assert(m3_shd_submit(p, &b, &tie[0], NULL));
    assert(m3_shd_record_real_completion(p, 1));
    assert(!m3_shd_record_real_completion(p, 1));
    assert(m3_shd_active_count(p) == 1);
    /* The surviving request is in a later ledger slot; it must still count. */
    assert(m3_shd_predict(p, &b, &tie[0], 1, &rec) == M3_SHD_DEFER);
    m3_shd_close(p);
}

static void test_overdue_active_load_never_defers_in_the_past(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[1];
    init_profile_set(profiles, resources, 1);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 1, &now);
    m3_shd_candidate_t c = {1, 1, {1}, 1};
    m3_shd_request_t active = request(1, UINT64_C(2000000000));
    assert(m3_shd_submit(p, &active, &c, NULL));
    now = UINT64_C(300000000);
    m3_shd_request_t waiting = request(2, UINT64_C(2000000000));
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &waiting, &c, 1, &rec)
           == M3_SHD_RESULT_UNKNOWN_BOUND);
    m3_shd_close(p);
}

static void test_text_parser(void) {
    const char *text =
        "# explicit topology and measured profile rows\n"
        "resource 1 0 10 4 419430400\n"
        "resource 10 1 0 4 419430400\n"
        "profile 1 3 0 5000000000 0 1000 4 950000\n"
        "profile 1 3 2 524288000 0 1000 4 950000\n"
        "profile 10 3 0 524288000 0 1000 4 950000\n"
        "profile 10 3 2 524288000 0 1000 4 950000\n"
        "conversion 4 419430400 3 0 100000000 0 4\n";
    m3_shd_resource_t resources[4];
    m3_shd_resource_profile_t profiles[4];
    m3_shd_weight_copy_t copies[4];
    m3_shd_topology_profile_t topology;
    m3_shd_calibration_set_t calibration;
    size_t nr = 0, np = 0;
    char err[128] = {0};
    assert(m3_shd_parse_text(text, 77, resources, 4, &nr, profiles, 4,
                             &np, copies, 4, &(size_t){0}, &topology,
                             &calibration, err, sizeof(err)) == 0);
    assert(nr == 2 && np == 2 && topology.resources == resources);
    assert(profiles[0].rate_bytes_per_s[M3_SHD_SIZE_LARGE][M3_SHD_LOAD_FREE]
           == UINT64_C(5000000000));
    assert(profiles[0].residual_percentile_ppm[M3_SHD_SIZE_LARGE]
           [M3_SHD_LOAD_FREE] == 950000);
    assert(calibration.conversion.enabled
           && calibration.conversion.max_inflight == 4
           && calibration.conversion.duration_ns[M3_SHD_SIZE_LARGE]
              [M3_SHD_LOAD_FREE] == UINT64_C(100000000));
    uint64_t now = 0;
    m3_shadow_plan_t *p = m3_shd_open(&topology, &calibration,
                                      (m3_shd_clock_t){fake_now, &now},
                                      err, sizeof(err));
    assert(p);
    m3_shd_request_t r = request(1, UINT64_C(1000000000));
    m3_shd_candidate_t c = path(1, 1, 10);
    m3_shd_decision_record_t rec;
    assert(m3_shd_predict(p, &r, &c, 1, &rec) == M3_SHD_OK);
    m3_shd_close(p);
    assert(m3_shd_parse_text("resource 1 0 0 1 1\nwat 1\n", 77,
                             resources, 4, &nr, profiles, 4, &np,
                             copies, 4, &(size_t){0},
                             &topology, &calibration, err, sizeof(err)) != 0);
    assert(m3_shd_parse_text(
               "resource 1 0 0 1 1\n"
               "profile 1 3 0 524288000 0 1000 1\n",
               77, resources, 4, &nr, profiles, 4, &np,
               copies, 4, &(size_t){0}, &topology, &calibration,
               err, sizeof(err)) != 0);
    assert(m3_shd_parse_text(
               "resource 1 0 0 1 1\n"
               "profile 1 3 0 524288000 0 1000 1 900000\n",
               77, resources, 4, &nr, profiles, 4, &np,
               copies, 4, &(size_t){0}, &topology, &calibration,
               err, sizeof(err)) != 0);
    assert(m3_shd_parse_text(
               "resource 1 0 0 1 1\n"
               "profile 1 3 0 524288000 0 1000 1 975000\n",
               77, resources, 4, &nr, profiles, 4, &np,
               copies, 4, &(size_t){0}, &topology, &calibration,
               err, sizeof(err)) != 0);
}

static void test_topology_rejections(void) {
    m3_shd_resource_t duplicate_resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, M3_SHD_KIND_CONTROLLER, 0, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t duplicate_profiles[2];
    init_profile_set(duplicate_profiles, duplicate_resources, 2);
    uint64_t now = 0;
    assert(open_plan(duplicate_resources, duplicate_profiles, 2, &now) == NULL);

    m3_shd_resource_t cyclic_resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 2, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_CONTROLLER, 1, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t cyclic_profiles[2];
    init_profile_set(cyclic_profiles, cyclic_resources, 2);
    assert(open_plan(cyclic_resources, cyclic_profiles, 2, &now) == NULL);

    m3_shd_resource_t copy_resources[] = {
        { 10, M3_SHD_KIND_DRIVE, 0, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 20, M3_SHD_KIND_CONTROLLER, 0, 1, IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t copy_profiles[2];
    init_profile_set(copy_profiles, copy_resources, 2);
    m3_shd_weight_copy_t invalid_copy = {
        1, 77, UINT64_C(0x100000001), IO_BYTES, 20, 1
    };
    m3_shd_topology_profile_t copy_topology = {
        H, 2, copy_resources, 1, &invalid_copy
    };
    m3_shd_calibration_set_t copy_calibration = {
        H, &copy_topology, 2, copy_profiles, M3_SHD_CALIB_VERSION, {0}, 0
    };
    char copy_error[128] = {0};
    assert(m3_shd_open(&copy_topology, &copy_calibration,
                       (m3_shd_clock_t){fake_now, &now},
                       copy_error, sizeof(copy_error)) == NULL);

    m3_shd_resource_t parsed_resources[2];
    m3_shd_resource_profile_t parsed_profiles[2];
    m3_shd_weight_copy_t parsed_copies[2];
    m3_shd_topology_profile_t parsed_topology;
    m3_shd_calibration_set_t parsed_calibration;
    size_t nr = 0, np = 0;
    char err[128] = {0};
    assert(m3_shd_parse_text(
               "resource 1 0 0 1 1048576\n"
               "resource 1 1 0 1 1048576\n"
               "profile 1 3 0 524288000 0 0 1\n",
               77, parsed_resources, 2, &nr, parsed_profiles, 2, &np,
               parsed_copies, 2, &(size_t){0},
               &parsed_topology, &parsed_calibration, err, sizeof(err)) != 0);
}

static void test_unavailable_copy_does_not_fallback(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 2, 2 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 0, 2, 2 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[2];
    init_profile_set(profiles, resources, 2);
    m3_shd_weight_copy_t copies[] = {
        {1, 77, UINT64_C(0x100000001), IO_BYTES, 1, 0}
    };
    m3_shd_topology_profile_t topology = {
        H, 2, resources, 1, copies
    };
    m3_shd_calibration_set_t calibration = {
        H, &topology, 2, profiles, M3_SHD_CALIB_VERSION, {0}, 0
    };
    uint64_t now = 0;
    char err[128] = {0};
    m3_shadow_plan_t *p = m3_shd_open(
        &topology, &calibration, (m3_shd_clock_t){fake_now, &now},
        err, sizeof(err));
    assert(p);
    m3_shd_candidate_t candidates[2];
    size_t count = 0;
    assert(!m3_shd_candidates_for_key(
        p, 77, (ColiExpertKey){1, 1}, IO_BYTES, 1,
        candidates, 2, &count));
    assert(count == 0);

    /* With no configured identity row, the explicitly observed drive may
     * still be used as the conservative actual-path fallback. */
    assert(m3_shd_candidates_for_key(
        p, 77, (ColiExpertKey){1, 2}, IO_BYTES, 2,
        candidates, 2, &count));
    assert(count == 1 && candidates[0].candidate_id == 2);
    m3_shd_close(p);

    m3_shd_weight_copy_t contradictory_copies[] = {
        {1, 77, UINT64_C(0x100000003), IO_BYTES, 1, 1},
        {2, 77, UINT64_C(0x100000003), 2 * IO_BYTES, 2, 1}
    };
    m3_shd_topology_profile_t contradictory_topology = {
        H, 2, resources, 2, contradictory_copies
    };
    m3_shd_calibration_set_t contradictory_calibration = {
        H, &contradictory_topology, 2, profiles, M3_SHD_CALIB_VERSION, {0}, 0
    };
    assert(m3_shd_open(
        &contradictory_topology, &contradictory_calibration,
        (m3_shd_clock_t){fake_now, &now}, err, sizeof(err)) == NULL);
}

static void test_bounded_age_queue(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[1];
    init_profile_set(profiles, resources, 1);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 1, &now);
    m3_shd_request_t newer = request(1, UINT64_C(1000000000));
    m3_shd_request_t older = request(2, UINT64_C(1000000000));
    newer.enqueue_ns = 20;
    older.enqueue_ns = 10;
    uint64_t id1, id2;
    assert(m3_shd_enqueue(p, &newer, &id1));
    assert(m3_shd_enqueue(p, &older, &id2));
    assert(m3_shd_queued_count(p) == 2);
    m3_shd_request_t next;
    assert(m3_shd_peek_next(p, &next));
    assert(next.request_id == id2);
    assert(m3_shd_remove_queued(p, id2));
    assert(m3_shd_queued_count(p) == 1);
    assert(m3_shd_remove_queued(p, id1));
    assert(m3_shd_queued_count(p) == 0);
    m3_shd_close(p);

    /* Once configured age is crossed, an older request wins even when a
     * newer request has the earlier consumer need.  The order among aged
     * requests remains enqueue-time then request-ID deterministic. */
    uint64_t aged_now = UINT64_C(100);
    m3_shadow_plan_t *aged = NULL;

    m3_shd_resource_t parsed_resources[1];
    m3_shd_resource_profile_t parsed_profiles[1];
    m3_shd_weight_copy_t parsed_copies[1];
    m3_shd_topology_profile_t parsed_topology;
    m3_shd_calibration_set_t parsed_calibration;
    size_t nr = 0, np = 0, nc = 0;
    char err[128] = {0};
    int parsed_ok = m3_shd_parse_text(
        "queue_age_ns 50\n"
        "resource 1 0 0 4 419430400\n"
        "profile 1 3 0 5000000000 0 0 4\n",
        H, parsed_resources, 1, &nr, parsed_profiles, 1, &np,
        parsed_copies, 1, &nc, &parsed_topology, &parsed_calibration,
        err, sizeof(err));
    assert(parsed_ok == 0);
    assert(parsed_calibration.max_queue_age_ns == 50);
    aged_now = 100;
    aged = m3_shd_open(&parsed_topology, &parsed_calibration,
                       (m3_shd_clock_t){fake_now, &aged_now},
                       err, sizeof(err));
    assert(aged);
    m3_shd_request_t urgent = request(3, UINT64_C(1));
    m3_shd_request_t old = request(4, UINT64_C(1000));
    urgent.enqueue_ns = 90;
    old.enqueue_ns = 10;
    uint64_t urgent_id = 0, old_id = 0;
    assert(m3_shd_enqueue(aged, &urgent, &urgent_id));
    assert(m3_shd_enqueue(aged, &old, &old_id));
    assert(m3_shd_peek_next(aged, &next));
    assert(next.request_id == old_id);
    m3_shd_close(aged);
}

static void test_request_identity_and_log_snapshot(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, M3_SHD_MAX_ACTIVE_REQUESTS, M3_SHD_MAX_ACTIVE_REQUESTS * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[1];
    init_profile_set(profiles, resources, 1);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 1, &now);
    m3_shd_request_t queued = request(100, UINT64_C(1000000000));
    m3_shd_candidate_t candidate = {100, 1, {1}, 1};
    uint64_t request_id = 0;
    assert(m3_shd_enqueue(p, &queued, &request_id));
    assert(request_id == 100 && m3_shd_queued_count(p) == 1);
    assert(!m3_shd_enqueue(p, &queued, NULL));

    m3_shd_request_t mismatch = queued;
    mismatch.bytes++;
    assert(!m3_shd_submit(p, &mismatch, &candidate, NULL));
    assert(m3_shd_queued_count(p) == 1);
    assert(m3_shd_submit(p, &queued, &candidate, &request_id));
    assert(request_id == 100 && m3_shd_queued_count(p) == 0);
    assert(!m3_shd_submit(p, &queued, &candidate, NULL));
    assert(m3_shd_record_real_completion(p, 100));

    m3_shd_request_t explicit_request = request(200, UINT64_C(1000000000));
    assert(m3_shd_submit(p, &explicit_request, &candidate, &request_id));
    assert(request_id == 200);
    m3_shd_request_t automatic_request = request(0, UINT64_C(1000000000));
    assert(m3_shd_submit(p, &automatic_request, &candidate, &request_id));
    assert(request_id == 201);
    m3_shd_request_t collision = request(201, UINT64_C(1000000000));
    assert(!m3_shd_submit(p, &collision, &candidate, NULL));
    assert(m3_shd_record_real_completion(p, 200));
    assert(m3_shd_record_real_completion(p, 201));

    m3_shd_decision_record_t decision;
    assert(m3_shd_decide(p, &queued, &candidate, 1, &decision)
           == M3_SHD_OK);
    size_t log_bytes = 0;
    assert(!m3_shd_decision_log_copy(p, NULL, 0, &log_bytes));
    assert(log_bytes > 0);
    char log_copy[4096];
    assert(log_bytes + 1 < sizeof(log_copy));
    size_t copied_bytes = 0;
    assert(m3_shd_decision_log_copy(p, log_copy, sizeof(log_copy),
                                    &copied_bytes));
    assert(copied_bytes == log_bytes && log_copy[copied_bytes] == '\0');
    const char *borrowed = m3_shd_decision_log(p, NULL);
    assert(borrowed && memcmp(borrowed, log_copy, log_bytes) == 0);
    char too_small[1];
    assert(!m3_shd_decision_log_copy(p, too_small, sizeof(too_small),
                                     &copied_bytes));
    assert(copied_bytes == log_bytes);

#pragma omp parallel sections
    {
#pragma omp section
        {
            for (int i = 0; i < 256; ++i) {
                m3_shd_request_t concurrent_request =
                    request((uint64_t)(1000 + i), UINT64_C(1000000000));
                assert(m3_shd_decide(p, &concurrent_request, &candidate, 1,
                                     &decision) == M3_SHD_OK);
            }
        }
#pragma omp section
        {
            for (int i = 0; i < 256; ++i) {
                char snapshot[4096];
                size_t snapshot_bytes = 0;
                int copied = m3_shd_decision_log_copy(
                    p, snapshot, sizeof(snapshot), &snapshot_bytes);
                if (copied) assert(snapshot[snapshot_bytes] == '\0');
                else assert(snapshot_bytes + 1 > sizeof(snapshot));
            }
        }
    }
    m3_shd_close(p);

    p = open_plan(resources, profiles, 1, &now);
    m3_shd_request_t near_limit = request(UINT64_MAX - 1,
                                          UINT64_C(1000000000));
    assert(m3_shd_enqueue(p, &near_limit, &request_id));
    assert(request_id == UINT64_MAX - 1);
    m3_shd_request_t exhausted = request(0, UINT64_C(1000000000));
    assert(!m3_shd_enqueue(p, &exhausted, NULL));
    m3_shd_request_t max_id = request(UINT64_MAX, UINT64_C(1000000000));
    assert(!m3_shd_submit(p, &max_id, &candidate, NULL));
    assert(m3_shd_remove_queued(p, UINT64_MAX - 1));
    m3_shd_close(p);
}

static void test_concurrent_submit_completion_capacity(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, M3_SHD_MAX_ACTIVE_REQUESTS, M3_SHD_MAX_ACTIVE_REQUESTS * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[1];
    init_profile_set(profiles, resources, 1);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 1, &now);
    int successes = 0;
#pragma omp parallel for schedule(static) reduction(+:successes)
    for (int i = 1; i <= 256; ++i) {
        m3_shd_request_t r = request((uint64_t)i,
                                     UINT64_C(1000000000));
        m3_shd_candidate_t c = {(uint64_t)i, 1, {1}, 1};
        if (m3_shd_submit(p, &r, &c, NULL)) {
            assert(m3_shd_record_real_completion(p, (uint64_t)i));
            successes++;
        }
    }
    assert(successes == 256);
    assert(m3_shd_active_count(p) == 0);
    assert(m3_shd_real_occupancy_bytes(p) == 0);
    m3_shd_close(p);
}

static void test_atomic_decide_and_real_start(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[2];
    init_profile_set(profiles, resources, 2);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 2, &now);
    m3_shd_candidate_t candidates[2] = {
        {1, 1, {1}, 1}, {2, 1, {2}, 1}
    };
    m3_shd_candidate_t actual = candidates[0];
    m3_shd_resource_occupancy_t first_snapshot[1];
    m3_shd_decision_record_t rec;
    uint64_t recorded = 0;
    m3_shd_request_t first = request(1, UINT64_C(1000000000));
    assert(m3_shd_decide_and_record_real_start(
               p, &first, candidates, 2, &actual, &rec, &recorded,
               first_snapshot, 1)
           == M3_SHD_OK);
    assert(recorded == 1 && rec.chosen_resource_id == 1);
    assert(first_snapshot[0].resource_id == 1
           && first_snapshot[0].inflight == 1);

    m3_shd_request_t second = request(2, UINT64_C(1000000000));
    m3_shd_resource_occupancy_t second_snapshot[1];
    assert(m3_shd_decide_and_record_real_start(
               p, &second, candidates, 2, &actual, &rec, &recorded,
               second_snapshot, 1)
           == M3_SHD_OK);
    assert(recorded == 2 && rec.chosen_resource_id == 2);
    assert(second_snapshot[0].resource_id == 1
           && second_snapshot[0].inflight == 2);
    assert(m3_shd_active_count(p) == 2);
    assert(m3_shd_record_real_completion(p, 1));
    assert(m3_shd_record_real_completion(p, 2));
    m3_shd_close(p);

    now = 0;
    p = open_plan(resources, profiles, 2, &now);
    m3_shd_request_t automatic = request(0, UINT64_C(1000000000));
    uint64_t automatic_id = 0;
    assert(m3_shd_decide_and_record_real_start(
               p, &automatic, candidates, 2, &actual, &rec, &automatic_id,
               NULL, 0)
           == M3_SHD_OK);
    assert(automatic_id == 1 && rec.request_id == automatic_id);
    assert(m3_shd_record_real_completion(p, automatic_id));

    m3_shd_candidate_t too_many[M3_SHD_MAX_RESOURCES + 1];
    for (size_t i = 0; i < M3_SHD_MAX_RESOURCES + 1; ++i)
        too_many[i] = candidates[0];
    automatic_id = 0;
    assert(m3_shd_decide_and_record_real_start(
               p, &automatic, too_many, M3_SHD_MAX_RESOURCES + 1,
               &actual, &rec, &automatic_id, NULL, 0) == M3_SHD_UNKNOWN);
    assert(automatic_id == 0 && m3_shd_active_count(p) == 0);
    m3_shd_close(p);
}

static void test_clock_callback_not_under_guard(void) {
    m3_shd_resource_t resource =
        { 1, M3_SHD_KIND_DRIVE, 0, 2, 2 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    m3_shd_resource_profile_t profile;
    init_profile_set(&profile, &resource, 1);
    reentrant_clock_t clock = {0, 0, 0};
    m3_shadow_plan_t *p = open_plan_with_clock(
        &resource, &profile, 1, (m3_shd_clock_t){reentrant_now, &clock});
    clock.plan = p;
    m3_shd_request_t r = request(1, UINT64_C(1000000000));
    m3_shd_candidate_t candidate = {1, 1, {1}, 1};
    m3_shd_decision_record_t decision;
    uint64_t recorded = 0;
    assert(m3_shd_decide_and_record_real_start(
               p, &r, &candidate, 1, &candidate, &decision, &recorded,
               NULL, 0)
           == M3_SHD_OK);
    assert(clock.calls == 1 && recorded == 1);
    assert(m3_shd_record_real_completion(p, recorded));
    m3_shd_close(p);
}

static void test_concurrent_atomic_decide_and_real_start(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 0, 4, 4 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[2];
    init_profile_set(profiles, resources, 2);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(resources, profiles, 2, &now);
    m3_shd_candidate_t candidates[2] = {
        {1, 1, {1}, 1}, {2, 1, {2}, 1}
    };
    m3_shd_candidate_t actual = candidates[0];
    m3_shd_decision_record_t decisions[2];
    m3_shd_resource_occupancy_t snapshots[2];
    uint64_t recorded_ids[2] = {0, 0};
    int results[2] = {0, 0};
#pragma omp parallel for num_threads(2) schedule(static)
    for (int i = 0; i < 2; ++i) {
        m3_shd_request_t r = request((uint64_t)i + 1,
                                     UINT64_C(1000000000));
        results[i] = m3_shd_decide_and_record_real_start(
            p, &r, candidates, 2, &actual, &decisions[i],
            &recorded_ids[i], &snapshots[i], 1) == M3_SHD_OK;
    }
    assert(results[0] && results[1]);
    assert(m3_shd_active_count(p) == 2);
    int saw_first = 0, saw_second = 0;
    for (size_t i = 0; i < 2; ++i) {
        if (decisions[i].chosen_resource_id == 1) {
            assert(decisions[i].occupancy_before == 0
                   && decisions[i].occupancy_after == 1);
            assert(snapshots[i].inflight == 1);
            saw_first = 1;
        } else {
            assert(decisions[i].chosen_resource_id == 2);
            assert(decisions[i].occupancy_before == 1
                   && decisions[i].occupancy_after == 2);
            assert(snapshots[i].inflight == 2);
            saw_second = 1;
        }
        assert(recorded_ids[i] != 0);
    }
    assert(saw_first && saw_second);
    assert(m3_shd_record_real_completion(p, recorded_ids[0]));
    assert(m3_shd_record_real_completion(p, recorded_ids[1]));
    m3_shd_close(p);
}

static void test_completion_order_permutations(void) {
    m3_shd_resource_t resources[] = {
        { 1, M3_SHD_KIND_DRIVE, 0, 8, 8 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, M3_SHD_KIND_DRIVE, 0, 8, 8 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 3, M3_SHD_KIND_DRIVE, 0, 8, 8 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    m3_shd_resource_profile_t profiles[3];
    init_profile_set(profiles, resources, 3);
    const int permutations[][3] = {
        {1, 2, 3}, {3, 1, 2}, {2, 3, 1}, {3, 2, 1}
    };
    char reference[1024] = {0};
    size_t reference_bytes = 0;
    for (size_t permutation = 0;
         permutation < sizeof(permutations) / sizeof(permutations[0]);
         ++permutation) {
        uint64_t now = 0;
        m3_shadow_plan_t *p = open_plan(resources, profiles, 3, &now);
        for (uint64_t id = 1; id <= 3; ++id) {
            m3_shd_request_t r = request(id, UINT64_C(1000000000));
            m3_shd_candidate_t c = {id, 1, {id}, 1};
            assert(m3_shd_submit(p, &r, &c, NULL));
        }
        for (size_t i = 0; i < 3; ++i)
            assert(m3_shd_record_real_completion(
                p, (uint64_t)permutations[permutation][i]));
        m3_shd_request_t next = request(9, UINT64_C(1000000000));
        m3_shd_candidate_t candidate = {1, 1, {1}, 1};
        m3_shd_decision_record_t decision;
        assert(m3_shd_decide(p, &next, &candidate, 1, &decision)
               == M3_SHD_OK);
        char log[1024] = {0};
        size_t bytes = 0;
        assert(m3_shd_decision_log_copy(p, log, sizeof(log), &bytes));
        if (permutation == 0) {
            reference_bytes = bytes;
            memcpy(reference, log, bytes);
        } else {
            assert(bytes == reference_bytes);
            assert(memcmp(log, reference, bytes) == 0);
        }
        m3_shd_close(p);
    }
}

static void test_profile_hash_covers_admission_and_startup(void) {
    m3_shd_resource_t resource =
        { 1, M3_SHD_KIND_DRIVE, 0, 2, 2 * IO_BYTES, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    m3_shd_resource_profile_t profile;
    profile_init(&profile, resource.resource_id, RATE, 0);
    uint64_t now = 0;
    m3_shadow_plan_t *p = open_plan(&resource, &profile, 1, &now);
    assert(p);
    uint64_t base = m3_shd_profile_hash(p);
    m3_shd_close(p);

    resource.max_inflight = 1;
    p = open_plan(&resource, &profile, 1, &now);
    assert(p && m3_shd_profile_hash(p) != base);
    m3_shd_close(p);

    resource.max_inflight = 2;
    profile.startup_ns[M3_SHD_SIZE_LARGE][M3_SHD_LOAD_FREE] = 1;
    p = open_plan(&resource, &profile, 1, &now);
    assert(p && m3_shd_profile_hash(p) != base);
    m3_shd_close(p);

    resource.rate_bytes_per_s = UINT64_C(123456789);
    p = open_plan(&resource, &profile, 1, &now);
    assert(p && m3_shd_profile_hash(p) != base);
    m3_shd_close(p);
}

int main(void) {
    test_shared_bandwidth();
    test_defer_and_independent_replica();
    test_controller_budget_and_contention_ranking();
    test_deadline_miss_and_stable_tie_break();
    test_latency_and_honesty();
    test_serial_conversion_stage();
    test_uint64_rate_precision();
    test_deterministic_log();
    test_three_controller_topology_and_bounds();
    test_missing_replica_tie_and_real_ledger();
    test_overdue_active_load_never_defers_in_the_past();
    test_text_parser();
    test_topology_rejections();
    test_unavailable_copy_does_not_fallback();
    test_bounded_age_queue();
    test_request_identity_and_log_snapshot();
    test_concurrent_submit_completion_capacity();
    test_atomic_decide_and_real_start();
    test_clock_callback_not_under_guard();
    test_concurrent_atomic_decide_and_real_start();
    test_completion_order_permutations();
    test_profile_hash_covers_admission_and_startup();
    puts("m3 shadow planner fixtures: PASS");
    return 0;
}
