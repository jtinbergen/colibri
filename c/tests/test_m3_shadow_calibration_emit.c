/* test_m3_shadow_calibration_emit.c -- 7b-ext end-to-end check.

Reads a planner-text dump produced by ``m3_shadow_calibrate.py --emit-planner-text``
and feeds it into ``m3_shd_parse_text`` to confirm the emit is
planner-acceptable.  The driver is intentionally tiny: it is a gate for
the emit step, not a calibration replacement.
*/
#include "../m3_shadow_plan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fake_now(void *opaque) { return *(uint64_t *)opaque; }

static uint64_t planner_hash(const char *text) {
    const char *marker = strstr(text, "# topology_hash = ");
    unsigned long long value = 0;
    if (!marker || sscanf(marker, "# topology_hash = %llu", &value) != 1
        || !value) {
        fprintf(stderr, "planner text has no valid topology hash\n");
        return 0;
    }
    return (uint64_t)value;
}

static void read_file(const char *path, char *buf, size_t cap) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", path);
        buf[0] = '\0';
        return;
    }
    size_t n = fread(buf, 1, cap - 1, fp);
    buf[n] = '\0';
    fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <planner-text-dump>\n", argv[0]);
        return 2;
    }
    static char text[2 * 1024 * 1024];
    read_file(argv[1], text, sizeof(text));
    if (!text[0]) return 2;
    uint64_t hash = planner_hash(text);
    if (!hash) return 2;

    /* The emit writes profile rows only for explicitly measured resources.
     * An unmeasured controller/upstream remains topology-only, preserving
     * UNKNOWN rather than inheriting drive capacity. */
    m3_shd_resource_t resources[128];
    m3_shd_resource_profile_t profiles[128];
    static m3_shd_weight_copy_t copies[M3_SHD_MAX_COPIES];
    memset(resources, 0, sizeof(resources));
    memset(profiles, 0, sizeof(profiles));
    memset(copies, 0, sizeof(copies));
    m3_shd_topology_profile_t topology = {0};
    m3_shd_calibration_set_t calibration = {0};
    size_t n_r = 0, n_p = 0, n_c = 0;
    char err[256] = {0};
    int rc = m3_shd_parse_text(text, hash, resources, 128, &n_r,
                               profiles, 128, &n_p, copies, M3_SHD_MAX_COPIES, &n_c,
                               &topology, &calibration, err, sizeof(err));
    if (rc) {
        fprintf(stderr, "parse failed: %s\n", err);
        fprintf(stderr, "n_r=%zu n_p=%zu\n", n_r, n_p);
        return 1;
    }
    uint64_t now = 0;
    m3_shadow_plan_t *p = m3_shd_open(&topology, &calibration,
                                      (m3_shd_clock_t){
                                          .now = &fake_now, .opaque = &now },
                                      err, sizeof(err));
    if (!p) {
        fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    int n_cpu = 0, n_gpu = 0, n_mem = 0, n_link = 0,
        n_drive = 0, n_ctrl = 0, n_up = 0;
    for (size_t i = 0; i < n_r; ++i) {
        switch (resources[i].kind) {
            case 0: n_drive++; break;
            case 1: n_ctrl++; break;
            case 2: n_up++; break;
            case 3: n_cpu++; break;
            case 4: n_gpu++; break;
            case 5: n_mem++; break;
            case 6: n_link++; break;
            default: break;
        }
    }
    printf("planner parse: PASS  n_r=%zu n_p=%zu  ", n_r, n_p);
    printf("drive=%d ctrl=%d up=%d cpu=%d gpu=%d mem=%d link=%d\n",
           n_drive, n_ctrl, n_up, n_cpu, n_gpu, n_mem, n_link);
    m3_shd_close(p);

    /* A drive-only calibration must still admit the declared topology, but
     * a path through the unmeasured controller/upstream must be UNKNOWN. */
    const char sparse_text[] =
        "resource 1 0 10 4 1024\n"
        "resource 10 1 20 4 1024\n"
        "resource 20 2 0 4 1024\n"
        "profile 1 0 0 1000 0 0 1\n";
    m3_shd_resource_t sparse_resources[4] = {0};
    m3_shd_resource_profile_t sparse_profiles[4] = {0};
    m3_shd_weight_copy_t sparse_copies[1] = {0};
    m3_shd_topology_profile_t sparse_topology = {0};
    m3_shd_calibration_set_t sparse_calibration = {0};
    size_t sparse_nr = 0, sparse_np = 0, sparse_nc = 0;
    memset(err, 0, sizeof(err));
    if (m3_shd_parse_text(sparse_text, hash, sparse_resources, 4, &sparse_nr,
                          sparse_profiles, 4, &sparse_np, sparse_copies, 1,
                          &sparse_nc, &sparse_topology, &sparse_calibration,
                          err, sizeof(err)) != 0) {
        fprintf(stderr, "sparse parse failed: %s\n", err);
        return 1;
    }
    m3_shadow_plan_t *sparse_plan = m3_shd_open(
        &sparse_topology, &sparse_calibration,
        (m3_shd_clock_t){.now = &fake_now, .opaque = &now}, err, sizeof(err));
    if (!sparse_plan) {
        fprintf(stderr, "sparse open failed: %s\n", err);
        return 1;
    }
    m3_shd_candidate_t sparse_path = {0};
    sparse_path.available = 1;
    sparse_path.n_resources = 3;
    sparse_path.resources[0] = 1;
    sparse_path.resources[1] = 10;
    sparse_path.resources[2] = 20;
    m3_shd_request_t sparse_request = {0};
    sparse_request.request_id = 1;
    sparse_request.key.layer = 1;
    sparse_request.key.expert = 1;
    sparse_request.bytes = 1024;
    sparse_request.consumer_need_ns = UINT64_C(1000000000);
    sparse_request.demand = 1;
    m3_shd_decision_record_t sparse_record = {0};
    if (m3_shd_predict(sparse_plan, &sparse_request, &sparse_path, 1,
                       &sparse_record) != M3_SHD_RESULT_UNKNOWN_BOUND) {
        fprintf(stderr, "sparse prediction was not UNKNOWN\n");
        m3_shd_close(sparse_plan);
        return 1;
    }
    m3_shd_close(sparse_plan);

    /* Supplied sparse maps still reject orphan and duplicate profile objects. */
    m3_shd_resource_t two_resources[2] = {0};
    two_resources[0].resource_id = 1;
    two_resources[0].kind = M3_SHD_KIND_DRIVE;
    two_resources[0].max_inflight = 4;
    two_resources[0].max_inflight_bytes = 1024;
    two_resources[1] = two_resources[0];
    two_resources[1].resource_id = 2;
    m3_shd_topology_profile_t one_topology = {
        .topology_hash = hash, .n_resources = 2, .resources = two_resources};
    m3_shd_resource_profile_t bad_profile = {0};
    bad_profile.resource_id = 99;
    bad_profile.topology_hash = hash;
    bad_profile.units_per_second_explicit = 1;
    bad_profile.sample_count = 1;
    bad_profile.version = M3_SHD_CALIB_VERSION;
    m3_shd_calibration_set_t bad_calibration = {
        .topology_hash = hash, .topology = &one_topology,
        .n_profiles = 1, .profiles = &bad_profile,
        .version = M3_SHD_CALIB_VERSION};
    if (m3_shd_open(&one_topology, &bad_calibration,
                    (m3_shd_clock_t){0}, err, sizeof(err)) != NULL) {
        fprintf(stderr, "orphan profile was accepted\n");
        return 1;
    }
    m3_shd_resource_profile_t duplicate_profiles[2] = {0};
    duplicate_profiles[0] = bad_profile;
    duplicate_profiles[1] = bad_profile;
    duplicate_profiles[0].resource_id = 1;
    duplicate_profiles[1].resource_id = 1;
    bad_calibration.n_profiles = 2;
    bad_calibration.profiles = duplicate_profiles;
    if (m3_shd_open(&one_topology, &bad_calibration,
                    (m3_shd_clock_t){0}, err, sizeof(err)) != NULL) {
        fprintf(stderr, "duplicate profiles were accepted\n");
        return 1;
    }
    printf("sparse calibration semantics: PASS\n");
    return 0;
}
