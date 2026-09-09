/* Scale evidence for the full MiniMax-M3 copy-table bound.
 *
 * This intentionally measures the current linear parser/open/lookup paths at
 * 60 layers x 128 experts x 2 replicas.  It is evidence, not a performance
 * promotion: the test records elapsed time and only fails on correctness or
 * capacity-bound violations.
 */
#include "../m3_shadow_plan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H UINT64_C(20260907)
#define LAYERS 60
#define EXPERTS 128
#define COPY_ROWS (LAYERS * EXPERTS * 2)
#define LOOKUP_REPEATS 1000
#define TEXT_CAP (4u * 1024u * 1024u)

static m3_shd_resource_t resources[M3_SHD_MAX_RESOURCES];
static m3_shd_resource_profile_t profiles[M3_SHD_MAX_RESOURCES];
static m3_shd_weight_copy_t copies[M3_SHD_MAX_COPIES];

static int append_text(char *text, size_t cap, size_t *used,
                       const char *format, unsigned long long a,
                       unsigned long long b, unsigned long long c,
                       unsigned long long d, unsigned long long e,
                       unsigned long long f) {
    if (*used >= cap) return 0;
    int n = snprintf(text + *used, cap - *used, format, a, b, c, d, e, f);
    if (n < 0 || (size_t)n >= cap - *used) return 0;
    *used += (size_t)n;
    return 1;
}

int main(void) {
    char *text = (char *)malloc(TEXT_CAP);
    if (!text) return 2;
    size_t used = 0;
    memset(resources, 0, sizeof(resources));
    memset(profiles, 0, sizeof(profiles));
    memset(copies, 0, sizeof(copies));

    const char *topology_text =
        "resource 101 0 201 64 1073741824\n"
        "resource 201 1 301 64 1073741824\n"
        "resource 301 2 0 64 1073741824\n"
        "resource 102 0 202 64 1073741824\n"
        "resource 202 1 302 64 1073741824\n"
        "resource 302 2 0 64 1073741824\n"
        "profile 101 0 0 1000000000 0 0 1\n"
        "profile 201 0 0 1000000000 0 0 1\n"
        "profile 301 0 0 1000000000 0 0 1\n"
        "profile 102 0 0 1000000000 0 0 1\n"
        "profile 202 0 0 1000000000 0 0 1\n"
        "profile 302 0 0 1000000000 0 0 1\n";
    size_t topology_len = strlen(topology_text);
    memcpy(text, topology_text, topology_len + 1);
    used = topology_len;
    uint64_t copy_id = 1;
    for (unsigned layer = 1; layer <= LAYERS; ++layer) {
        for (unsigned expert = 0; expert < EXPERTS; ++expert) {
            uint64_t tensor = ((uint64_t)layer << 32) | expert;
            if (!append_text(text, TEXT_CAP, &used,
                             "copy %llu 7 %llu 512 101 1\n",
                             (unsigned long long)copy_id++,
                             (unsigned long long)tensor, 0, 0, 0, 0)
                || !append_text(text, TEXT_CAP, &used,
                                "copy %llu 7 %llu 512 102 1\n",
                                (unsigned long long)copy_id++,
                                (unsigned long long)tensor, 0, 0, 0, 0)) {
                free(text);
                return 2;
            }
        }
    }
    if (copy_id - 1 != COPY_ROWS || copy_id - 1 > M3_SHD_MAX_COPIES) {
        free(text);
        return 2;
    }

    m3_shd_topology_profile_t topology = {0};
    m3_shd_calibration_set_t calibration = {0};
    size_t resource_count = 0, profile_count = 0, copy_count = 0;
    char error[256] = {0};
    clock_t parse_begin = clock();
    int rc = m3_shd_parse_text(text, H, resources, M3_SHD_MAX_RESOURCES,
                               &resource_count, profiles,
                               M3_SHD_MAX_RESOURCES, &profile_count, copies,
                               M3_SHD_MAX_COPIES, &copy_count, &topology,
                               &calibration, error, sizeof(error));
    clock_t parse_end = clock();
    if (rc || resource_count != 6 || profile_count != 6
        || copy_count != COPY_ROWS) {
        fprintf(stderr, "scale parse failed: %s r=%zu p=%zu c=%zu\n",
                error, resource_count, profile_count, copy_count);
        free(text);
        return 1;
    }
    m3_shadow_plan_t *plan = m3_shd_open(
        &topology, &calibration, (m3_shd_clock_t){0}, error, sizeof(error));
    clock_t open_end = clock();
    if (!plan) {
        fprintf(stderr, "scale open failed: %s\n", error);
        free(text);
        return 1;
    }

    const unsigned probes[] = {1, (LAYERS * EXPERTS) / 2, LAYERS * EXPERTS};
    size_t total_candidates = 0;
    clock_t lookup_begin = clock();
    for (size_t repeat = 0; repeat < LOOKUP_REPEATS; ++repeat) {
        for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
            unsigned ordinal = probes[i] - 1;
            ColiExpertKey key = {(int)(ordinal / EXPERTS) + 1,
                                 (int)(ordinal % EXPERTS)};
            m3_shd_candidate_t candidates[2] = {0};
            size_t candidate_count = 0;
            if (!m3_shd_candidates_for_key(plan, 7, key, 512, 101,
                                           candidates, 2, &candidate_count)
                || candidate_count != 2) {
                fprintf(stderr, "scale lookup failed at layer=%d expert=%d\n",
                        key.layer, key.expert);
                m3_shd_close(plan);
                free(text);
                return 1;
            }
            total_candidates += candidate_count;
        }
    }
    clock_t lookup_end = clock();
    printf("copy scale: PASS rows=%zu resources=%zu profiles=%zu "
           "parse_ms=%.2f open_ms=%.2f probe_total_ms=%.2f "
           "probe_calls=%u candidates=%zu\n",
           copy_count, resource_count, profile_count,
           1000.0 * (double)(parse_end - parse_begin) / CLOCKS_PER_SEC,
           1000.0 * (double)(open_end - parse_end) / CLOCKS_PER_SEC,
           1000.0 * (double)(lookup_end - lookup_begin) / CLOCKS_PER_SEC,
           (unsigned)(LOOKUP_REPEATS * 3u),
           total_candidates);
    m3_shd_close(plan);
    free(text);
    return 0;
}
