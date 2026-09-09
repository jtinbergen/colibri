#include "../m3_shadow_m3_runtime.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint64_t forward_id;
    int ok;
} concurrent_load_t;

static void *concurrent_load(void *opaque) {
    concurrent_load_t *state = (concurrent_load_t *)opaque;
    m3_shd_m3_exec_context_t context = {
        7, state->forward_id, 9, state->forward_id, 0, 1
    };
    m3_shd_m3_exec_context_t previous;
    m3_shd_m3_runtime_context_push(&context, &previous);
    m3_shd_m3_load_observation_t observation;
    m3_shd_m3_runtime_begin_load(&observation);
    state->ok = m3_shd_m3_runtime_issue_mapped(
        &observation, 7, (ColiExpertKey){0, 0}, 100, 0, 1,
        M3_SHD_M3_COMPONENT_MASK_ALL);
    if (state->ok) m3_shd_m3_runtime_finish(&observation, 0);
    m3_shd_m3_runtime_context_pop(&previous);
    return NULL;
}

int main(void) {
    const char *config =
        "resource 301 2 0 4 1048576\n"
        "resource 201 1 301 4 1048576\n"
        "resource 101 0 201 4 1048576\n"
        "resource 302 2 0 4 1048576\n"
        "resource 202 1 302 4 1048576\n"
        "resource 102 0 202 4 1048576\n"
        "profile 301 0 0 1000000000 0 0 1\n"
        "profile 201 0 0 1000000000 0 0 1\n"
        "profile 101 0 0 1000000000 0 0 1\n"
        "profile 302 0 0 1000000000 0 0 1\n"
        "profile 202 0 0 1000000000 0 0 1\n"
        "profile 102 0 0 1000000000 0 0 1\n"
        "copy 1 7 0 100 101 1\n"
        "copy 2 7 0 100 102 1\n";
    const uint64_t drives[] = {101, 102};
    assert(m3_shd_m3_runtime_init_text(config, 99, drives, 2));
    assert(m3_shd_m3_runtime_enabled());

    m3_shd_m3_load_observation_t missing_context;
    m3_shd_m3_runtime_begin_load(&missing_context);
    assert(!m3_shd_m3_runtime_issue(&missing_context, 7,
                                    (ColiExpertKey){0, 0}, 100, 0, 1));

    m3_shd_m3_exec_context_t context = {
        7, 1, 2, 3, m3_shd_m3_runtime_now_ns() + 1000000000u, 1
    };
    m3_shd_m3_exec_context_t previous;
    m3_shd_m3_runtime_context_push(&context, &previous);
    m3_shd_m3_load_observation_t observation;
    m3_shd_m3_runtime_begin_load(&observation);
    assert(m3_shd_m3_runtime_issue_mapped(
        &observation, 7, (ColiExpertKey){0, 0}, 100, 0, 1,
        M3_SHD_M3_COMPONENT_MASK_ALL));
    assert(observation.issued && observation.request_id != 0);
    m3_shd_m3_load_observation_t second_observation;
    m3_shd_m3_runtime_begin_load(&second_observation);
    assert(m3_shd_m3_runtime_issue_mapped(
        &second_observation, 7, (ColiExpertKey){0, 0}, 100, 0, 1,
        M3_SHD_M3_COMPONENT_MASK_ALL));
    assert(second_observation.issued
           && second_observation.request_id != observation.request_id);
    assert(m3_shd_m3_runtime_active_count() == 2);
    m3_shd_m3_runtime_finish(&second_observation, -1);
    m3_shd_m3_runtime_finish(&observation, -1);
    assert(!observation.issued && !second_observation.issued);
    assert(m3_shd_m3_runtime_active_count() == 0);
    m3_shd_m3_runtime_context_pop(&previous);

    concurrent_load_t concurrent[4] = {
        {11, 0}, {12, 0}, {13, 0}, {14, 0}
    };
    pthread_t threads[4];
    for (size_t i = 0; i < 4; ++i)
        assert(pthread_create(&threads[i], NULL, concurrent_load,
                              &concurrent[i]) == 0);
    char concurrent_trace[8192];
    size_t concurrent_bytes = 0;
    for (int i = 0; i < 32; ++i)
        (void)m3_shd_m3_runtime_trace_copy(concurrent_trace,
                                            sizeof(concurrent_trace),
                                            &concurrent_bytes);
    for (size_t i = 0; i < 4; ++i) {
        assert(pthread_join(threads[i], NULL) == 0);
        assert(concurrent[i].ok);
    }
    assert(m3_shd_m3_runtime_active_count() == 0);

    char trace[4096];
    size_t trace_bytes = 0;
    assert(m3_shd_m3_runtime_trace_copy(trace, sizeof(trace), &trace_bytes));
    assert(trace_bytes > 0);
    assert(strstr(trace, "request_id,model_id,forward_id") != NULL);
    assert(strstr(trace, "component_mask") != NULL);
    assert(strstr(trace, "path_inflight,path_bytes,concurrency,request_type") != NULL);
    assert(strstr(trace, ",63,101,101|201|301,") != NULL);
    assert(strstr(trace, ",102,") != NULL);
    assert(strstr(trace, ",101,101|201|301,") != NULL);
    assert(strstr(trace, "101:1:100|201:1:100|301:1:100,") != NULL);
    assert(strstr(trace, ",1,demand,OBSERVED,") != NULL);
    assert(strstr(trace, ",1\n") != NULL);

    m3_shd_m3_exec_context_t no_deadline = context;
    no_deadline.consumer_need_ns = 0;
    m3_shd_m3_runtime_context_push(&no_deadline, &previous);
    m3_shd_m3_runtime_begin_load(&observation);
    assert(m3_shd_m3_runtime_issue_mapped(
        &observation, 7, (ColiExpertKey){0, 0}, 100, 0, 1,
        M3_SHD_M3_COMPONENT_MASK_ALL));
    m3_shd_m3_runtime_finish(&observation, 0);
    m3_shd_m3_runtime_context_pop(&previous);

    char log[4096];
    size_t bytes = 0;
    assert(m3_shd_m3_runtime_decision_log_copy(log, sizeof(log), &bytes));
    assert(bytes > 0 && strstr(log, "1,") != NULL);
    /* The second issue is admitted while the first actual path is still
     * registered, so the live combined operation must expose drive 102 as
     * the counterfactual with occupancy 1 -> 2. */
    assert(strstr(log, ",102,1,1,2,") != NULL);
    puts("m3 shadow M3 runtime seam: PASS");
    return 0;
}
