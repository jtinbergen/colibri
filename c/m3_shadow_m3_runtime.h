#ifndef COLIBRI_M3_SHADOW_M3_RUNTIME_H
#define COLIBRI_M3_SHADOW_M3_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "m3_shadow_m3_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Context is copied into each concrete load job.  It is deliberately not a
 * pointer into moe()'s stack: PIPE workers and OpenMP iterations outlive the
 * producer's local variables. */
typedef struct {
    uint64_t model_id;
    uint64_t forward_id;
    uint64_t generation;
    uint64_t consumer_node;
    uint64_t consumer_need_ns;
    uint8_t valid;
} m3_shd_m3_exec_context_t;

typedef struct {
    uint8_t issued;
    uint64_t request_id;
    uint32_t trace_index;
} m3_shd_m3_load_observation_t;

/* A bounded, observer-only correlation row.  It joins the planner's
 * counterfactual prediction to the actual load completion without becoming a
 * second admission ledger.  UINT32_MAX in an observation means the bounded
 * trace was full; the engine still proceeds normally. */
#define M3_SHD_M3_TRACE_INDEX_NONE UINT32_MAX
#define M3_SHD_M3_TRACE_PATH_MAX 8u
#define M3_SHD_M3_COMPONENT_COUNT 6u
#define M3_SHD_M3_COMPONENT_MASK_ALL ((uint8_t)((1u << M3_SHD_M3_COMPONENT_COUNT) - 1u))

/* Initialization is intentionally process-lifetime.  It publishes only a
 * fully parsed and opened planner; malformed configuration leaves the engine
 * completely operational with observation disabled. */
int m3_shd_m3_runtime_init_from_env(void);
/* Test/integration entry point using caller-owned text.  The runtime copies
 * the bounded tables into process-lifetime storage before publication. */
int m3_shd_m3_runtime_init_text(const char *text, uint64_t topology_hash,
                                const uint64_t *replica_drive_ids,
                                size_t replica_count);
int m3_shd_m3_runtime_enabled(void);
uint64_t m3_shd_m3_runtime_now_ns(void);
uint64_t m3_shd_m3_runtime_next_forward(void);
uint64_t m3_shd_m3_runtime_model_id(uint64_t fallback);
/* Returns a need-time only when the caller supplied an explicit synthetic
 * delta for fixture/replay work.  Production may also leave the deadline
 * unknown; zero is retained in the trace and never becomes a fabricated
 * capacity or admission guarantee. */
uint64_t m3_shd_m3_runtime_synthetic_need_ns(void);

void m3_shd_m3_runtime_context_push(
    const m3_shd_m3_exec_context_t *next,
    m3_shd_m3_exec_context_t *previous);
void m3_shd_m3_runtime_context_pop(
    const m3_shd_m3_exec_context_t *previous);

void m3_shd_m3_runtime_begin_load(m3_shd_m3_load_observation_t *observation);
/* Returns 1 when an actual shadow-ledger entry was created.  A false return
 * means context, mapping, or calibration was incomplete; the engine must
 * continue normally and the observation remains non-issued. */
int m3_shd_m3_runtime_issue(m3_shd_m3_load_observation_t *observation,
                            uint64_t model_id, ColiExpertKey key,
                            uint64_t bytes, int actual_replica,
                            int path_complete);
/* Production form: component_mask uses bits 0..2 for the three weight
 * tensors and bits 3..5 for their three quantization/scale tensors.  The
 * observer records the mask; it does not change I/O.  A mask other than ALL
 * remains an incomplete mapping in the evidence trace. */
int m3_shd_m3_runtime_issue_mapped(
    m3_shd_m3_load_observation_t *observation,
    uint64_t model_id, ColiExpertKey key, uint64_t bytes,
    int actual_replica, int path_complete, uint8_t component_mask);
void m3_shd_m3_runtime_finish(m3_shd_m3_load_observation_t *observation,
                              int load_result);
size_t m3_shd_m3_runtime_active_count(void);

/* Bounded diagnostic snapshot for tests and offline evidence. */
int m3_shd_m3_runtime_decision_log_copy(char *dst, size_t cap,
                                        size_t *bytes_out);
/* Copy a CSV snapshot for offline evidence.  The snapshot includes the
 * actual resource path, consumer need, predicted ready time, and observed
 * weight-ready time.  It is intentionally separate from the decision log:
 * counterfactual rows and actual completion rows must not be conflated. */
int m3_shd_m3_runtime_trace_copy(char *dst, size_t cap, size_t *bytes_out);

#ifdef __cplusplus
}
#endif

#endif
