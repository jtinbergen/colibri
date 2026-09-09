#ifndef COLIBRI_M3_SHADOW_PLAN_H
#define COLIBRI_M3_SHADOW_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"
#include "m3_shadow_plan_calib.h"
#include "m3_shadow_plan_log.h"

#ifdef __cplusplus
extern "C" {
#endif

#define M3_SHD_MAGIC 0x53484450u
#define M3_SHD_VERSION 1u
#define M3_SHD_MAX_RESOURCES 128u
/* Full MiniMax-M3 has 60*128 expert identities.  Two physical replicas
 * therefore need 15,360 bounded copy rows; keep this domain separate from
 * resource/active/candidate limits so those arrays remain small. */
#define M3_SHD_MAX_COPIES 16384u
#define M3_SHD_MAX_PATH_RESOURCES 8u
#define M3_SHD_MAX_ACTIVE_REQUESTS 64u
#define M3_SHD_MAX_QUEUED_REQUESTS 64u
#define M3_SHD_MAX_QUEUED_BYTES (UINT64_C(1) << 34)
#define M3_SHD_MAX_DECISION_LOG_BYTES (1024u * 1024u)
#define M3_SHD_MAX_RESIDENCY_CANDIDATES 256u
#define M3_SHD_MAX_RESIDENCY_ACTIONS 256u

typedef uint64_t (*m3_shd_now_fn)(void *opaque);
typedef struct {
    m3_shd_now_fn now;
    void *opaque;
} m3_shd_clock_t;

typedef struct m3_shadow_plan m3_shadow_plan_t;

typedef enum {
    M3_SHD_STATE_QUEUED = 0,
    M3_SHD_STATE_RESERVED,
    M3_SHD_STATE_ISSUED,
    M3_SHD_STATE_COMPLETED,
    M3_SHD_STATE_FAILED,
    M3_SHD_STATE_CANCELLED,
    M3_SHD_STATE_RETIRED
} m3_shd_request_state_t;

typedef struct {
    uint64_t request_id;       /* zero means: let the planner assign one */
    uint64_t model_id;
    ColiExpertKey key;
    uint64_t bytes;
    uint64_t consumer_need_ns;
    uint64_t enqueue_ns;
    uint8_t demand;            /* 1 = demand, 0 = prefetch */
    uint64_t forward_id;
    uint64_t generation;
    uint64_t consumer_node;
    m3_shd_request_state_t state;
    uint64_t selected_copy_id;
    size_t n_reserved_resources;
    m3_shd_resource_id_t reserved_resources[M3_SHD_MAX_PATH_RESOURCES];
} m3_shd_request_t;

typedef struct {
    uint64_t candidate_id;
    size_t n_resources;
    m3_shd_resource_id_t resources[M3_SHD_MAX_PATH_RESOURCES];
    uint8_t available;
} m3_shd_candidate_t;

typedef struct {
    m3_shd_resource_id_t resource_id;
    uint32_t inflight;
    uint64_t inflight_bytes;
} m3_shd_resource_occupancy_t;

typedef struct {
    uint64_t lookup_begins;
    uint64_t lookup_failures;
    uint64_t lookup_bytes;
    uint64_t releases;
} m3_shd_observer_stats_t;

/* Step 8c: pure shadow-residency proposal input.  All fields are immutable
 * caller facts from one ownership snapshot.  The planner never receives a
 * Model/ESlot pointer and cannot perform a cache operation. */
typedef struct {
    uint64_t content_id;
    uint64_t content_generation;
    m3_shd_resource_id_t source_domain;
    m3_shd_resource_id_t target_domain;
    m3_shd_resource_id_t resident_domain;
    uint64_t resident_slot_id;
    uint64_t cache_snapshot_generation;
    uint64_t bytes;
    uint64_t expected_wait_saved_ns;
    uint64_t movement_cost_ns;
    uint8_t source_available;
    uint8_t target_eligible;
    uint8_t resident;
    uint8_t evictable;
    uint8_t pinned;
    uint8_t in_use;
} m3_shd_residency_candidate_t;

typedef struct {
    m3_shd_resource_id_t domain_id;
    uint64_t available_bytes;
    uint32_t available_slots;
} m3_shd_residency_budget_t;

typedef enum {
    M3_SHD_RESIDENCY_PIN = 1,
    M3_SHD_RESIDENCY_MOVE = 2,
    M3_SHD_RESIDENCY_EVICT = 3
} m3_shd_residency_action_kind_t;

typedef struct {
    m3_shd_residency_action_kind_t kind;
    uint64_t content_id;
    uint64_t content_generation;
    m3_shd_resource_id_t source_domain;
    m3_shd_resource_id_t target_domain;
    uint64_t resident_slot_id;
    uint64_t bytes;
    uint64_t expected_wait_saved_ns;
    uint64_t movement_cost_ns;
    uint64_t cache_snapshot_generation;
} m3_shd_residency_action_t;

/* Produces deterministic, shadow-only actions from one fixed cache snapshot.
 * MOVE consumes only caller-declared available target budget.  EVICT returns
 * evidence for an already evictable resident entry and never supplies budget
 * to another action in the same call. */
int m3_shd_plan_residency(
    const m3_shd_residency_candidate_t *candidates, size_t candidate_count,
    const m3_shd_residency_budget_t *budgets, size_t budget_count,
    uint64_t cache_snapshot_generation,
    m3_shd_residency_action_t *out, size_t out_cap, size_t *out_count);

/* All input arrays are borrowed and must remain immutable for the lifetime
 * of the planner.  The planner copies only its bounded active ledger. */
m3_shadow_plan_t *m3_shd_open(const m3_shd_topology_profile_t *topology,
                              const m3_shd_calibration_set_t *calibration,
                              m3_shd_clock_t clock,
                              char *err, size_t errsz);
void m3_shd_close(m3_shadow_plan_t *p);

int m3_shd_submit(m3_shadow_plan_t *p, const m3_shd_request_t *request,
                  const m3_shd_candidate_t *chosen, uint64_t *request_id_out);
int m3_shd_enqueue(m3_shadow_plan_t *p, const m3_shd_request_t *request,
                   uint64_t *request_id_out);
int m3_shd_peek_next(const m3_shadow_plan_t *p, m3_shd_request_t *out);
int m3_shd_remove_queued(m3_shadow_plan_t *p, uint64_t request_id);
size_t m3_shd_queued_count(const m3_shadow_plan_t *p);
int m3_shd_request_priority_cmp(const m3_shd_request_t *a,
                                const m3_shd_request_t *b);
int m3_shd_record_real_completion(m3_shadow_plan_t *p, uint64_t request_id);
int m3_shd_record_real_failure(m3_shadow_plan_t *p, uint64_t request_id);
/* Register an observed real load in the planner-owned ledger without
 * changing the engine's admission or route.  The path is the path actually
 * used by the engine, while m3_shd_decide() may have logged a different
 * counterfactual choice.  If the configured hard bound is already exceeded,
 * the observation is retained as UNKNOWN rather than being discarded. */
int m3_shd_record_real_start(m3_shadow_plan_t *p,
                             const m3_shd_request_t *request,
                             const m3_shd_candidate_t *actual_path,
                             uint64_t *request_id_out);

/* Pure evaluation: no log, occupancy, or real-side state is changed. */
m3_shd_result_t m3_shd_predict(const m3_shadow_plan_t *p,
                               const m3_shd_request_t *request,
                               const m3_shd_candidate_t *candidates,
                               size_t n_candidates,
                               m3_shd_decision_record_t *out);

/* The non-const form additionally appends one deterministic record to the
 * planner-owned replay log. */
m3_shd_result_t m3_shd_decide(m3_shadow_plan_t *p,
                              const m3_shd_request_t *request,
                              const m3_shd_candidate_t *candidates,
                              size_t n_candidates,
                              m3_shd_decision_record_t *out);
/* Atomic observer operation for live integrations: evaluate the immutable
 * snapshot and register the actually issued path under one planner guard.
 * This prevents concurrent observer calls from both deciding against the
 * same stale occupancy snapshot.  `out` and `request_id_out` are required so
 * a successful registration is always correlated with its decision.  It
 * still changes no engine-side state.  When `occupancy_out` is supplied, the
 * path occupancy is copied under the same guard immediately after
 * registration, producing one consistent issue snapshot for telemetry. */
m3_shd_result_t m3_shd_decide_and_record_real_start(
    m3_shadow_plan_t *p, const m3_shd_request_t *request,
    const m3_shd_candidate_t *candidates, size_t n_candidates,
    const m3_shd_candidate_t *actual_path,
    m3_shd_decision_record_t *out, uint64_t *request_id_out,
    m3_shd_resource_occupancy_t *occupancy_out, size_t occupancy_cap);

size_t m3_shd_active_count(const m3_shadow_plan_t *p);
uint64_t m3_shd_real_occupancy_bytes(const m3_shadow_plan_t *p);
/* Read-only actual-load view used by the runtime evidence seam.  The result
 * includes the request just recorded when queried after
 * m3_shd_record_real_start(). */
int m3_shd_real_resource_occupancy(
    const m3_shadow_plan_t *p, m3_shd_resource_id_t resource_id,
    m3_shd_resource_occupancy_t *out);
/* Borrowed-pointer compatibility accessor.  The returned buffer and length
 * are valid only while the caller externally prevents decide/close calls.
 * Use m3_shd_decision_log_copy() for concurrent or independent readers. */
const char *m3_shd_decision_log(const m3_shadow_plan_t *p, size_t *bytes_out);
/* Copy the committed log prefix and its NUL terminator while holding the
 * planner guard.  bytes_out reports the committed byte count even when dst
 * is NULL or cap is too small.  Returns 1 only when the complete snapshot
 * (including terminator) was copied. */
int m3_shd_decision_log_copy(const m3_shadow_plan_t *p, char *dst,
                             size_t cap, size_t *bytes_out);
uint64_t m3_shd_profile_hash(const m3_shadow_plan_t *p);
/* Expand a provider's drive-only candidate with its declared controller and
 * upstream ancestors. A full path is left unchanged; a missing/contradictory
 * ancestor chain returns nonzero. */
int m3_shd_expand_candidate(const m3_shadow_plan_t *p,
                            m3_shd_candidate_t *candidate);
/* Build the complete set of configured, equivalent drive candidates for an
 * M3 key.  tensor_id is encoded as (uint32_t)layer << 32 | (uint32_t)expert
 * in the topology copy table.  The actual drive is retained as a fallback
 * candidate when no copy row is present, provided it is a declared resource. */
int m3_shd_candidates_for_key(const m3_shadow_plan_t *p,
                              uint64_t model_id, ColiExpertKey key,
                              uint64_t bytes,
                              m3_shd_resource_id_t actual_drive_id,
                              m3_shd_candidate_t *out, size_t cap,
                              size_t *count_out);
void m3_shd_observer_stats(const m3_shadow_plan_t *p,
                           m3_shd_observer_stats_t *out);

/* Parse the bounded, line-oriented calibration format used by shadow
 * captures.  Storage is caller-provided; no unbounded queue or hidden
 * allocation is allowed.  Lines are either:
 *
 *   resource <id> <kind> <parent-id> <max-inflight> <max-bytes>
 *   cpu_island <id> <parent-mem> <isa-bits> <cores> [<eligible>]
 *   gpu_island <id> <parent-link> <vram-bytes> <p2p-class>
 *                <pcie-parent> [<eligible>]
 *   memory_dom <id> <parent-id> <rate-bps> <bandwidth-class>
 *   link <id> <parent-cpu> <pcie-parent> <rate-bps>
 *   profile  <id> <size-class> <load-class> <rate-bytes/s>
 *            <startup-ns> <residual-ns> <sample-count> [<tail-percentile-ppm>]
 *   copy     <copy-id> <model-id> <tensor-id> <bytes> <drive-id> <available>
 *   conversion <max-inflight> <max-bytes> <size-class> <load-class>
 *              <duration-ns> <residual-ns> <sample-count>
 *              [<tail-percentile-ppm>]
 *   queue_age_ns <max-age-ns>
 *
 * A profile may have multiple rows.  Missing rows remain zero and therefore
 * produce UNKNOWN at prediction time rather than an invented capacity.  A
 * non-zero residual requires the optional tail percentile, expressed in
 * parts per million (for example 950000 for p95); unlabelled tail margins
 * are rejected.  Parse rejects unknown kinds, duplicate ids, cyclic parent
 * chains, parent-id chains longer than M3_SHD_MAX_PATH_RESOURCES, and
 * malformed payload fields.  The five new line kinds are additions only;
 * existing `resource` rows must remain valid and parse byte-identically. */
int m3_shd_parse_text(const char *text, uint64_t topology_hash,
                      m3_shd_resource_t *resources, size_t resource_cap,
                      size_t *resource_count,
                      m3_shd_resource_profile_t *profiles,
                      size_t profile_cap, size_t *profile_count,
                      m3_shd_weight_copy_t *copies, size_t copy_cap,
                      size_t *copy_count,
                      m3_shd_topology_profile_t *topology,
                      m3_shd_calibration_set_t *calibration,
                      char *err, size_t errsz);

/* Observer hooks are deliberately telemetry-only.  They do not replace or
 * alter the engine's store vtable; this is the safe shadow boundary. */
void m3_shd_observe_lookup_begin(m3_shadow_plan_t *p,
                                 const ColiExpertKey *key,
                                 uint64_t consumer_need_ns);
void m3_shd_observe_lookup_end(m3_shadow_plan_t *p,
                               const ColiExpertKey *key, int result,
                               uint64_t bytes);
void m3_shd_observe_release(m3_shadow_plan_t *p, const ColiExpertKey *key);

#ifdef __cplusplus
}
#endif

#endif
