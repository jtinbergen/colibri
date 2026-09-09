#ifndef COLIBRI_M3_SHADOW_M3_BRIDGE_H
#define COLIBRI_M3_SHADOW_M3_BRIDGE_H

#include "m3_shadow_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Immutable facts that the MiniMax-M3 load/DAG callsite must provide.  This
 * adapter performs no I/O and does not read global engine state; the caller is
 * responsible for deriving consumer_need_ns from the currently known DAG. */
typedef struct {
    uint64_t model_id;
    uint64_t forward_id;
    uint64_t generation;
    uint64_t consumer_node;
    uint64_t request_id;
    uint64_t consumer_need_ns;
    ColiExpertKey key;
    uint64_t bytes;
    m3_shd_resource_id_t drive_resource_id;
    uint8_t demand;
} m3_shd_m3_load_snapshot_t;

/* Convert one M3 demand-load snapshot into the planner's bounded request and
 * a drive-only candidate.  The planner expands the candidate through its
 * configured controller/upstream ancestors before admission. */
int m3_shd_m3_make_request(const m3_shd_m3_load_snapshot_t *snapshot,
                           m3_shd_request_t *request,
                           m3_shd_candidate_t *candidate);

#ifdef __cplusplus
}
#endif

#endif
