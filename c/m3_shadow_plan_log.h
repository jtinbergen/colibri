#ifndef COLIBRI_M3_SHADOW_PLAN_LOG_H
#define COLIBRI_M3_SHADOW_PLAN_LOG_H

#include <stdint.h>
#include <stddef.h>

#include "m3_shadow_plan_calib.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    M3_SHD_OK = 0,
    M3_SHD_UNKNOWN = 1,
    M3_SHD_OVERDUE_UNKNOWN = 2,
    M3_SHD_DEFER = 3,
    M3_SHD_RESULT_UNKNOWN_BOUND = 4
} m3_shd_result_t;

/* Fixed-width decision record.  The textual log is generated from this
 * record in field order, so replay comparisons do not depend on libc's
 * struct padding or locale. */
typedef enum {
    M3_SHD_REASON_ADMIT = 0,
    M3_SHD_REASON_DEFER,
    M3_SHD_REASON_UNKNOWN_PROFILE,
    M3_SHD_REASON_BOUND,
    M3_SHD_REASON_NO_FEASIBLE
} m3_shd_reason_t;

typedef struct {
    uint64_t request_id;
    uint64_t snapshot_hash;
    uint64_t profile_hash;
    uint64_t dispatch_ns;
    uint64_t ready_ns;
    uint64_t consumer_need_ns;
    uint64_t score_ns;
    uint64_t uncertainty_ns;
    m3_shd_resource_id_t chosen_resource_id;
    uint32_t candidate_index;
    uint32_t occupancy_before;
    uint32_t occupancy_after;
    m3_shd_result_t result;
    m3_shd_reason_t reason;
} m3_shd_decision_record_t;

#ifdef __cplusplus
}
#endif

#endif
