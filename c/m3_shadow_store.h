#ifndef COLIBRI_M3_SHADOW_STORE_H
#define COLIBRI_M3_SHADOW_STORE_H

#include "expert_store.h"
#include "m3_shadow_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*m3_shd_snapshot_provider_fn)(
    void *opaque, ColiExpertKey key, m3_shd_candidate_t *candidates,
    size_t *candidate_count, uint64_t *consumer_need_ns);

/* Takes ownership of inner and, when non-NULL, planner.  The wrapper forwards
 * every store operation to inner and places only telemetry hooks around
 * lookup/release; it never changes a view, routing decision, or return code. */
ColiExpertStore *m3_shd_store_wrap(ColiExpertStore *inner,
                                   m3_shadow_plan_t *planner);
ColiExpertStore *m3_shd_store_wrap_with_provider(
    ColiExpertStore *inner, m3_shadow_plan_t *planner,
    m3_shd_snapshot_provider_fn provider, void *provider_opaque);
int m3_shd_store_is_wrapper(const ColiExpertStore *store);

#ifdef __cplusplus
}
#endif

#endif
