#include "../m3_shadow_plan.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    const m3_shd_residency_budget_t budgets[] = {
        {1, 1000, 4}, {2, 100, 1},
    };
    const m3_shd_residency_candidate_t candidates[] = {
        /* Best move consumes the one available target slot. */
        {.content_id=1, .content_generation=1, .source_domain=1, .target_domain=2,
         .bytes=80, .expected_wait_saved_ns=200, .movement_cost_ns=20,
         .source_available=1, .target_eligible=1},
        /* Still beneficial, but cannot use an EVICT proposal as capacity. */
        {.content_id=2, .content_generation=1, .source_domain=1, .target_domain=2,
         .bytes=80, .expected_wait_saved_ns=150, .movement_cost_ns=0,
         .source_available=1, .target_eligible=1},
        {.content_id=3, .content_generation=1, .target_domain=2, .resident_domain=2,
         .resident_slot_id=9, .cache_snapshot_generation=7, .bytes=64,
         .expected_wait_saved_ns=100, .target_eligible=1, .resident=1},
        {.content_id=4, .content_generation=1, .resident_domain=2,
         .resident_slot_id=10, .cache_snapshot_generation=7, .bytes=80,
         .resident=1, .evictable=1},
        /* These must produce no action. */
        {.content_id=5, .content_generation=1, .source_domain=1, .target_domain=2,
         .bytes=40, .expected_wait_saved_ns=999, .source_available=1,
         .target_eligible=1, .in_use=1},
        {.content_id=6, .content_generation=1, .resident_domain=2,
         .resident_slot_id=11, .cache_snapshot_generation=8, .bytes=40,
         .resident=1, .evictable=1},
        {.content_id=7, .content_generation=1, .source_domain=1, .target_domain=99,
         .bytes=40, .expected_wait_saved_ns=999, .source_available=1,
         .target_eligible=1},
        {.content_id=8, .content_generation=1, .source_domain=1, .target_domain=2,
         .bytes=40, .expected_wait_saved_ns=999, .source_available=1,
         .target_eligible=1},
        {.content_id=8, .content_generation=1, .source_domain=1, .target_domain=2,
         .bytes=40, .expected_wait_saved_ns=999, .source_available=1,
         .target_eligible=1},
    };
    m3_shd_residency_action_t actions[8];
    size_t count = 0;
    assert(m3_shd_plan_residency(candidates,
                                 sizeof(candidates) / sizeof(candidates[0]),
                                 budgets, sizeof(budgets) / sizeof(budgets[0]),
                                 7, actions, 8, &count));
    assert(count == 3);
    assert(actions[0].kind == M3_SHD_RESIDENCY_MOVE
           && actions[0].content_id == 1 && actions[0].target_domain == 2);
    assert(actions[1].kind == M3_SHD_RESIDENCY_PIN
           && actions[1].content_id == 3 && actions[1].resident_slot_id == 9);
    assert(actions[2].kind == M3_SHD_RESIDENCY_EVICT
           && actions[2].content_id == 4 && actions[2].resident_slot_id == 10
           && actions[2].target_domain == 2);

    /* A short output buffer fails atomically rather than returning a prefix. */
    count = 99;
    assert(!m3_shd_plan_residency(candidates,
                                  sizeof(candidates) / sizeof(candidates[0]),
                                  budgets, sizeof(budgets) / sizeof(budgets[0]),
                                  7, actions, 2, &count));
    assert(count == 0);
    puts("m3 shadow residency actions: PASS");
    return 0;
}
