#include "m3_active_admission.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t used;
    uint8_t released;
    m3_adm_consumer_token_t token;
    uint64_t demand_request_id;
    uint64_t demand_generation;
    uint64_t demand_lease_id;
    uint64_t demand_lease_generation;
} m3_adm_consumer_t;

typedef struct {
    uint8_t used;
    uint64_t ticket_nonce;
    m3_adm_request_desc_t desc;
    m3_shd_request_state_t state;
    uint32_t issued_parts;
    uint32_t terminal_parts;
    uint32_t issued_mask;
    uint32_t terminal_mask;
    uint8_t any_part_failed;
    uint8_t cancel_pending;
    uint8_t issue_closed;
    uint8_t resources_released;
    uint8_t lease_released;
    uint8_t prefetch_unwanted;
    uint64_t predicted_wait_before_ns;
    uint64_t predicted_wait_after_ns;
    m3_adm_consumer_t consumers[M3_ADM_MAX_CONSUMERS];
} m3_adm_request_t;

typedef struct {
    uint32_t reserved_requests;
    uint32_t issued_requests;
    uint64_t reserved_bytes;
    uint64_t issued_bytes;
} m3_adm_counter_t;

struct m3_adm_dispatcher {
    const m3_shd_topology_profile_t *topology;
    uint64_t topology_hash;
    uint64_t profile_hash;
    uint8_t trusted_policy;
    uint8_t accepting;
    uint64_t next_ticket_nonce;
    m3_adm_demand_reserve_t reserves[M3_SHD_MAX_RESOURCES];
    size_t reserve_count;
    m3_adm_prefetch_predict_fn prefetch_predict;
    void *prefetch_predict_opaque;
    atomic_flag guard;
    m3_adm_request_t requests[M3_ADM_MAX_REQUESTS];
    m3_adm_counter_t counters[M3_SHD_MAX_RESOURCES];
};

static void lock(m3_adm_dispatcher_t *d) {
    while (atomic_flag_test_and_set_explicit(&d->guard, memory_order_acquire)) {}
}
static void unlock(m3_adm_dispatcher_t *d) {
    atomic_flag_clear_explicit(&d->guard, memory_order_release);
}
static void set_error(char *err, size_t errsz, const char *message) {
    if (err && errsz) snprintf(err, errsz, "%s", message);
}
static int resource_index(const m3_adm_dispatcher_t *d,
                          m3_shd_resource_id_t id) {
    if (!d || !d->topology || !id) return -1;
    for (size_t i = 0; i < d->topology->n_resources; ++i)
        if (d->topology->resources[i].resource_id == id) return (int)i;
    return -1;
}
static int same_ticket(const m3_adm_request_t *r, const m3_adm_ticket_t *t) {
    return r && t && r->used && r->desc.request_id == t->request_id
        && r->desc.generation == t->generation
        && r->desc.lease_id == t->lease_id
        && r->desc.lease_generation == t->lease_generation
        && r->ticket_nonce == t->ticket_nonce;
}
static m3_adm_request_t *find_locked(m3_adm_dispatcher_t *d,
                                     const m3_adm_ticket_t *ticket) {
    if (!d || !ticket) return NULL;
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i)
        if (same_ticket(&d->requests[i], ticket)) return &d->requests[i];
    return NULL;
}
static int duplicate_identity_locked(const m3_adm_dispatcher_t *d,
                                     const m3_adm_request_desc_t *desc) {
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i) {
        const m3_adm_request_t *r = &d->requests[i];
        if (r->used && r->desc.request_id == desc->request_id
            && r->desc.generation == desc->generation) return 1;
        if (r->used && r->desc.lease_id == desc->lease_id
            && r->desc.lease_generation == desc->lease_generation) return 1;
    }
    return 0;
}
static int path_valid(const m3_adm_dispatcher_t *d,
                      const m3_adm_request_desc_t *desc) {
    if (!d || !desc || !desc->request_id || !desc->generation || !desc->bytes
        || !desc->lease_id || !desc->lease_generation || desc->demand > 1
        || !desc->expected_parts || desc->expected_parts > M3_ADM_MAX_PARTS
        || !desc->path.n_resources
        || desc->path.n_resources > M3_SHD_MAX_PATH_RESOURCES)
        return 0;
    for (size_t i = 0; i < desc->path.n_resources; ++i) {
        int ri = resource_index(d, desc->path.resources[i]);
        if (ri < 0 || !d->topology->resources[ri].eligible) return 0;
        for (size_t j = 0; j < i; ++j)
            if (desc->path.resources[j] == desc->path.resources[i]) return 0;
        /* A caller may not omit the controller/upstream ancestors of an
         * otherwise valid drive.  Reserving only `{drive}` would silently
         * bypass shared-resource contention. */
        m3_shd_resource_id_t parent = d->topology->resources[ri].parent_id;
        for (size_t depth = 0; parent; ++depth) {
            if (depth >= M3_SHD_MAX_PATH_RESOURCES) return 0;
            int pi = resource_index(d, parent);
            if (pi < 0 || !d->topology->resources[pi].eligible) return 0;
            int present = 0;
            for (size_t j = 0; j < desc->path.n_resources; ++j)
                if (desc->path.resources[j] == parent) { present = 1; break; }
            if (!present) return 0;
            parent = d->topology->resources[pi].parent_id;
        }
    }
    return 1;
}
static int can_reserve_locked(const m3_adm_dispatcher_t *d,
                              const m3_adm_request_desc_t *desc) {
    for (size_t i = 0; i < desc->path.n_resources; ++i) {
        int ri = resource_index(d, desc->path.resources[i]);
        const m3_shd_resource_t *resource = &d->topology->resources[ri];
        const m3_adm_counter_t *counter = &d->counters[ri];
        if (counter->reserved_requests >= resource->max_inflight
            || desc->bytes > resource->max_inflight_bytes
            || counter->reserved_bytes > resource->max_inflight_bytes - desc->bytes)
            return 0;
    }
    return 1;
}
static int storage_resource(const m3_shd_resource_t *resource) {
    return resource && (resource->kind == M3_SHD_KIND_DRIVE
        || resource->kind == M3_SHD_KIND_CONTROLLER
        || resource->kind == M3_SHD_KIND_UPSTREAM);
}
static const m3_adm_demand_reserve_t *reserve_for_locked(
    const m3_adm_dispatcher_t *d, m3_shd_resource_id_t resource_id) {
    for (size_t i = 0; i < d->reserve_count; ++i)
        if (d->reserves[i].resource_id == resource_id) return &d->reserves[i];
    return NULL;
}
static int prefetch_allowed_locked(m3_adm_dispatcher_t *d,
                                   m3_adm_request_t *r) {
    if (!d->prefetch_predict) return 0;
    uint64_t before = 0, after = 0;
    if (!d->prefetch_predict(d->prefetch_predict_opaque, &r->desc,
                             &before, &after) || after > before) return 0;
    for (size_t i = 0; i < r->desc.path.n_resources; ++i) {
        int ri = resource_index(d, r->desc.path.resources[i]);
        const m3_shd_resource_t *resource = &d->topology->resources[ri];
        const m3_adm_counter_t *counter = &d->counters[ri];
        const m3_adm_demand_reserve_t *reserve = reserve_for_locked(
            d, r->desc.path.resources[i]);
        uint32_t reserve_requests = reserve ? reserve->reserve_requests : 0;
        uint64_t reserve_bytes = reserve ? reserve->reserve_bytes : 0;
        if (counter->reserved_requests >= resource->max_inflight - reserve_requests
            || r->desc.bytes > resource->max_inflight_bytes - reserve_bytes
            || counter->reserved_bytes > resource->max_inflight_bytes - reserve_bytes
                - r->desc.bytes) return 0;
    }
    r->predicted_wait_before_ns = before;
    r->predicted_wait_after_ns = after;
    return 1;
}
static void reserve_locked(m3_adm_dispatcher_t *d, const m3_adm_request_t *r) {
    for (size_t i = 0; i < r->desc.path.n_resources; ++i) {
        int ri = resource_index(d, r->desc.path.resources[i]);
        d->counters[ri].reserved_requests++;
        d->counters[ri].reserved_bytes += r->desc.bytes;
    }
}
static void release_resources_locked(m3_adm_dispatcher_t *d,
                                     m3_adm_request_t *r) {
    if (r->resources_released) return;
    for (size_t i = 0; i < r->desc.path.n_resources; ++i) {
        int ri = resource_index(d, r->desc.path.resources[i]);
        m3_adm_counter_t *c = &d->counters[ri];
        if (!c->reserved_requests || c->reserved_bytes < r->desc.bytes) abort();
        c->reserved_requests--;
        c->reserved_bytes -= r->desc.bytes;
        if (r->issued_parts) {
            if (!c->issued_requests || c->issued_bytes < r->desc.bytes) abort();
            c->issued_requests--;
            c->issued_bytes -= r->desc.bytes;
        }
    }
    r->resources_released = 1;
}
static void complete_if_drained_locked(m3_adm_dispatcher_t *d,
                                       m3_adm_request_t *r,
                                       m3_adm_part_status_t last_status) {
    if (!r->issue_closed || r->terminal_parts != r->issued_parts) return;
    r->state = last_status == M3_ADM_PART_COMPLETED && !r->any_part_failed
        && !r->cancel_pending
        && r->issued_parts == r->desc.expected_parts
        ? M3_SHD_STATE_COMPLETED : M3_SHD_STATE_FAILED;
    release_resources_locked(d, r);
}

m3_adm_dispatcher_t *m3_adm_open(const m3_shd_topology_profile_t *topology,
                                  uint64_t topology_hash,
                                  uint64_t profile_hash,
                                  int trusted_policy,
                                  char *err, size_t errsz) {
    if (!topology || !topology_hash || !profile_hash
        || topology->topology_hash != topology_hash || !topology->resources
        || !topology->n_resources
        || topology->n_resources > M3_SHD_MAX_RESOURCES) {
        set_error(err, errsz, "invalid admission topology/profile");
        return NULL;
    }
    m3_adm_dispatcher_t *d = calloc(1, sizeof(*d));
    if (!d) { set_error(err, errsz, "admission allocation failed"); return NULL; }
    d->topology = topology;
    d->topology_hash = topology_hash;
    d->profile_hash = profile_hash;
    d->trusted_policy = trusted_policy ? 1u : 0u;
    d->accepting = 1;
    atomic_flag_clear(&d->guard);
    return d;
}
int m3_adm_close(m3_adm_dispatcher_t *d) {
    if (!d) return 1;
    lock(d);
    d->accepting = 0;
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i) {
        if (d->requests[i].used) {
            unlock(d);
            return 0;
        }
    }
    unlock(d);
    free(d);
    return 1;
}

int m3_adm_configure_prefetch(m3_adm_dispatcher_t *d,
                              const m3_adm_demand_reserve_t *reserves,
                              size_t reserve_count,
                              m3_adm_prefetch_predict_fn predict,
                              void *predict_opaque) {
    if (!d || reserve_count > M3_SHD_MAX_RESOURCES
        || (reserve_count && !reserves)) return 0;
    lock(d);
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i)
        if (d->requests[i].used) { unlock(d); return 0; }
    for (size_t i = 0; i < reserve_count; ++i) {
        int ri = resource_index(d, reserves[i].resource_id);
        if (ri < 0 || !storage_resource(&d->topology->resources[ri])
            || !d->topology->resources[ri].eligible
            || reserves[i].reserve_requests > d->topology->resources[ri].max_inflight
            || reserves[i].reserve_bytes > d->topology->resources[ri].max_inflight_bytes) {
            unlock(d); return 0;
        }
        for (size_t j = 0; j < i; ++j)
            if (reserves[j].resource_id == reserves[i].resource_id) {
                unlock(d); return 0;
            }
    }
    memset(d->reserves, 0, sizeof(d->reserves));
    if (reserve_count) memcpy(d->reserves, reserves,
                              reserve_count * sizeof(reserves[0]));
    d->reserve_count = reserve_count;
    d->prefetch_predict = predict;
    d->prefetch_predict_opaque = predict_opaque;
    unlock(d);
    return 1;
}

m3_adm_admit_result_t m3_adm_admit(m3_adm_dispatcher_t *d,
                                    const m3_adm_request_desc_t *desc,
                                    m3_adm_ticket_t *ticket_out) {
    if (ticket_out) *ticket_out = (m3_adm_ticket_t){0};
    if (!d) return M3_ADM_REJECTED;
    lock(d);
    /* The inactive production seam must never reinterpret an admission
     * attempt as a planner rejection.  Fallback is checked before any
     * descriptor validation so unknown or incomplete mappings continue on
     * the unchanged engine path without changing dispatcher state. */
    if (!d->trusted_policy) { unlock(d); return M3_ADM_FALLBACK; }
    if (!ticket_out || !desc || !path_valid(d, desc)
        || !d->accepting || desc->topology_hash != d->topology_hash
        || desc->profile_hash != d->profile_hash || duplicate_identity_locked(d, desc)) {
        unlock(d); return M3_ADM_REJECTED;
    }
    m3_adm_request_t *slot = NULL;
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i)
        if (!d->requests[i].used) { slot = &d->requests[i]; break; }
    if (!slot) { unlock(d); return M3_ADM_REJECTED; }
    if (d->next_ticket_nonce == UINT64_MAX) {
        unlock(d);
        return M3_ADM_REJECTED;
    }
    uint64_t ticket_nonce = ++d->next_ticket_nonce;
    *slot = (m3_adm_request_t){0};
    slot->used = 1;
    slot->ticket_nonce = ticket_nonce;
    slot->desc = *desc;
    slot->state = M3_SHD_STATE_QUEUED;
    if (ticket_out) *ticket_out = (m3_adm_ticket_t){ desc->request_id,
        desc->generation, desc->lease_id, desc->lease_generation,
        ticket_nonce };
    if (!can_reserve_locked(d, desc)
        || (!desc->demand && !prefetch_allowed_locked(d, slot))) {
        unlock(d); return M3_ADM_QUEUED;
    }
    reserve_locked(d, slot);
    slot->state = M3_SHD_STATE_RESERVED;
    unlock(d);
    return M3_ADM_RESERVED;
}

int m3_adm_reserve_all(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket) {
    if (!d || !ticket) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    if (!d->trusted_policy || !d->accepting || !r
        || r->state != M3_SHD_STATE_QUEUED
        || !can_reserve_locked(d, &r->desc)
        || (!r->desc.demand && !prefetch_allowed_locked(d, r))) {
        unlock(d); return 0;
    }
    reserve_locked(d, r);
    r->state = M3_SHD_STATE_RESERVED;
    unlock(d);
    return 1;
}

int m3_adm_mark_issued(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket,
                       uint32_t part_id) {
    if (!d || !ticket || part_id >= M3_ADM_MAX_PARTS) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    if (!r || part_id >= r->desc.expected_parts || r->issue_closed
        || (r->state != M3_SHD_STATE_RESERVED && r->state != M3_SHD_STATE_ISSUED)
        || (r->issued_mask & (UINT32_C(1) << part_id))) { unlock(d); return 0; }
    if (!r->issued_parts) {
        for (size_t i = 0; i < r->desc.path.n_resources; ++i) {
            int ri = resource_index(d, r->desc.path.resources[i]);
            d->counters[ri].issued_requests++;
            d->counters[ri].issued_bytes += r->desc.bytes;
        }
    }
    r->issued_mask |= UINT32_C(1) << part_id;
    r->issued_parts++;
    r->state = M3_SHD_STATE_ISSUED;
    if (r->issued_parts == r->desc.expected_parts) r->issue_closed = 1;
    unlock(d);
    return 1;
}
int m3_adm_close_issue(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket) {
    if (!d || !ticket) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    if (!r || r->state != M3_SHD_STATE_ISSUED || r->issue_closed) {
        unlock(d); return 0;
    }
    r->issue_closed = 1;
    complete_if_drained_locked(d, r, M3_ADM_PART_FAILED);
    unlock(d);
    return 1;
}
int m3_adm_terminal(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket,
                    uint32_t part_id, m3_adm_part_status_t status) {
    if (!d || !ticket || part_id >= M3_ADM_MAX_PARTS
        || (status != M3_ADM_PART_COMPLETED && status != M3_ADM_PART_FAILED)) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    uint32_t bit = UINT32_C(1) << part_id;
    if (!r || r->state != M3_SHD_STATE_ISSUED || !(r->issued_mask & bit)
        || (r->terminal_mask & bit)) { unlock(d); return 0; }
    r->terminal_mask |= bit;
    r->terminal_parts++;
    if (status == M3_ADM_PART_FAILED) r->any_part_failed = 1;
    complete_if_drained_locked(d, r, status);
    unlock(d);
    return 1;
}
int m3_adm_cancel(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket) {
    if (!d || !ticket) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    if (!r) { unlock(d); return 0; }
    if (r->state == M3_SHD_STATE_QUEUED) {
        r->state = M3_SHD_STATE_CANCELLED;
        r->resources_released = 1;
    }
    else if (r->state == M3_SHD_STATE_RESERVED) {
        r->state = M3_SHD_STATE_CANCELLED; release_resources_locked(d, r);
    } else if (r->state == M3_SHD_STATE_ISSUED) {
        r->cancel_pending = 1;
        r->issue_closed = 1;
        complete_if_drained_locked(d, r, M3_ADM_PART_FAILED);
    }
    else { unlock(d); return 0; }
    unlock(d);
    return 1;
}
static int same_coalesce(const m3_adm_request_desc_t *a,
                         const m3_adm_request_desc_t *b) {
    if (!a || !b || !b->demand || a->content_id != b->content_id
        || a->selected_copy_id != b->selected_copy_id
        || a->byte_offset != b->byte_offset || a->bytes != b->bytes
        || a->topology_hash != b->topology_hash
        || a->profile_hash != b->profile_hash
        || a->path.n_resources != b->path.n_resources) return 0;
    for (size_t i = 0; i < a->path.n_resources; ++i)
        if (a->path.resources[i] != b->path.resources[i]) return 0;
    return 1;
}
static int consumer_token_in_use_locked(const m3_adm_dispatcher_t *d,
                                        const m3_adm_consumer_token_t *consumer) {
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i) {
        const m3_adm_request_t *request = &d->requests[i];
        if (!request->used) continue;
        if (request->desc.lease_id == consumer->lease_id
            && request->desc.lease_generation == consumer->lease_generation)
            return 1;
        for (size_t j = 0; j < M3_ADM_MAX_CONSUMERS; ++j)
            if (request->consumers[j].used
                && request->consumers[j].token.lease_id == consumer->lease_id
                && request->consumers[j].token.lease_generation == consumer->lease_generation)
                return 1;
    }
    return 0;
}
static int attach_consumer_locked(m3_adm_dispatcher_t *d, m3_adm_request_t *r,
                                  const m3_adm_request_desc_t *demand_match,
                                  const m3_adm_consumer_token_t *consumer) {
    if (!r || !consumer || !consumer->lease_id || !consumer->lease_generation
        || (consumer->lease_id == r->desc.lease_id
            && consumer->lease_generation == r->desc.lease_generation)
        || !demand_match || !demand_match->request_id || !demand_match->generation
        || !demand_match->lease_id || !demand_match->lease_generation
        || consumer_token_in_use_locked(d, consumer)) return 0;
    for (size_t i = 0; i < M3_ADM_MAX_CONSUMERS; ++i)
        if (!r->consumers[i].used) {
            r->consumers[i].used = 1;
            r->consumers[i].token = *consumer;
            r->consumers[i].demand_request_id = demand_match->request_id;
            r->consumers[i].demand_generation = demand_match->generation;
            r->consumers[i].demand_lease_id = demand_match->lease_id;
            r->consumers[i].demand_lease_generation = demand_match->lease_generation;
            return 1;
        }
    return 0;
}
static int all_consumers_released_locked(const m3_adm_request_t *r) {
    for (size_t i = 0; i < M3_ADM_MAX_CONSUMERS; ++i)
        if (r->consumers[i].used && !r->consumers[i].released) return 0;
    return 1;
}
m3_adm_promote_result_t m3_adm_promote_prefetch(
    m3_adm_dispatcher_t *d, const m3_adm_ticket_t *prefetch,
    const m3_adm_request_desc_t *demand_match,
    const m3_adm_consumer_token_t *consumer) {
    if (!d || !prefetch || !demand_match || !consumer) return M3_ADM_PROMOTE_REJECTED;
    lock(d);
    m3_adm_request_t *r = find_locked(d, prefetch);
    if (!r || r->desc.demand || !same_coalesce(&r->desc, demand_match)
        || (r->state != M3_SHD_STATE_QUEUED && r->state != M3_SHD_STATE_RESERVED
            && r->state != M3_SHD_STATE_ISSUED)
        || !attach_consumer_locked(d, r, demand_match, consumer)) {
        unlock(d); return M3_ADM_PROMOTE_REJECTED;
    }
    r->desc.demand = 1;
    r->prefetch_unwanted = 0;
    if (r->state == M3_SHD_STATE_QUEUED) {
        if (!d->accepting || !can_reserve_locked(d, &r->desc)) {
            unlock(d); return M3_ADM_PROMOTE_QUEUED;
        }
        reserve_locked(d, r);
        r->state = M3_SHD_STATE_RESERVED;
    }
    unlock(d);
    return M3_ADM_PROMOTED;
}
int m3_adm_release_consumer(m3_adm_dispatcher_t *d,
                            const m3_adm_ticket_t *ticket,
                            const m3_adm_consumer_token_t *consumer) {
    if (!d || !ticket || !consumer) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    if (!r || !r->resources_released) { unlock(d); return 0; }
    for (size_t i = 0; i < M3_ADM_MAX_CONSUMERS; ++i)
        if (r->consumers[i].used
            && r->consumers[i].token.lease_id == consumer->lease_id
            && r->consumers[i].token.lease_generation == consumer->lease_generation
            && !r->consumers[i].released) {
            r->consumers[i].released = 1;
            unlock(d);
            return 1;
        }
    unlock(d);
    return 0;
}
int m3_adm_cancel_prefetch(m3_adm_dispatcher_t *d,
                           const m3_adm_ticket_t *ticket) {
    if (!d || !ticket) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    if (!r || r->desc.demand) { unlock(d); return 0; }
    if (r->state == M3_SHD_STATE_QUEUED) {
        r->state = M3_SHD_STATE_CANCELLED;
        r->resources_released = 1;
    } else if (r->state == M3_SHD_STATE_RESERVED) {
        r->state = M3_SHD_STATE_CANCELLED;
        release_resources_locked(d, r);
    } else if (r->state == M3_SHD_STATE_ISSUED) {
        r->prefetch_unwanted = 1;
        unlock(d);
        return 0;
    } else { unlock(d); return 0; }
    unlock(d);
    return 1;
}
int m3_adm_release_lease(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket) {
    if (!d || !ticket) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    if (!r || r->lease_released || r->state == M3_SHD_STATE_RESERVED
        || r->state == M3_SHD_STATE_ISSUED || r->state == M3_SHD_STATE_QUEUED) {
        unlock(d); return 0;
    }
    r->lease_released = 1;
    unlock(d);
    return 1;
}
int m3_adm_retire(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket) {
    if (!d || !ticket) return 0;
    lock(d);
    m3_adm_request_t *r = find_locked(d, ticket);
    if (!r || !r->resources_released || !r->lease_released
        || !all_consumers_released_locked(r)
        || (r->state != M3_SHD_STATE_COMPLETED && r->state != M3_SHD_STATE_FAILED
            && r->state != M3_SHD_STATE_CANCELLED)) { unlock(d); return 0; }
    r->state = M3_SHD_STATE_RETIRED;
    r->used = 0;
    unlock(d);
    return 1;
}
void m3_adm_reset_begin(m3_adm_dispatcher_t *d) {
    if (!d) return;
    lock(d);
    d->accepting = 0;
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i) {
        m3_adm_request_t *r = &d->requests[i];
        if (!r->used) continue;
        if (r->state == M3_SHD_STATE_QUEUED) {
            r->state = M3_SHD_STATE_CANCELLED;
            r->resources_released = 1;
        }
        else if (r->state == M3_SHD_STATE_RESERVED) {
            r->state = M3_SHD_STATE_CANCELLED; release_resources_locked(d, r);
        } else if (r->state == M3_SHD_STATE_ISSUED) {
            r->cancel_pending = 1;
            r->issue_closed = 1;
            complete_if_drained_locked(d, r, M3_ADM_PART_FAILED);
        }
    }
    unlock(d);
}
int m3_adm_reset_drained(const m3_adm_dispatcher_t *d) {
    if (!d) return 1;
    lock((m3_adm_dispatcher_t *)d);
    int drained = 1;
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i)
        if (d->requests[i].used) drained = 0;
    unlock((m3_adm_dispatcher_t *)d);
    return drained;
}
int m3_adm_resource_snapshot(const m3_adm_dispatcher_t *d,
                             m3_shd_resource_id_t resource_id,
                             m3_adm_resource_snapshot_t *out) {
    if (out) *out = (m3_adm_resource_snapshot_t){0};
    if (!d || !out) return 0;
    lock((m3_adm_dispatcher_t *)d);
    int ri = resource_index(d, resource_id);
    if (ri >= 0) *out = (m3_adm_resource_snapshot_t){ resource_id,
        d->counters[ri].reserved_requests, d->counters[ri].issued_requests,
        d->counters[ri].reserved_bytes, d->counters[ri].issued_bytes };
    unlock((m3_adm_dispatcher_t *)d);
    return ri >= 0;
}
int m3_adm_request_snapshot(const m3_adm_dispatcher_t *d,
                            const m3_adm_ticket_t *ticket,
                            m3_adm_request_snapshot_t *out) {
    if (out) *out = (m3_adm_request_snapshot_t){0};
    if (!d || !ticket || !out) return 0;
    lock((m3_adm_dispatcher_t *)d);
    m3_adm_request_t *r = find_locked((m3_adm_dispatcher_t *)d, ticket);
    if (r) *out = (m3_adm_request_snapshot_t){ r->state, r->issued_parts,
        r->terminal_parts, r->cancel_pending, r->issue_closed, r->resources_released,
        r->lease_released };
    unlock((m3_adm_dispatcher_t *)d);
    return r != NULL;
}
size_t m3_adm_active_count(const m3_adm_dispatcher_t *d) {
    if (!d) return 0;
    lock((m3_adm_dispatcher_t *)d);
    size_t count = 0;
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i) if (d->requests[i].used) count++;
    unlock((m3_adm_dispatcher_t *)d);
    return count;
}
