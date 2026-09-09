#ifndef COLIBRI_M3_ACTIVE_ADMISSION_H
#define COLIBRI_M3_ACTIVE_ADMISSION_H

/*
 * Bounded dispatcher for the Step-8 admission seam.
 *
 * This is deliberately separate from m3_shadow_plan: the latter observes an
 * already-issued load, whereas this object owns the reservation transaction
 * that has to precede I/O publication.  It is inactive by default; callers
 * without an explicitly trusted policy receive FALLBACK and must use the
 * unchanged engine path.
 */

#include <stddef.h>
#include <stdint.h>

#include "m3_shadow_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

#define M3_ADM_MAX_REQUESTS M3_SHD_MAX_ACTIVE_REQUESTS
#define M3_ADM_MAX_PARTS 32u
#define M3_ADM_MAX_CONSUMERS 32u

typedef struct m3_adm_dispatcher m3_adm_dispatcher_t;

typedef enum {
    M3_ADM_FALLBACK = 0,
    M3_ADM_QUEUED,
    M3_ADM_RESERVED,
    M3_ADM_REJECTED
} m3_adm_admit_result_t;

typedef enum {
    M3_ADM_PART_COMPLETED = 0,
    M3_ADM_PART_FAILED = 1
} m3_adm_part_status_t;

typedef enum {
    M3_ADM_PROMOTE_REJECTED = 0,
    M3_ADM_PROMOTE_QUEUED,
    M3_ADM_PROMOTED
} m3_adm_promote_result_t;

typedef struct {
    uint64_t request_id;
    uint64_t generation;
    uint64_t topology_hash;
    uint64_t profile_hash;
    uint64_t selected_copy_id;
    uint64_t content_id;
    uint64_t byte_offset;
    uint64_t bytes;
    uint64_t lease_id;
    uint64_t lease_generation;
    uint32_t expected_parts;
    uint8_t demand;
    m3_shd_candidate_t path;
} m3_adm_request_desc_t;

typedef struct {
    uint64_t request_id;
    uint64_t generation;
    uint64_t lease_id;
    uint64_t lease_generation;
    /* Dispatcher-local nonce prevents a retired request's stale event from
     * matching a later request that reuses the external identity. */
    uint64_t ticket_nonce;
} m3_adm_ticket_t;

typedef struct {
    uint64_t lease_id;
    uint64_t lease_generation;
} m3_adm_consumer_token_t;

typedef struct {
    m3_shd_resource_id_t resource_id;
    uint32_t reserve_requests;
    uint64_t reserve_bytes;
} m3_adm_demand_reserve_t;

/* The callback is pure and has no dispatcher access.  It returns nonzero
 * only for a known prediction, and reports wait before/after adding the
 * candidate prefetch.  A prefetch is admitted only when after <= before. */
typedef int (*m3_adm_prefetch_predict_fn)(void *opaque,
                                          const m3_adm_request_desc_t *request,
                                          uint64_t *demand_wait_before_ns,
                                          uint64_t *demand_wait_after_ns);

typedef struct {
    m3_shd_resource_id_t resource_id;
    uint32_t reserved_requests;
    uint32_t issued_requests;
    uint64_t reserved_bytes;
    uint64_t issued_bytes;
} m3_adm_resource_snapshot_t;

typedef struct {
    m3_shd_request_state_t state;
    uint32_t issued_parts;
    uint32_t terminal_parts;
    uint8_t cancel_pending;
    uint8_t issue_closed;
    uint8_t resources_released;
    uint8_t lease_released;
} m3_adm_request_snapshot_t;

/* Inputs are borrowed immutable profile arrays.  trusted_policy is a hard
 * gate: false means every admission request returns FALLBACK without changing
 * state.  It is false for the current production runtime. */
m3_adm_dispatcher_t *m3_adm_open(const m3_shd_topology_profile_t *topology,
                                  uint64_t topology_hash,
                                  uint64_t profile_hash,
                                  int trusted_policy,
                                  char *err, size_t errsz);
/* Returns nonzero only when the dispatcher was freed.  This closes admission
 * before checking the ledger.  The caller must quiesce all other API calls
 * around teardown; an object pointer cannot safely outlive its dispatcher. */
int m3_adm_close(m3_adm_dispatcher_t *d);
/* Configures bounded prefetch-only policy before any request is admitted.
 * Reserve rows must name unique eligible storage resources and fit their hard
 * capacity.  NULL prediction intentionally fail-closes all prefetch. */
int m3_adm_configure_prefetch(m3_adm_dispatcher_t *d,
                              const m3_adm_demand_reserve_t *reserves,
                              size_t reserve_count,
                              m3_adm_prefetch_predict_fn predict,
                              void *predict_opaque);

/* Enqueue then atomically reserve the full path + lease.  A resource limit
 * miss leaves a QUEUED request and changes no resource/lease counter.  A
 * trusted admission requires ticket_out: the returned nonce is the only
 * supported handle for later completion, lease release, and retirement. */
m3_adm_admit_result_t m3_adm_admit(m3_adm_dispatcher_t *d,
                                    const m3_adm_request_desc_t *desc,
                                    m3_adm_ticket_t *ticket_out);
/* Retries the all-or-nothing transaction for a request retained as QUEUED. */
int m3_adm_reserve_all(m3_adm_dispatcher_t *d,
                       const m3_adm_ticket_t *ticket);

/* Publish exactly one bounded I/O part.  The request has already reserved the
 * full path, so a partial issue retains that reservation until all issued
 * parts are terminal. */
int m3_adm_mark_issued(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket,
                       uint32_t part_id);
/* Closes submission after a partial issue.  It retains the complete
 * reservation until all issued parts terminally report. */
int m3_adm_close_issue(m3_adm_dispatcher_t *d,
                       const m3_adm_ticket_t *ticket);
int m3_adm_terminal(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket,
                    uint32_t part_id, m3_adm_part_status_t status);
int m3_adm_cancel(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket);
/* Consumer/conversion retirement is the only endpoint that releases a lease.
 * It may follow terminal I/O; it never changes I/O resource counters. */
int m3_adm_release_lease(m3_adm_dispatcher_t *d,
                         const m3_adm_ticket_t *ticket);
int m3_adm_retire(m3_adm_dispatcher_t *d, const m3_adm_ticket_t *ticket);

/* Promotes the exact immutable coalescing key without another I/O request.
 * A queued prefetch is reclassified and attempts normal demand admission;
 * reserved/issued work retains its existing path and reservation. */
m3_adm_promote_result_t m3_adm_promote_prefetch(
    m3_adm_dispatcher_t *d, const m3_adm_ticket_t *prefetch,
    const m3_adm_request_desc_t *demand_match,
    const m3_adm_consumer_token_t *consumer);
int m3_adm_release_consumer(m3_adm_dispatcher_t *d,
                            const m3_adm_ticket_t *ticket,
                            const m3_adm_consumer_token_t *consumer);
/* Speculative cancellation is permitted only before issue.  An issued
 * prefetch is merely marked unwanted and retains capacity until terminal. */
int m3_adm_cancel_prefetch(m3_adm_dispatcher_t *d,
                           const m3_adm_ticket_t *ticket);

/* Reset closes admission, cancels queued requests and rolls back unissued
 * reservations.  It never releases an ISSUED request: callers must drain its
 * terminal parts before a topology or lease reuse. */
void m3_adm_reset_begin(m3_adm_dispatcher_t *d);
int m3_adm_reset_drained(const m3_adm_dispatcher_t *d);

int m3_adm_resource_snapshot(const m3_adm_dispatcher_t *d,
                             m3_shd_resource_id_t resource_id,
                             m3_adm_resource_snapshot_t *out);
int m3_adm_request_snapshot(const m3_adm_dispatcher_t *d,
                            const m3_adm_ticket_t *ticket,
                            m3_adm_request_snapshot_t *out);
size_t m3_adm_active_count(const m3_adm_dispatcher_t *d);

#ifdef __cplusplus
}
#endif

#endif
