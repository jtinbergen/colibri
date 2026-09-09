#include "../m3_active_admission.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static m3_adm_request_desc_t request(uint64_t id, uint64_t generation,
                                     uint64_t lease, uint64_t drive,
                                     uint64_t controller, uint64_t upstream) {
    m3_adm_request_desc_t r = {0};
    r.request_id = id;
    r.generation = generation;
    r.topology_hash = 77;
    r.profile_hash = 99;
    r.selected_copy_id = drive;
    r.bytes = 100;
    r.lease_id = lease;
    r.lease_generation = generation;
    r.expected_parts = 1;
    r.demand = 1;
    r.path.candidate_id = drive;
    r.path.available = 1;
    r.path.n_resources = 3;
    r.path.resources[0] = drive;
    r.path.resources[1] = controller;
    r.path.resources[2] = upstream;
    return r;
}

static void expect_zero(m3_adm_dispatcher_t *d, uint64_t resource) {
    m3_adm_resource_snapshot_t s;
    assert(m3_adm_resource_snapshot(d, resource, &s));
    assert(s.reserved_requests == 0 && s.issued_requests == 0);
    assert(s.reserved_bytes == 0 && s.issued_bytes == 0);
}

static int predict_equal(void *opaque, const m3_adm_request_desc_t *request,
                         uint64_t *before, uint64_t *after) {
    (void)opaque; (void)request;
    *before = 100;
    *after = 100;
    return 1;
}
static int predict_worse(void *opaque, const m3_adm_request_desc_t *request,
                         uint64_t *before, uint64_t *after) {
    (void)opaque; (void)request;
    *before = 100;
    *after = 101;
    return 1;
}

int main(void) {
    m3_shd_resource_t resources[] = {
        {.resource_id=101, .kind=M3_SHD_KIND_DRIVE, .parent_id=201, .max_inflight=1, .max_inflight_bytes=100, .eligible=1},
        {.resource_id=102, .kind=M3_SHD_KIND_DRIVE, .parent_id=202, .max_inflight=1, .max_inflight_bytes=100, .eligible=1},
        {.resource_id=103, .kind=M3_SHD_KIND_DRIVE, .parent_id=203, .max_inflight=1, .max_inflight_bytes=100, .eligible=1},
        {.resource_id=201, .kind=M3_SHD_KIND_CONTROLLER, .parent_id=301, .max_inflight=1, .max_inflight_bytes=100, .eligible=1},
        {.resource_id=202, .kind=M3_SHD_KIND_CONTROLLER, .parent_id=301, .max_inflight=1, .max_inflight_bytes=100, .eligible=1},
        {.resource_id=203, .kind=M3_SHD_KIND_CONTROLLER, .parent_id=303, .max_inflight=1, .max_inflight_bytes=100, .eligible=1},
        {.resource_id=301, .kind=M3_SHD_KIND_UPSTREAM, .max_inflight=1, .max_inflight_bytes=100, .eligible=1},
        {.resource_id=303, .kind=M3_SHD_KIND_UPSTREAM, .max_inflight=1, .max_inflight_bytes=100, .eligible=1},
    };
    m3_shd_topology_profile_t topology = {77, 8, resources, 0, NULL};
    char error[96] = {0};

    /* The production default is an explicit no-profile fallback. */
    m3_adm_dispatcher_t *off = m3_adm_open(&topology, 77, 99, 0,
                                            error, sizeof(error));
    assert(off);
    m3_adm_ticket_t ignored;
    m3_adm_request_desc_t fallback = request(1, 1, 1, 101, 201, 301);
    assert(m3_adm_admit(off, &fallback, &ignored)
           == M3_ADM_FALLBACK);
    assert(m3_adm_admit(off, NULL, &ignored) == M3_ADM_FALLBACK);
    fallback.path.n_resources = 1; /* inactive mode still falls back */
    assert(m3_adm_admit(off, &fallback, &ignored)
           == M3_ADM_FALLBACK);
    assert(m3_adm_active_count(off) == 0);
    m3_adm_close(off);

    m3_adm_dispatcher_t *d = m3_adm_open(&topology, 77, 99, 1,
                                          error, sizeof(error));
    assert(d);
    m3_adm_ticket_t a, shared, independent;
    m3_adm_request_desc_t no_ticket = request(9, 1, 900, 101, 201, 301);
    assert(m3_adm_admit(d, &no_ticket, NULL) == M3_ADM_REJECTED);
    assert(m3_adm_active_count(d) == 0);
    m3_adm_request_desc_t ra = request(10, 1, 1000, 101, 201, 301);
    assert(m3_adm_admit(d, &ra, &a) == M3_ADM_RESERVED);
    m3_adm_request_desc_t incomplete = request(14, 1, 1400, 101, 201, 301);
    incomplete.path.n_resources = 1; /* omits controller/upstream */
    assert(m3_adm_admit(d, &incomplete, &ignored) == M3_ADM_REJECTED);
    m3_adm_request_desc_t duplicate_lease = request(13, 1, 1000, 103, 203, 303);
    assert(m3_adm_admit(d, &duplicate_lease, &ignored) == M3_ADM_REJECTED);
    m3_adm_resource_snapshot_t s;
    assert(m3_adm_resource_snapshot(d, 301, &s));
    assert(s.reserved_requests == 1 && s.reserved_bytes == 100);

    /* Shared upstream saturation defers B atomically; neither its drive nor
     * controller acquires a partial reservation.  Independent C progresses. */
    m3_adm_request_desc_t rb = request(11, 1, 1001, 102, 202, 301);
    assert(m3_adm_admit(d, &rb, &shared) == M3_ADM_QUEUED);
    expect_zero(d, 102); expect_zero(d, 202);
    m3_adm_request_desc_t rc = request(12, 1, 1002, 103, 203, 303);
    assert(m3_adm_admit(d, &rc, &independent)
           == M3_ADM_RESERVED);
    assert(m3_adm_mark_issued(d, &independent, 0));
    assert(m3_adm_terminal(d, &independent, 0, M3_ADM_PART_COMPLETED));
    assert(m3_adm_release_lease(d, &independent));
    assert(m3_adm_retire(d, &independent));

    /* Retiring a request does not make its old ticket valid for a later
     * request that reuses the same external identity. */
    m3_adm_ticket_t reused;
    assert(m3_adm_admit(d, &rc, &reused) == M3_ADM_RESERVED);
    assert(reused.ticket_nonce != independent.ticket_nonce);
    assert(m3_adm_mark_issued(d, &reused, 0));
    assert(!m3_adm_terminal(d, &independent, 0, M3_ADM_PART_COMPLETED));
    m3_adm_request_snapshot_t reused_snapshot;
    assert(m3_adm_request_snapshot(d, &reused, &reused_snapshot));
    assert(reused_snapshot.state == M3_SHD_STATE_ISSUED
           && reused_snapshot.issued_parts == 1
           && reused_snapshot.terminal_parts == 0);
    assert(m3_adm_terminal(d, &reused, 0, M3_ADM_PART_COMPLETED));
    assert(m3_adm_release_lease(d, &reused));
    assert(m3_adm_retire(d, &reused));

    /* An issued cancellation retains all real capacity until its final,
     * generation- and part-matched completion. */
    assert(m3_adm_mark_issued(d, &a, 0));
    assert(!m3_adm_mark_issued(d, &a, 0));
    assert(m3_adm_cancel(d, &a));
    assert(!m3_adm_release_lease(d, &a));
    m3_adm_ticket_t stale = a; stale.generation++;
    assert(!m3_adm_terminal(d, &stale, 0, M3_ADM_PART_COMPLETED));
    assert(m3_adm_terminal(d, &a, 0, M3_ADM_PART_COMPLETED));
    assert(!m3_adm_terminal(d, &a, 0, M3_ADM_PART_COMPLETED));
    expect_zero(d, 101); expect_zero(d, 201); expect_zero(d, 301);
    m3_adm_request_snapshot_t rs;
    assert(m3_adm_request_snapshot(d, &a, &rs));
    assert(rs.state == M3_SHD_STATE_FAILED && rs.cancel_pending
           && rs.resources_released && !rs.lease_released);
    assert(!m3_adm_retire(d, &a));
    assert(m3_adm_release_lease(d, &a));
    assert(m3_adm_retire(d, &a));

    /* Deferred work retries the same all-or-nothing transaction only after
     * its shared upstream becomes free. */
    assert(m3_adm_reserve_all(d, &shared));
    assert(m3_adm_mark_issued(d, &shared, 0));
    assert(m3_adm_terminal(d, &shared, 0, M3_ADM_PART_COMPLETED));
    assert(m3_adm_release_lease(d, &shared));
    assert(m3_adm_retire(d, &shared));

    /* A partial issue closes submission, drains only the actually issued
     * part, and still reports failure without early capacity release. */
    m3_adm_ticket_t partial;
    m3_adm_request_desc_t rpartial = request(15, 1, 1500, 101, 201, 301);
    rpartial.expected_parts = 2;
    assert(m3_adm_admit(d, &rpartial, &partial) == M3_ADM_RESERVED);
    assert(m3_adm_mark_issued(d, &partial, 0));
    assert(m3_adm_terminal(d, &partial, 0, M3_ADM_PART_FAILED));
    /* The first part can report before submit failure/cancel closes issue. */
    assert(m3_adm_cancel(d, &partial));
    expect_zero(d, 101); expect_zero(d, 201); expect_zero(d, 301);
    assert(m3_adm_release_lease(d, &partial));
    assert(m3_adm_retire(d, &partial));

    /* Completion outcome is cumulative, not determined by the last part. */
    m3_adm_ticket_t mixed;
    m3_adm_request_desc_t rmixed = request(18, 1, 1800, 101, 201, 301);
    rmixed.expected_parts = 2;
    assert(m3_adm_admit(d, &rmixed, &mixed) == M3_ADM_RESERVED);
    assert(m3_adm_mark_issued(d, &mixed, 0));
    assert(m3_adm_mark_issued(d, &mixed, 1));
    assert(m3_adm_terminal(d, &mixed, 0, M3_ADM_PART_FAILED));
    assert(m3_adm_terminal(d, &mixed, 1, M3_ADM_PART_COMPLETED));
    assert(m3_adm_request_snapshot(d, &mixed, &rs));
    assert(rs.state == M3_SHD_STATE_FAILED);
    assert(m3_adm_release_lease(d, &mixed));
    assert(m3_adm_retire(d, &mixed));

    /* Explicit close_issue handles submit failure after only one part was
     * issued and retains the reservation until that part is terminal. */
    m3_adm_ticket_t explicitly_closed;
    m3_adm_request_desc_t rexplicit = request(19, 1, 1900, 101, 201, 301);
    rexplicit.expected_parts = 2;
    assert(m3_adm_admit(d, &rexplicit, &explicitly_closed)
           == M3_ADM_RESERVED);
    assert(m3_adm_mark_issued(d, &explicitly_closed, 0));
    assert(m3_adm_close_issue(d, &explicitly_closed));
    assert(!m3_adm_close_issue(d, &explicitly_closed));
    assert(m3_adm_terminal(d, &explicitly_closed, 0, M3_ADM_PART_COMPLETED));
    assert(m3_adm_release_lease(d, &explicitly_closed));
    assert(m3_adm_retire(d, &explicitly_closed));

    /* A never-issued queued cancellation has no resource release to perform,
     * but its lease remains a distinct lifecycle obligation. */
    m3_adm_ticket_t blocker, queued_cancel;
    m3_adm_request_desc_t rblock = request(16, 1, 1600, 101, 201, 301);
    m3_adm_request_desc_t rqueued = request(17, 1, 1700, 102, 202, 301);
    assert(m3_adm_admit(d, &rblock, &blocker) == M3_ADM_RESERVED);
    assert(m3_adm_admit(d, &rqueued, &queued_cancel) == M3_ADM_QUEUED);
    assert(m3_adm_cancel(d, &queued_cancel));
    assert(m3_adm_release_lease(d, &queued_cancel));
    assert(m3_adm_retire(d, &queued_cancel));
    assert(m3_adm_cancel(d, &blocker));
    assert(m3_adm_release_lease(d, &blocker));
    assert(m3_adm_retire(d, &blocker));

    /* Reset rolls a RESERVED request back, but drains an ISSUED one. */
    m3_adm_ticket_t reserved, issued;
    m3_adm_request_desc_t rreserved = request(20, 1, 2000, 101, 201, 301);
    m3_adm_request_desc_t rissued = request(21, 1, 2001, 103, 203, 303);
    assert(m3_adm_admit(d, &rreserved, &reserved)
           == M3_ADM_RESERVED);
    assert(m3_adm_admit(d, &rissued, &issued)
           == M3_ADM_RESERVED);
    assert(m3_adm_mark_issued(d, &issued, 0));
    m3_adm_reset_begin(d);
    expect_zero(d, 101); expect_zero(d, 201); expect_zero(d, 301);
    assert(!m3_adm_reset_drained(d));
    assert(m3_adm_terminal(d, &issued, 0, M3_ADM_PART_COMPLETED));
    assert(!m3_adm_reset_drained(d));
    assert(m3_adm_release_lease(d, &reserved));
    assert(m3_adm_retire(d, &reserved));
    assert(m3_adm_release_lease(d, &issued));
    assert(m3_adm_retire(d, &issued));
    assert(m3_adm_reset_drained(d));
    expect_zero(d, 103); expect_zero(d, 203); expect_zero(d, 303);
    assert(m3_adm_close(d));

    /* Close shuts admission before refusing a live terminal request; the
     * owner can finish lease retirement and then close successfully. */
    m3_adm_dispatcher_t *close_check = m3_adm_open(
        &topology, 77, 99, 1, error, sizeof(error));
    assert(close_check);
    m3_adm_ticket_t close_ticket;
    m3_adm_request_desc_t close_request = request(22, 1, 2200, 101, 201, 301);
    m3_adm_ticket_t queued_before_close;
    m3_adm_request_desc_t queued_request = request(23, 1, 2300, 102, 202, 301);
    assert(m3_adm_admit(close_check, &close_request, &close_ticket)
           == M3_ADM_RESERVED);
    assert(m3_adm_admit(close_check, &queued_request, &queued_before_close)
           == M3_ADM_QUEUED);
    assert(m3_adm_mark_issued(close_check, &close_ticket, 0));
    assert(m3_adm_terminal(close_check, &close_ticket, 0,
                           M3_ADM_PART_COMPLETED));
    assert(!m3_adm_close(close_check));
    m3_adm_ticket_t fresh_after_close;
    m3_adm_request_desc_t fresh_request =
        request(24, 1, 2400, 103, 203, 303);
    assert(m3_adm_admit(close_check, &fresh_request, &fresh_after_close)
           == M3_ADM_REJECTED);
    assert(!m3_adm_reserve_all(close_check, &queued_before_close));
    m3_adm_reset_begin(close_check);
    assert(m3_adm_release_lease(close_check, &close_ticket));
    assert(m3_adm_retire(close_check, &close_ticket));
    assert(m3_adm_release_lease(close_check, &queued_before_close));
    assert(m3_adm_retire(close_check, &queued_before_close));
    assert(m3_adm_close(close_check));

    /* Fixed storage is a bounded queue: the 65th identity is rejected rather
     * than overwriting an outstanding request. */
    m3_adm_dispatcher_t *full = m3_adm_open(&topology, 77, 99, 1,
                                             error, sizeof(error));
    assert(full);
    m3_adm_ticket_t full_tickets[M3_ADM_MAX_REQUESTS];
    for (uint64_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i) {
        m3_adm_request_desc_t q = request(1000 + i, 1, 3000 + i,
                                           101, 201, 301);
        m3_adm_admit_result_t result = m3_adm_admit(full, &q,
                                                    &full_tickets[i]);
        assert(result == (i == 0 ? M3_ADM_RESERVED : M3_ADM_QUEUED));
    }
    m3_adm_ticket_t overflow_ticket;
    m3_adm_request_desc_t overflow = request(2000, 1, 4000, 101, 201, 301);
    assert(m3_adm_admit(full, &overflow, &overflow_ticket) == M3_ADM_REJECTED);
    assert(!m3_adm_close(full));
    m3_adm_reset_begin(full);
    for (size_t i = 0; i < M3_ADM_MAX_REQUESTS; ++i) {
        assert(m3_adm_release_lease(full, &full_tickets[i]));
        assert(m3_adm_retire(full, &full_tickets[i]));
    }
    assert(m3_adm_reset_drained(full));
    assert(m3_adm_close(full));

    /* Step 8b: prefetch uses the same counters but cannot spend the demand
     * reserve.  Promotion attaches a distinct consumer and never re-issues. */
    m3_shd_resource_t prefetch_resources[] = {
        {.resource_id=101, .kind=M3_SHD_KIND_DRIVE, .parent_id=201, .max_inflight=2, .max_inflight_bytes=200, .eligible=1},
        {.resource_id=201, .kind=M3_SHD_KIND_CONTROLLER, .parent_id=301, .max_inflight=2, .max_inflight_bytes=200, .eligible=1},
        {.resource_id=301, .kind=M3_SHD_KIND_UPSTREAM, .max_inflight=2, .max_inflight_bytes=200, .eligible=1},
    };
    m3_shd_topology_profile_t prefetch_topology = {77, 3, prefetch_resources, 0, NULL};
    m3_adm_demand_reserve_t reserves[] = {
        {101, 1, 100}, {201, 1, 100}, {301, 1, 100},
    };
    m3_adm_dispatcher_t *prefetch_d = m3_adm_open(
        &prefetch_topology, 77, 99, 1, error, sizeof(error));
    assert(prefetch_d);
    assert(m3_adm_configure_prefetch(prefetch_d, reserves, 3,
                                     predict_equal, NULL));
    m3_adm_ticket_t prefetch_ticket, queued_prefetch;
    m3_adm_request_desc_t prefetch = request(3000, 1, 5000, 101, 201, 301);
    prefetch.demand = 0; prefetch.content_id = 123; prefetch.byte_offset = 64;
    assert(m3_adm_admit(prefetch_d, &prefetch, &prefetch_ticket)
           == M3_ADM_RESERVED);
    m3_adm_request_desc_t second_prefetch = prefetch;
    second_prefetch.request_id = 3001; second_prefetch.lease_id = 5001;
    assert(m3_adm_admit(prefetch_d, &second_prefetch, &queued_prefetch)
           == M3_ADM_QUEUED);
    m3_adm_request_desc_t demand_match = prefetch;
    demand_match.demand = 1; demand_match.request_id = 4000;
    demand_match.lease_id = 6000;
    m3_adm_consumer_token_t consumer = {7000, 1};
    assert(m3_adm_promote_prefetch(prefetch_d, &prefetch_ticket, &demand_match,
                                   &consumer) == M3_ADM_PROMOTED);
    m3_adm_request_desc_t duplicate_match = second_prefetch;
    duplicate_match.demand = 1;
    duplicate_match.request_id = 4999;
    duplicate_match.lease_id = 6999;
    assert(m3_adm_promote_prefetch(prefetch_d, &queued_prefetch,
                                   &duplicate_match, &consumer)
           == M3_ADM_PROMOTE_REJECTED);
    assert(m3_adm_mark_issued(prefetch_d, &prefetch_ticket, 0));
    assert(m3_adm_terminal(prefetch_d, &prefetch_ticket, 0,
                           M3_ADM_PART_COMPLETED));
    assert(m3_adm_request_snapshot(prefetch_d, &prefetch_ticket, &rs));
    assert(rs.issued_parts == 1 && !m3_adm_retire(prefetch_d, &prefetch_ticket));
    assert(m3_adm_release_lease(prefetch_d, &prefetch_ticket));
    assert(!m3_adm_retire(prefetch_d, &prefetch_ticket));
    assert(m3_adm_release_consumer(prefetch_d, &prefetch_ticket, &consumer));
    assert(!m3_adm_release_consumer(prefetch_d, &prefetch_ticket, &consumer));
    assert(m3_adm_retire(prefetch_d, &prefetch_ticket));

    /* Queued prefetch promotion reclassifies as normal demand admission and
     * bypasses the prefetch reserve/prediction deferral. */
    m3_adm_ticket_t blocker2;
    m3_adm_request_desc_t demand_blocker = request(3002, 1, 5002, 101, 201, 301);
    assert(m3_adm_admit(prefetch_d, &demand_blocker, &blocker2) == M3_ADM_RESERVED);
    m3_adm_request_desc_t queued_match = second_prefetch;
    queued_match.demand = 1; queued_match.request_id = 4001; queued_match.lease_id = 6001;
    m3_adm_consumer_token_t queued_consumer = {7001, 1};
    assert(m3_adm_promote_prefetch(prefetch_d, &queued_prefetch, &queued_match,
                                   &queued_consumer) == M3_ADM_PROMOTED);
    assert(m3_adm_cancel(prefetch_d, &blocker2));
    assert(m3_adm_release_lease(prefetch_d, &blocker2));
    assert(m3_adm_retire(prefetch_d, &blocker2));
    assert(m3_adm_mark_issued(prefetch_d, &queued_prefetch, 0));
    assert(m3_adm_terminal(prefetch_d, &queued_prefetch, 0, M3_ADM_PART_COMPLETED));
    assert(m3_adm_release_lease(prefetch_d, &queued_prefetch));
    assert(m3_adm_release_consumer(prefetch_d, &queued_prefetch, &queued_consumer));
    assert(m3_adm_retire(prefetch_d, &queued_prefetch));

    assert(m3_adm_configure_prefetch(prefetch_d, reserves, 3, predict_worse, NULL));
    m3_adm_ticket_t bad_prediction;
    m3_adm_request_desc_t bad = prefetch;
    bad.request_id = 3003; bad.lease_id = 5003;
    assert(m3_adm_admit(prefetch_d, &bad, &bad_prediction) == M3_ADM_QUEUED);
    assert(!m3_adm_reserve_all(prefetch_d, &bad_prediction));
    assert(m3_adm_cancel_prefetch(prefetch_d, &bad_prediction));
    assert(m3_adm_release_lease(prefetch_d, &bad_prediction));
    assert(m3_adm_retire(prefetch_d, &bad_prediction));
    m3_adm_close(prefetch_d);
    puts("m3 active admission dispatcher: PASS");
    return 0;
}
