#include "m3_shadow_plan.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

typedef struct {
    uint8_t used;
    uint64_t id;
    m3_shd_request_t request;
    m3_shd_candidate_t path;
    uint64_t predicted_ready_ns;
} m3_shd_active_t;

struct m3_shadow_plan {
    const m3_shd_topology_profile_t *topology;
    const m3_shd_calibration_set_t *calibration;
    const m3_shd_conversion_profile_t *conversion;
    m3_shd_clock_t clock;
    uint64_t profile_hash;
    uint64_t next_request_id;
    atomic_flag guard;
    size_t queued_count;
    uint64_t queued_bytes;
    m3_shd_request_t queued[M3_SHD_MAX_QUEUED_REQUESTS];
    uint8_t queued_used[M3_SHD_MAX_QUEUED_REQUESTS];
    m3_shd_observer_stats_t observer_stats;
    size_t active_count;
    m3_shd_active_t active[M3_SHD_MAX_ACTIVE_REQUESTS];
    char log[M3_SHD_MAX_DECISION_LOG_BYTES];
    size_t log_bytes;
};

static int real_resource_occupancy_locked(
    const m3_shadow_plan_t *p, m3_shd_resource_id_t resource_id,
    m3_shd_resource_occupancy_t *out);
static int real_path_occupancy_locked(
    const m3_shadow_plan_t *p, const m3_shd_candidate_t *path,
    m3_shd_resource_occupancy_t *out, size_t cap);

static void planner_lock(m3_shadow_plan_t *p) {
    while (atomic_flag_test_and_set_explicit(&p->guard, memory_order_acquire)) {
        /* The planner has a fixed-size ledger and no blocking I/O. */
    }
}

static void planner_unlock(m3_shadow_plan_t *p) {
    atomic_flag_clear_explicit(&p->guard, memory_order_release);
}

typedef struct {
    const m3_shd_residency_candidate_t *candidate;
    m3_shd_residency_action_kind_t kind;
    uint64_t numerator;
    uint64_t denominator;
} m3_shd_residency_ranked_t;

/* Compare positive rational numbers without multiplication overflow.  When
 * integer quotients tie, reciprocal Euclidean steps reverse the comparison. */
static int residency_fraction_cmp(uint64_t an, uint64_t ad,
                                  uint64_t bn, uint64_t bd) {
    int direction = 1;
    for (;;) {
        uint64_t aq = an / ad, bq = bn / bd;
        if (aq != bq) return (aq > bq ? 1 : -1) * direction;
        uint64_t ar = an % ad, br = bn % bd;
        if (!ar || !br) {
            if (!ar && !br) return 0;
            return (!ar ? -1 : 1) * direction;
        }
        an = ad; ad = ar;
        bn = bd; bd = br;
        direction = -direction;
    }
}

static int residency_rank_cmp(const m3_shd_residency_ranked_t *a,
                              const m3_shd_residency_ranked_t *b) {
    int score = residency_fraction_cmp(a->numerator, a->denominator,
                                       b->numerator, b->denominator);
    if (score) return score > 0 ? -1 : 1;
    const m3_shd_residency_candidate_t *ac = a->candidate;
    const m3_shd_residency_candidate_t *bc = b->candidate;
    if (ac->expected_wait_saved_ns != bc->expected_wait_saved_ns)
        return ac->expected_wait_saved_ns > bc->expected_wait_saved_ns ? -1 : 1;
    if (ac->content_id != bc->content_id)
        return ac->content_id < bc->content_id ? -1 : 1;
    if (ac->content_generation != bc->content_generation)
        return ac->content_generation < bc->content_generation ? -1 : 1;
    if (ac->source_domain != bc->source_domain)
        return ac->source_domain < bc->source_domain ? -1 : 1;
    if (ac->target_domain != bc->target_domain)
        return ac->target_domain < bc->target_domain ? -1 : 1;
    return (int)a->kind - (int)b->kind;
}

static int residency_budget_index(const m3_shd_residency_budget_t *budgets,
                                  size_t count, m3_shd_resource_id_t domain) {
    for (size_t i = 0; i < count; ++i)
        if (budgets[i].domain_id == domain) return (int)i;
    return -1;
}

static int residency_duplicate_content(
    const m3_shd_residency_candidate_t *candidates, size_t count, size_t index) {
    for (size_t i = 0; i < count; ++i) {
        if (i == index) continue;
        if (candidates[i].content_id == candidates[index].content_id
            && candidates[i].content_generation
                == candidates[index].content_generation) return 1;
    }
    return 0;
}

int m3_shd_plan_residency(
    const m3_shd_residency_candidate_t *candidates, size_t candidate_count,
    const m3_shd_residency_budget_t *budgets, size_t budget_count,
    uint64_t cache_snapshot_generation,
    m3_shd_residency_action_t *out, size_t out_cap, size_t *out_count) {
    if (out_count) *out_count = 0;
    if (!candidates || !budgets || !out || !out_count || !cache_snapshot_generation
        || candidate_count > M3_SHD_MAX_RESIDENCY_CANDIDATES
        || budget_count == 0 || budget_count > M3_SHD_MAX_RESOURCES
        || out_cap == 0) return 0;
    m3_shd_residency_budget_t remaining[M3_SHD_MAX_RESOURCES];
    for (size_t i = 0; i < budget_count; ++i) {
        if (!budgets[i].domain_id) return 0;
        for (size_t j = 0; j < i; ++j)
            if (budgets[j].domain_id == budgets[i].domain_id) return 0;
        remaining[i] = budgets[i];
    }
    m3_shd_residency_ranked_t ranked[M3_SHD_MAX_RESIDENCY_CANDIDATES];
    size_t ranked_count = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        const m3_shd_residency_candidate_t *c = &candidates[i];
        if (!c->content_id || !c->content_generation || !c->bytes || c->in_use
            || residency_duplicate_content(candidates, candidate_count, i)) continue;
        m3_shd_residency_action_kind_t kind = 0;
        m3_shd_resource_id_t budget_domain = c->target_domain;
        if (c->resident) {
            if (c->cache_snapshot_generation != cache_snapshot_generation
                || !c->resident_domain || !c->resident_slot_id) continue;
            if (c->target_eligible && c->target_domain == c->resident_domain
                && !c->pinned) kind = M3_SHD_RESIDENCY_PIN;
            else if (c->source_available && c->target_eligible
                     && c->target_domain && c->target_domain != c->resident_domain)
                kind = M3_SHD_RESIDENCY_MOVE;
            else if (c->evictable && !c->pinned) {
                kind = M3_SHD_RESIDENCY_EVICT;
                budget_domain = c->resident_domain;
            }
        } else if (c->source_available && c->target_eligible && c->target_domain) {
            kind = M3_SHD_RESIDENCY_MOVE;
        }
        if (!kind || residency_budget_index(remaining, budget_count, budget_domain) < 0)
            continue;
        uint64_t net = c->expected_wait_saved_ns > c->movement_cost_ns
            ? c->expected_wait_saved_ns - c->movement_cost_ns : 0;
        ranked[ranked_count++] = (m3_shd_residency_ranked_t){c, kind, net, c->bytes};
    }
    for (size_t i = 1; i < ranked_count; ++i) {
        m3_shd_residency_ranked_t value = ranked[i];
        size_t j = i;
        while (j && residency_rank_cmp(&value, &ranked[j - 1]) < 0) {
            ranked[j] = ranked[j - 1];
            --j;
        }
        ranked[j] = value;
    }
    m3_shd_residency_action_t chosen[M3_SHD_MAX_RESIDENCY_ACTIONS];
    size_t chosen_count = 0;
    for (size_t i = 0; i < ranked_count; ++i) {
        const m3_shd_residency_candidate_t *c = ranked[i].candidate;
        int bi = residency_budget_index(remaining, budget_count,
            ranked[i].kind == M3_SHD_RESIDENCY_EVICT
                ? c->resident_domain : c->target_domain);
        if (bi < 0) continue;
        if (ranked[i].kind == M3_SHD_RESIDENCY_MOVE) {
            if (!remaining[bi].available_slots
                || c->bytes > remaining[bi].available_bytes) continue;
            remaining[bi].available_slots--;
            remaining[bi].available_bytes -= c->bytes;
        }
        m3_shd_residency_action_t *a = &chosen[chosen_count++];
        *a = (m3_shd_residency_action_t){
            ranked[i].kind, c->content_id, c->content_generation,
            c->source_domain, ranked[i].kind == M3_SHD_RESIDENCY_EVICT
                ? c->resident_domain : c->target_domain,
            c->resident_slot_id, c->bytes, c->expected_wait_saved_ns,
            c->movement_cost_ns, c->cache_snapshot_generation
        };
    }
    if (chosen_count > out_cap) return 0;
    memcpy(out, chosen, chosen_count * sizeof(chosen[0]));
    *out_count = chosen_count;
    return 1;
}

typedef struct {
    uint32_t inflight;
    uint64_t bytes;
} m3_shd_count_t;

typedef struct {
    const m3_shd_request_t *request;
    const m3_shd_candidate_t *path;
    uint64_t remaining;
    uint64_t start_ns;
    uint64_t ready_ns;
    uint64_t uncertainty_ns;
    int state; /* 0 waiting, 1 transferring, 2 complete */
} m3_shd_flow_t;

static void set_error(char *err, size_t errsz, const char *msg) {
    if (err && errsz) {
        snprintf(err, errsz, "%s", msg);
    }
}

static int parse_error(char *err, size_t errsz, const char *msg,
                       size_t line_no) {
    if (err && errsz) snprintf(err, errsz, "line %zu: %s", line_no, msg);
    return -1;
}

static uint64_t fnv1a_u64(uint64_t h, uint64_t x) {
    for (unsigned i = 0; i < 8; ++i) {
        h ^= (uint8_t)(x >> (i * 8));
        h *= UINT64_C(1099511628211);
    }
    return h;
}

int m3_shd_parse_text(const char *text, uint64_t topology_hash,
                      m3_shd_resource_t *resources, size_t resource_cap,
                      size_t *resource_count,
                      m3_shd_resource_profile_t *profiles,
                      size_t profile_cap, size_t *profile_count,
                      m3_shd_weight_copy_t *copies, size_t copy_cap,
                      size_t *copy_count,
                      m3_shd_topology_profile_t *topology,
                      m3_shd_calibration_set_t *calibration,
                      char *err, size_t errsz) {
    if (!text || !resources || !resource_count || !profiles ||
        !profile_count || !copies || !copy_count || !topology || !calibration || !resource_cap ||
        !profile_cap || resource_cap > M3_SHD_MAX_RESOURCES ||
        profile_cap > M3_SHD_MAX_RESOURCES || copy_cap > M3_SHD_MAX_COPIES) {
        set_error(err, errsz, "invalid parser buffers");
        return -1;
    }
    memset(resources, 0, resource_cap * sizeof(*resources));
    memset(profiles, 0, profile_cap * sizeof(*profiles));
    memset(copies, 0, copy_cap * sizeof(*copies));
    memset(calibration, 0, sizeof(*calibration));
    *resource_count = 0;
    *profile_count = 0;
    *copy_count = 0;
    size_t line_no = 0;
    const char *cursor = text;
    while (*cursor) {
        char line[512];
        size_t len = 0;
        while (cursor[len] && cursor[len] != '\n') len++;
        if (len >= sizeof(line)) return parse_error(err, errsz,
                                                     "line too long", line_no + 1);
        memcpy(line, cursor, len);
        line[len] = '\0';
        cursor += len;
        if (*cursor == '\n') cursor++;
        line_no++;
        char *s = line;
        while (*s == ' ' || *s == '\t' || *s == '\r') s++;
        if (!*s || *s == '#') continue;

        char tag[16] = {0};
        unsigned long long id = 0, parent = 0, max_bytes = 0;
        unsigned kind = 0, max_inflight = 0;
        int n = sscanf(s, "%15s %llu %u %llu %u %llu", tag, &id, &kind,
                       &parent, &max_inflight, &max_bytes);
        if (!strcmp(tag, "resource")) {
            if (n != 6 || !id || kind >= M3_SHD_KIND_LAST || !max_inflight
                || !max_bytes) {
                return parse_error(err, errsz, "invalid resource row", line_no);
            }
            if (*resource_count >= resource_cap)
                return parse_error(err, errsz, "resource capacity exceeded", line_no);
            for (size_t i = 0; i < *resource_count; ++i)
                if (resources[i].resource_id == (uint64_t)id)
                    return parse_error(err, errsz, "duplicate resource id", line_no);
            m3_shd_resource_t f = {0};
            f.resource_id = (uint64_t)id;
            f.kind = (m3_shd_resource_kind_t)kind;
            f.parent_id = (uint64_t)parent;
            f.max_inflight = max_inflight;
            f.max_inflight_bytes = (uint64_t)max_bytes;
            f.eligible = 1;
            resources[*resource_count] = f;
            (*resource_count)++;
            continue;
        }

        /* Slice 7a-ext: kind-specific lines for compute islands, memory
         * domains and interconnect links.  Each line stores its payload in
         * the corresponding m3_shd_resource_t fields; the parser enforces
         * no duplicate ids, no unknown kinds, and no negative numerics.
         * Storage-style max_inflight / max_inflight_bytes are NOT set here:
         * the planner's open-time bound check only enforces them for
         * storage kinds (drive / controller / upstream); the new kinds
         * have no admission semantics and remain zero, which is read by
         * m3_shd_open as "not applicable". */
        if (!strcmp(tag, "cpu_island")) {
            unsigned long long parent_mem = 0, isa_bits = 0, cores = 0;
            unsigned long long eligible = 1;
            int nn = sscanf(s, "%15s %llu %llu %llu %llu %llu", tag, &id,
                            &parent_mem, &isa_bits, &cores, &eligible);
            if (nn < 5 || !id || !cores
                || cores > UINT32_MAX || isa_bits > UINT32_MAX
                || eligible > 1) {
                return parse_error(err, errsz, "invalid cpu_island row", line_no);
            }
            if (*resource_count >= resource_cap)
                return parse_error(err, errsz,
                                   "resource capacity exceeded", line_no);
            for (size_t i = 0; i < *resource_count; ++i)
                if (resources[i].resource_id == (uint64_t)id)
                    return parse_error(err, errsz,
                                       "duplicate resource id", line_no);
            m3_shd_resource_t f = {0};
            f.resource_id = (uint64_t)id;
            f.kind = M3_SHD_KIND_CPU_ISLAND;
            f.isa_caps = (uint32_t)isa_bits;
            f.cores = (uint32_t)cores;
            f.parent_id = (uint64_t)parent_mem; /* mirrored for cycle walker */
            f.parent_mem_id = (uint64_t)parent_mem;
            f.eligible = (uint8_t)eligible;
            resources[*resource_count] = f;
            (*resource_count)++;
            continue;
        }

        if (!strcmp(tag, "gpu_island")) {
            unsigned long long parent_link = 0, vram = 0, p2p = 0;
            unsigned long long pcie_parent = 0, eligible = 1;
            int nn = sscanf(s, "%15s %llu %llu %llu %llu %llu %llu", tag,
                            &id, &parent_link, &vram, &p2p, &pcie_parent,
                            &eligible);
            if (nn < 6 || !id || p2p > UINT16_MAX || vram > UINT64_MAX
                || eligible > 1) {
                return parse_error(err, errsz, "invalid gpu_island row", line_no);
            }
            if (*resource_count >= resource_cap)
                return parse_error(err, errsz,
                                   "resource capacity exceeded", line_no);
            for (size_t i = 0; i < *resource_count; ++i)
                if (resources[i].resource_id == (uint64_t)id)
                    return parse_error(err, errsz,
                                       "duplicate resource id", line_no);
            m3_shd_resource_t f = {0};
            f.resource_id = (uint64_t)id;
            f.kind = M3_SHD_KIND_GPU_ISLAND;
            f.vram_bytes = (uint64_t)vram;
            f.p2p_class = (uint16_t)p2p;
            f.parent_id = (uint64_t)parent_link; /* mirrored for cycle walker */
            f.pcie_parent_id = (uint64_t)pcie_parent;
            f.parent_mem_id = (uint64_t)parent_link;
            f.eligible = (uint8_t)eligible;
            resources[*resource_count] = f;
            (*resource_count)++;
            continue;
        }

        if (!strcmp(tag, "memory_dom")) {
            unsigned long long parent_id_v = 0, rate = 0, bw_class = 0;
            int nn = sscanf(s, "%15s %llu %llu %llu %llu", tag, &id,
                            &parent_id_v, &rate, &bw_class);
            if (nn != 5 || !id || bw_class > UINT32_MAX) {
                return parse_error(err, errsz, "invalid memory_dom row", line_no);
            }
            if (*resource_count >= resource_cap)
                return parse_error(err, errsz,
                                   "resource capacity exceeded", line_no);
            for (size_t i = 0; i < *resource_count; ++i)
                if (resources[i].resource_id == (uint64_t)id)
                    return parse_error(err, errsz,
                                       "duplicate resource id", line_no);
            m3_shd_resource_t f = {0};
            f.resource_id = (uint64_t)id;
            f.kind = M3_SHD_KIND_MEMORY_DOMAIN;
            f.parent_id = (uint64_t)parent_id_v;
            f.rate_bytes_per_s = (uint64_t)rate;
            f.bandwidth_class = (uint32_t)bw_class;
            f.eligible = 1;
            resources[*resource_count] = f;
            (*resource_count)++;
            continue;
        }

        if (!strcmp(tag, "link")) {
            unsigned long long parent_cpu = 0, pcie_parent = 0, rate = 0;
            int nn = sscanf(s, "%15s %llu %llu %llu %llu", tag, &id,
                            &parent_cpu, &pcie_parent, &rate);
            if (nn != 5 || !id) {
                return parse_error(err, errsz, "invalid link row", line_no);
            }
            if (*resource_count >= resource_cap)
                return parse_error(err, errsz,
                                   "resource capacity exceeded", line_no);
            for (size_t i = 0; i < *resource_count; ++i)
                if (resources[i].resource_id == (uint64_t)id)
                    return parse_error(err, errsz,
                                       "duplicate resource id", line_no);
            m3_shd_resource_t f = {0};
            f.resource_id = (uint64_t)id;
            f.kind = M3_SHD_KIND_LINK;
            f.parent_id = (uint64_t)pcie_parent; /* mirrored for cycle walker */
            f.pcie_parent_id = (uint64_t)pcie_parent;
            f.parent_mem_id = (uint64_t)parent_cpu;
            f.rate_bytes_per_s = (uint64_t)rate;
            f.eligible = 1;
            resources[*resource_count] = f;
            (*resource_count)++;
            continue;
        }

        unsigned long long copy_id = 0, copy_model = 0, copy_tensor = 0;
        unsigned long long copy_bytes = 0, copy_drive = 0;
        unsigned copy_available = 0;
        n = sscanf(s, "%15s %llu %llu %llu %llu %llu %u", tag,
                   &copy_id, &copy_model, &copy_tensor, &copy_bytes,
                   &copy_drive, &copy_available);
        if (!strcmp(tag, "copy")) {
            if (n != 7 || !copy_id || !copy_model || !copy_bytes
                || !copy_drive || copy_available > 1
                || *copy_count >= copy_cap) {
                return parse_error(err, errsz, "invalid copy row", line_no);
            }
            for (size_t i = 0; i < *copy_count; ++i)
                if (copies[i].copy_id == (uint64_t)copy_id)
                    return parse_error(err, errsz, "duplicate copy id", line_no);
            copies[*copy_count] = (m3_shd_weight_copy_t){
                (uint64_t)copy_id, (uint64_t)copy_model,
                (uint64_t)copy_tensor, (uint64_t)copy_bytes,
                (uint64_t)copy_drive, (uint8_t)copy_available
            };
            (*copy_count)++;
            continue;
        }

        unsigned conversion_max_inflight = 0;
        unsigned conversion_size_class = 0, conversion_load_class = 0;
        unsigned long long conversion_max_bytes = 0;
        unsigned long long conversion_duration = 0;
        unsigned long long conversion_residual = 0;
        unsigned long long conversion_samples = 0;
        unsigned long long conversion_tail_percentile = 0;
        n = sscanf(s, "%15s %u %llu %u %u %llu %llu %llu %llu", tag,
                   &conversion_max_inflight, &conversion_max_bytes,
                   &conversion_size_class, &conversion_load_class,
                   &conversion_duration, &conversion_residual,
                   &conversion_samples, &conversion_tail_percentile);
        if (!strcmp(tag, "conversion")) {
            if ((n != 8 && n != 9) || !conversion_max_inflight
                || !conversion_max_bytes
                || conversion_size_class >= M3_SHD_SIZE_CLASS_COUNT - 1
                || conversion_load_class >= M3_SHD_LOAD_CLASS_COUNT - 1
                || !conversion_duration || !conversion_samples
                || conversion_samples > UINT32_MAX
                || conversion_tail_percentile > UINT32_MAX
                || (conversion_residual && n != 9)
                || (conversion_residual
                    && conversion_tail_percentile != M3_SHD_TAIL_PERCENTILE_P95_PPM
                    && conversion_tail_percentile != M3_SHD_TAIL_PERCENTILE_P99_PPM)
                || (!conversion_residual && conversion_tail_percentile != 0)) {
                return parse_error(err, errsz, "invalid conversion row", line_no);
            }
            if (!calibration->conversion.enabled) {
                calibration->conversion.enabled = 1;
                calibration->conversion.version = M3_SHD_CALIB_VERSION;
                calibration->conversion.topology_hash = topology_hash;
                calibration->conversion.max_inflight = conversion_max_inflight;
                calibration->conversion.max_inflight_bytes =
                    (uint64_t)conversion_max_bytes;
            } else if (calibration->conversion.max_inflight
                       != conversion_max_inflight
                       || calibration->conversion.max_inflight_bytes
                       != (uint64_t)conversion_max_bytes) {
                return parse_error(err, errsz,
                                   "contradictory conversion bounds", line_no);
            }
            if (conversion_samples > calibration->conversion.sample_count)
                calibration->conversion.sample_count =
                    (uint32_t)conversion_samples;
            calibration->conversion.duration_ns[conversion_size_class]
                [conversion_load_class] = (uint64_t)conversion_duration;
            calibration->conversion.residual_ns[conversion_size_class]
                [conversion_load_class] = (uint64_t)conversion_residual;
            calibration->conversion.residual_percentile_ppm[conversion_size_class]
                [conversion_load_class] = (uint32_t)conversion_tail_percentile;
            continue;
        }

        unsigned long long queue_age_ns = 0;
        n = sscanf(s, "%15s %llu", tag, &queue_age_ns);
        if (!strcmp(tag, "queue_age_ns")) {
            if (n != 2 || !queue_age_ns || calibration->max_queue_age_ns)
                return parse_error(err, errsz,
                                   "invalid or duplicate queue age", line_no);
            calibration->max_queue_age_ns = (uint64_t)queue_age_ns;
            continue;
        }

        unsigned size_class = 0, load_class = 0;
        unsigned long long rate = 0, startup = 0, residual = 0, samples = 0;
        unsigned long long tail_percentile = 0;
        n = sscanf(s, "%15s %llu %u %u %llu %llu %llu %llu %llu", tag, &id,
                   &size_class, &load_class, &rate, &startup,
                   &residual, &samples, &tail_percentile);
        if (!strcmp(tag, "profile")) {
            if ((n != 8 && n != 9) || !id
                || (residual && n != 9)
                || size_class >= M3_SHD_SIZE_CLASS_COUNT - 1
                || load_class >= M3_SHD_LOAD_CLASS_COUNT - 1 || !rate
                || !samples || samples > UINT32_MAX
                || tail_percentile > UINT32_MAX
                || (residual
                    && tail_percentile != M3_SHD_TAIL_PERCENTILE_P95_PPM
                    && tail_percentile != M3_SHD_TAIL_PERCENTILE_P99_PPM)
                || (!residual && tail_percentile != 0)) {
                return parse_error(err, errsz, "invalid profile row", line_no);
            }
            size_t pi = 0;
            for (; pi < *profile_count; ++pi)
                if (profiles[pi].resource_id == (uint64_t)id) break;
            if (pi == *profile_count) {
                if (*profile_count >= profile_cap)
                    return parse_error(err, errsz, "profile capacity exceeded", line_no);
                profiles[pi].resource_id = (uint64_t)id;
                profiles[pi].topology_hash = topology_hash;
                profiles[pi].units_per_second_explicit = 1;
                profiles[pi].version = M3_SHD_CALIB_VERSION;
                (*profile_count)++;
            }
            if (samples > profiles[pi].sample_count)
                profiles[pi].sample_count = samples;
            profiles[pi].rate_bytes_per_s[size_class][load_class] = rate;
            profiles[pi].startup_ns[size_class][load_class] = startup;
            profiles[pi].residual_ns[size_class][load_class] = residual;
            profiles[pi].residual_percentile_ppm[size_class][load_class] =
                (uint32_t)tail_percentile;
            continue;
        }
        return parse_error(err, errsz, "unknown calibration row", line_no);
    }
    /* A topology may be partially calibrated.  Missing resource profiles
     * are deliberately retained as UNKNOWN by the predictor; they must not
     * be replaced with inferred capacity. */
    if (!*resource_count)
        return parse_error(err, errsz, "resource/profile set is incomplete", line_no);
    topology->topology_hash = topology_hash;
    topology->n_resources = *resource_count;
    topology->resources = resources;
    topology->n_copies = *copy_count;
    topology->copies = *copy_count ? copies : NULL;
    calibration->topology_hash = topology_hash;
    calibration->topology = topology;
    calibration->n_profiles = *profile_count;
    calibration->profiles = profiles;
    calibration->version = M3_SHD_CALIB_VERSION;
    if (!m3_shd_calib_is_sound(calibration))
        return parse_error(err, errsz, "calibration is not sound", line_no);
    return 0;
}

static int resource_index(const m3_shadow_plan_t *p,
                          m3_shd_resource_id_t id) {
    for (size_t i = 0; i < p->topology->n_resources; ++i) {
        if (p->topology->resources[i].resource_id == id) return (int)i;
    }
    return -1;
}

static const m3_shd_resource_profile_t *profile_for(
    const m3_shadow_plan_t *p, m3_shd_resource_id_t id) {
    for (size_t i = 0; i < p->calibration->n_profiles; ++i) {
        if (p->calibration->profiles[i].resource_id == id)
            return &p->calibration->profiles[i];
    }
    return NULL;
}

int m3_shd_expand_candidate(const m3_shadow_plan_t *p,
                            m3_shd_candidate_t *candidate) {
    if (!p || !p->topology || !candidate || !candidate->available
        || candidate->n_resources == 0
        || candidate->n_resources > M3_SHD_MAX_PATH_RESOURCES) return -1;
    if (candidate->n_resources != 1) return 0;

    int ri = resource_index(p, candidate->resources[0]);
    if (ri < 0 || p->topology->resources[ri].kind != M3_SHD_KIND_DRIVE)
        return -1;

    m3_shd_resource_id_t parent = p->topology->resources[ri].parent_id;
    size_t hops = 0;
    while (parent) {
        if (candidate->n_resources >= M3_SHD_MAX_PATH_RESOURCES
            || ++hops > p->topology->n_resources) return -1;
        int pi = resource_index(p, parent);
        if (pi < 0) return -1;
        for (size_t i = 0; i < candidate->n_resources; ++i)
            if (candidate->resources[i] == parent) return -1;
        candidate->resources[candidate->n_resources++] = parent;
        parent = p->topology->resources[pi].parent_id;
    }
    return 0;
}

int m3_shd_candidates_for_key(const m3_shadow_plan_t *p,
                              uint64_t model_id, ColiExpertKey key,
                              uint64_t bytes,
                              m3_shd_resource_id_t actual_drive_id,
                              m3_shd_candidate_t *out, size_t cap,
                              size_t *count_out) {
    if (count_out) *count_out = 0;
    if (!p || !p->topology || !out || !cap || !bytes || !model_id
        || key.layer < 0 || key.expert < 0 || !count_out) return 0;
    uint64_t tensor_id = ((uint64_t)(uint32_t)key.layer << 32)
                       | (uint32_t)key.expert;
    size_t n = 0;
    int matched_copy_identity = 0;
    for (size_t i = 0; i < p->topology->n_copies; ++i) {
        const m3_shd_weight_copy_t *copy = &p->topology->copies[i];
        if (copy->model_id != model_id || copy->tensor_id != tensor_id)
            continue;
        matched_copy_identity = 1;
        if (copy->bytes != bytes || !copy->available) continue;
        int duplicate = 0;
        for (size_t j = 0; j < n; ++j)
            if (out[j].candidate_id == copy->drive_id) duplicate = 1;
        if (duplicate) continue;
        if (n >= cap) return 0;
        out[n] = (m3_shd_candidate_t){0};
        out[n].candidate_id = copy->drive_id;
        out[n].n_resources = 1;
        out[n].resources[0] = copy->drive_id;
        out[n].available = 1;
        if (m3_shd_expand_candidate(p, &out[n])) return 0;
        n++;
    }
    /* A fallback actual drive is safe only when configuration contains no
     * row for this model/tensor/byte identity.  If rows exist but are all
     * unavailable, silently inventing an actual candidate would violate the
     * mapping contract and turn a missing replica into a planable one. */
    if (!n && !matched_copy_identity && actual_drive_id) {
        if (resource_index(p, actual_drive_id) < 0 || n >= cap) return 0;
        out[n] = (m3_shd_candidate_t){0};
        out[n].candidate_id = actual_drive_id;
        out[n].n_resources = 1;
        out[n].resources[0] = actual_drive_id;
        out[n].available = 1;
        if (m3_shd_expand_candidate(p, &out[n])) return 0;
        n++;
    }
    *count_out = n;
    return n != 0;
}

static int path_valid(const m3_shadow_plan_t *p,
                      const m3_shd_candidate_t *path) {
    if (!path || !path->available || path->n_resources == 0
        || path->n_resources > M3_SHD_MAX_PATH_RESOURCES) return 0;
    for (size_t i = 0; i < path->n_resources; ++i) {
        if (resource_index(p, path->resources[i]) < 0) return 0;
        for (size_t j = 0; j < i; ++j) {
            if (path->resources[i] == path->resources[j]) return 0;
        }
    }
    /* A candidate cannot bypass a declared shared parent.  This is the
     * admission invariant that prevents a drive request from accidentally
     * receiving independent-controller capacity in the simulator. */
    for (size_t i = 0; i < path->n_resources; ++i) {
        int ri = resource_index(p, path->resources[i]);
        m3_shd_resource_id_t parent = p->topology->resources[ri].parent_id;
        size_t hops = 0;
        while (parent) {
            int pi = resource_index(p, parent);
            if (pi < 0) return 0;
            int present = 0;
            for (size_t j = 0; j < path->n_resources; ++j)
                if (path->resources[j] == parent) present = 1;
            if (!present || ++hops > p->topology->n_resources) return 0;
            parent = p->topology->resources[pi].parent_id;
        }
    }
    return 1;
}

static uint64_t sat_add(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

/* The caller owns the synchronization boundary.  Clock callbacks are
 * external code, so obtain their value before taking planner_lock; otherwise
 * a callback that observes this planner can deadlock on the same guard. */
static uint64_t planner_now(const m3_shadow_plan_t *p) {
    return p && p->clock.now ? p->clock.now(p->clock.opaque) : 0;
}

int m3_shd_request_priority_cmp(const m3_shd_request_t *a,
                                const m3_shd_request_t *b) {
    if (!a || !b) return 0;
    if (a->consumer_need_ns != b->consumer_need_ns)
        return a->consumer_need_ns < b->consumer_need_ns ? -1 : 1;
    if (a->enqueue_ns != b->enqueue_ns)
        return a->enqueue_ns < b->enqueue_ns ? -1 : 1;
    if (a->request_id != b->request_id)
        return a->request_id < b->request_id ? -1 : 1;
    return 0;
}

static int remove_queued_locked(m3_shadow_plan_t *p, uint64_t request_id) {
    for (size_t i = 0; i < M3_SHD_MAX_QUEUED_REQUESTS; ++i) {
        if (p->queued_used[i] && p->queued[i].request_id == request_id) {
            if (p->queued_bytes >= p->queued[i].bytes)
                p->queued_bytes -= p->queued[i].bytes;
            else
                p->queued_bytes = 0;
            memset(&p->queued[i], 0, sizeof(p->queued[i]));
            p->queued_used[i] = 0;
            p->queued_count--;
            return 1;
        }
    }
    return 0;
}

static size_t queued_index_locked(const m3_shadow_plan_t *p,
                                  uint64_t request_id) {
    for (size_t i = 0; i < M3_SHD_MAX_QUEUED_REQUESTS; ++i)
        if (p->queued_used[i] && p->queued[i].request_id == request_id)
            return i;
    return M3_SHD_MAX_QUEUED_REQUESTS;
}

static int active_id_locked(const m3_shadow_plan_t *p, uint64_t request_id) {
    for (size_t i = 0; i < M3_SHD_MAX_ACTIVE_REQUESTS; ++i)
        if (p->active[i].used && p->active[i].id == request_id) return 1;
    return 0;
}

static int request_id_live_locked(const m3_shadow_plan_t *p,
                                  uint64_t request_id) {
    return active_id_locked(p, request_id)
        || queued_index_locked(p, request_id) != M3_SHD_MAX_QUEUED_REQUESTS;
}

static int request_facts_match(const m3_shd_request_t *a,
                               const m3_shd_request_t *b) {
    return a && b
        && a->request_id == b->request_id
        && a->model_id == b->model_id
        && a->key.layer == b->key.layer
        && a->key.expert == b->key.expert
        && a->bytes == b->bytes
        && a->consumer_need_ns == b->consumer_need_ns
        && a->demand == b->demand
        && a->forward_id == b->forward_id
        && a->generation == b->generation
        && a->consumer_node == b->consumer_node;
}

/* Automatically allocated IDs are monotone.  UINT64_MAX is reserved as the
 * exhausted-frontier sentinel, so no allocator increment can wrap. Explicit
 * IDs must still be absent from both live ledgers, but may be reused after
 * retirement because the planner keeps a bounded history. */
static int prepare_request_id_locked(m3_shadow_plan_t *p,
                                     uint64_t requested,
                                     uint64_t *assigned) {
    if (!p || !assigned) return 0;
    if (requested) {
        if (requested == UINT64_MAX || request_id_live_locked(p, requested))
            return 0;
        *assigned = requested;
        return 1;
    }
    if (p->next_request_id == 0 || p->next_request_id >= UINT64_MAX)
        return 0;
    *assigned = p->next_request_id++;
    return 1;
}

static void commit_explicit_request_id_locked(m3_shadow_plan_t *p,
                                              uint64_t request_id) {
    if (request_id < p->next_request_id) return;
    p->next_request_id = request_id == UINT64_MAX - 1
        ? UINT64_MAX : request_id + 1;
}

static uint64_t duration_ns(uint64_t bytes, uint64_t rate_bytes_per_s) {
    if (!rate_bytes_per_s) return UINT64_MAX;
    {
        unsigned __int128 numerator = (unsigned __int128)bytes * UINT64_C(1000000000);
        numerator += rate_bytes_per_s - 1u;
        numerator /= rate_bytes_per_s;
        return numerator > UINT64_MAX ? UINT64_MAX : (uint64_t)numerator;
    }
}

static int candidate_occupancy(const m3_shadow_plan_t *p,
                               const m3_shd_request_t *request,
                               const m3_shd_candidate_t *candidate,
                               uint64_t at_ns, int drop_completed,
                               m3_shd_count_t *counts) {
    uint32_t conversion_inflight = 0;
    uint64_t conversion_bytes = 0;
    memset(counts, 0, sizeof(*counts) * p->topology->n_resources);
    for (size_t a = 0; a < M3_SHD_MAX_ACTIVE_REQUESTS; ++a) {
        const m3_shd_active_t *active = &p->active[a];
        if (!active->used) continue;
        if (drop_completed && active->predicted_ready_ns <= at_ns) continue;
        conversion_inflight++;
        conversion_bytes = sat_add(conversion_bytes, active->request.bytes);
        for (size_t j = 0; j < active->path.n_resources; ++j) {
            int ri = resource_index(p, active->path.resources[j]);
            if (ri < 0) return 0;
            counts[ri].inflight++;
            counts[ri].bytes = sat_add(counts[ri].bytes, active->request.bytes);
        }
    }
    for (size_t j = 0; j < candidate->n_resources; ++j) {
        int ri = resource_index(p, candidate->resources[j]);
        if (ri < 0) return 0;
        counts[ri].inflight++;
        counts[ri].bytes = sat_add(counts[ri].bytes, request->bytes);
    }
    conversion_inflight++;
    conversion_bytes = sat_add(conversion_bytes, request->bytes);
    for (size_t i = 0; i < p->topology->n_resources; ++i) {
        const m3_shd_resource_t *r = &p->topology->resources[i];
        if (counts[i].inflight > r->max_inflight
            || counts[i].bytes > r->max_inflight_bytes) return -1;
    }
    if (p->conversion
        && (conversion_inflight > p->conversion->max_inflight
            || conversion_bytes > p->conversion->max_inflight_bytes))
        return -1;
    return 1;
}

static int simulate_flows(const m3_shadow_plan_t *p,
                          m3_shd_flow_t *flows, size_t n_flows,
                          const m3_shd_count_t *counts,
                          uint64_t dispatch_ns) {
    size_t complete = 0;
    uint64_t now = dispatch_ns;
    for (size_t i = 0; i < n_flows; ++i) {
        uint64_t startup = 0;
        m3_shd_size_class_t size_class =
            m3_shd_classify_size(flows[i].request->bytes);
        if (size_class == M3_SHD_SIZE_UNKNOWN) return 0;
        for (size_t j = 0; j < flows[i].path->n_resources; ++j) {
            int ri = resource_index(p, flows[i].path->resources[j]);
            const m3_shd_resource_profile_t *prof =
                profile_for(p, flows[i].path->resources[j]);
            m3_shd_load_class_t load = m3_shd_classify_load(counts[ri].inflight);
            if (!prof || load == M3_SHD_LOAD_UNKNOWN) return 0;
            if (prof->startup_ns[size_class][load] > startup)
                startup = prof->startup_ns[size_class][load];
            if (prof->residual_ns[size_class][load] > flows[i].uncertainty_ns)
                flows[i].uncertainty_ns = prof->residual_ns[size_class][load];
        }
        flows[i].remaining = flows[i].request->bytes;
        flows[i].start_ns = sat_add(dispatch_ns, startup);
        flows[i].state = flows[i].start_ns <= dispatch_ns ? 1 : 0;
    }

    while (complete < n_flows) {
        uint32_t transfer_count[M3_SHD_MAX_RESOURCES] = {0};
        uint64_t next_start = UINT64_MAX;
        for (size_t i = 0; i < n_flows; ++i) {
            if (flows[i].state == 0 && flows[i].start_ns < next_start)
                next_start = flows[i].start_ns;
            if (flows[i].state == 1) {
                for (size_t j = 0; j < flows[i].path->n_resources; ++j) {
                    int ri = resource_index(p, flows[i].path->resources[j]);
                    transfer_count[ri]++;
                }
            }
        }
        uint64_t completion_time[M3_SHD_MAX_ACTIVE_REQUESTS + 1];
        for (size_t i = 0; i < n_flows; ++i)
            completion_time[i] = UINT64_MAX;
        uint64_t next_completion = UINT64_MAX;
        for (size_t i = 0; i < n_flows; ++i) {
            if (flows[i].state != 1) continue;
            m3_shd_size_class_t size_class =
                m3_shd_classify_size(flows[i].request->bytes);
            uint64_t rate = UINT64_MAX;
            for (size_t j = 0; j < flows[i].path->n_resources; ++j) {
                int ri = resource_index(p, flows[i].path->resources[j]);
                const m3_shd_resource_profile_t *prof =
                    profile_for(p, flows[i].path->resources[j]);
                m3_shd_load_class_t load =
                    m3_shd_classify_load(counts[ri].inflight);
                uint64_t raw_rate = prof->rate_bytes_per_s[size_class][load];
                if (!raw_rate || !transfer_count[ri]) return 0;
                uint64_t shared_rate = raw_rate / transfer_count[ri];
                if (!shared_rate) return 0;
                if (shared_rate < rate) rate = shared_rate;
            }
            completion_time[i] = sat_add(now,
                                         duration_ns(flows[i].remaining, rate));
            if (completion_time[i] < next_completion)
                next_completion = completion_time[i];
        }
        uint64_t next = next_start < next_completion ? next_start : next_completion;
        if (next == UINT64_MAX || next < now) return 0;
        uint64_t elapsed = next - now;
        for (size_t i = 0; i < n_flows; ++i) {
            if (flows[i].state != 1) continue;
            m3_shd_size_class_t size_class =
                m3_shd_classify_size(flows[i].request->bytes);
            uint64_t rate = UINT64_MAX;
            for (size_t j = 0; j < flows[i].path->n_resources; ++j) {
                int ri = resource_index(p, flows[i].path->resources[j]);
                const m3_shd_resource_profile_t *prof =
                    profile_for(p, flows[i].path->resources[j]);
                m3_shd_load_class_t load =
                    m3_shd_classify_load(counts[ri].inflight);
                uint64_t shared_rate = prof->rate_bytes_per_s[size_class][load]
                                      / transfer_count[ri];
                if (shared_rate < rate) rate = shared_rate;
            }
            unsigned __int128 served = (unsigned __int128)rate * elapsed;
            served /= UINT64_C(1000000000);
            if (served >= flows[i].remaining) flows[i].remaining = 0;
            else flows[i].remaining -= (uint64_t)served;
        }
        now = next;
        for (size_t i = 0; i < n_flows; ++i) {
            if (flows[i].state == 1 && completion_time[i] <= now) {
                flows[i].state = 2;
                flows[i].ready_ns = now;
                complete++;
            } else if (flows[i].state == 0 && flows[i].start_ns <= now) {
                flows[i].state = 1;
            }
        }
    }
    return 1;
}

/* I/O completion is not weight readiness when a calibrated conversion stage
 * exists.  The virtual conversion resource is deliberately serial: flows
 * enter it in I/O-ready order, with request ID as the deterministic tie
 * break.  This stage only changes the private prediction copy. */
static int simulate_conversion(const m3_shadow_plan_t *p,
                               m3_shd_flow_t *flows, size_t n_flows) {
    if (!p->conversion) return 1;
    if (!n_flows || n_flows > M3_SHD_MAX_ACTIVE_REQUESTS + 1)
        return 0;
    m3_shd_load_class_t load = m3_shd_classify_load((uint32_t)n_flows);
    if (load == M3_SHD_LOAD_UNKNOWN) return 0;
    size_t order[M3_SHD_MAX_ACTIVE_REQUESTS + 1];
    for (size_t i = 0; i < n_flows; ++i) {
        size_t position = i;
        while (position > 0) {
            size_t previous = order[position - 1];
            if (flows[previous].ready_ns < flows[i].ready_ns
                || (flows[previous].ready_ns == flows[i].ready_ns
                    && flows[previous].request->request_id
                       <= flows[i].request->request_id))
                break;
            order[position] = previous;
            position--;
        }
        order[position] = i;
    }
    uint64_t conversion_end = 0;
    for (size_t position = 0; position < n_flows; ++position) {
        size_t i = order[position];
        m3_shd_size_class_t size_class =
            m3_shd_classify_size(flows[i].request->bytes);
        if (size_class == M3_SHD_SIZE_UNKNOWN) return 0;
        uint64_t duration = p->conversion->duration_ns[size_class][load];
        if (!duration) return 0;
        uint64_t start = flows[i].ready_ns > conversion_end
            ? flows[i].ready_ns : conversion_end;
        flows[i].ready_ns = sat_add(start, duration);
        conversion_end = flows[i].ready_ns;
        if (p->conversion->residual_ns[size_class][load]
            > flows[i].uncertainty_ns)
            flows[i].uncertainty_ns =
                p->conversion->residual_ns[size_class][load];
    }
    return 1;
}

static int evaluate_one(const m3_shadow_plan_t *p,
                        const m3_shd_request_t *request,
                        const m3_shd_candidate_t *candidate,
                        uint64_t dispatch_ns, int drop_completed,
                        m3_shd_decision_record_t *out) {
    m3_shd_count_t counts[M3_SHD_MAX_RESOURCES];
    uint64_t total_score = 0;
    uint64_t new_ready = 0;
    uint64_t new_uncertainty = 0;
    int occ = candidate_occupancy(p, request, candidate, dispatch_ns,
                                  drop_completed, counts);
    if (occ <= 0) return occ;

    m3_shd_flow_t flows[M3_SHD_MAX_ACTIVE_REQUESTS + 1];
    size_t n_flows = 0;
    for (size_t a = 0; a < M3_SHD_MAX_ACTIVE_REQUESTS; ++a) {
        if (!p->active[a].used) continue;
        if (drop_completed && p->active[a].predicted_ready_ns <= dispatch_ns)
            continue;
        flows[n_flows++] = (m3_shd_flow_t){
            &p->active[a].request, &p->active[a].path, 0, 0, 0, 0, 0
        };
    }
    flows[n_flows++] = (m3_shd_flow_t){request, candidate, 0, 0, 0, 0, 0};
    if (!simulate_flows(p, flows, n_flows, counts, dispatch_ns)
        || !simulate_conversion(p, flows, n_flows)) return 0;
    for (size_t i = 0; i < n_flows; ++i) {
        uint64_t guarded_ready = sat_add(flows[i].ready_ns,
                                         flows[i].uncertainty_ns);
        if (guarded_ready > flows[i].request->consumer_need_ns)
            total_score = sat_add(total_score,
                                  guarded_ready - flows[i].request->consumer_need_ns);
        if (i == n_flows - 1) {
            new_ready = flows[i].ready_ns;
            new_uncertainty = flows[i].uncertainty_ns;
        }
    }
    if (out) {
        memset(out, 0, sizeof(*out));
        out->request_id = request->request_id;
        out->dispatch_ns = dispatch_ns;
        out->ready_ns = new_ready;
        out->consumer_need_ns = request->consumer_need_ns;
        out->score_ns = total_score;
        out->uncertainty_ns = new_uncertainty;
        out->chosen_resource_id = candidate->resources[0];
        out->occupancy_before = (uint32_t)p->active_count;
        out->occupancy_after = (uint32_t)p->active_count + 1u;
        out->result = M3_SHD_OK;
        out->reason = M3_SHD_REASON_ADMIT;
    }
    return 1;
}

static int better(const m3_shd_decision_record_t *a,
                  const m3_shd_decision_record_t *b) {
    if (a->score_ns != b->score_ns) return a->score_ns < b->score_ns;
    if (a->ready_ns != b->ready_ns) return a->ready_ns < b->ready_ns;
    return a->chosen_resource_id < b->chosen_resource_id;
}

static m3_shd_result_t decide_internal(const m3_shadow_plan_t *p,
                                       const m3_shd_request_t *request,
                                       const m3_shd_candidate_t *candidates,
                                       size_t n_candidates,
                                       uint64_t decision_ns,
                                       m3_shd_decision_record_t *out) {
    if (!p || !request || !out || !request->demand || !request->bytes
        || n_candidates > M3_SHD_MAX_RESOURCES) {
        return M3_SHD_UNKNOWN;
    }
    m3_shd_decision_record_t best;
    int have_best = 0;
    for (size_t i = 0; i < n_candidates; ++i) {
        if (!path_valid(p, &candidates[i])) continue;
        m3_shd_decision_record_t trial;
        int rc = evaluate_one(p, request, &candidates[i],
                              decision_ns,
                              0, &trial);
        if (rc == -1) continue;
        if (rc == 0) continue;
        trial.candidate_index = (uint32_t)i;
        trial.request_id = request->request_id;
        trial.profile_hash = p->profile_hash;
        if (!have_best || better(&trial, &best)) {
            best = trial;
            have_best = 1;
        }
    }

    /* A bounded defer candidate is evaluated at the earliest predicted
     * completion of an active request.  Real completion is never recorded by
     * this path; the drop is only on the private hypothetical ledger. */
    if (p->active_count) {
        uint64_t defer_ns = UINT64_MAX;
        int overdue = 0;
        for (size_t i = 0; i < M3_SHD_MAX_ACTIVE_REQUESTS; ++i) {
            if (!p->active[i].used) continue;
            if (p->active[i].predicted_ready_ns < decision_ns) {
                /* An observed read that has outlived its model prediction is
                 * still active.  Never evaluate a defer dispatch in the
                 * past; keep its occupancy and return UNKNOWN if no current
                 * candidate is feasible. */
                overdue = 1;
                break;
            }
            if (p->active[i].predicted_ready_ns < defer_ns)
                defer_ns = p->active[i].predicted_ready_ns;
        }
        if (!overdue && defer_ns != UINT64_MAX && defer_ns >= decision_ns) {
            for (size_t i = 0; i < n_candidates; ++i) {
                if (!path_valid(p, &candidates[i])) continue;
                m3_shd_decision_record_t trial;
                int rc = evaluate_one(p, request, &candidates[i], defer_ns,
                                      1, &trial);
                if (rc != 1) continue;
                trial.candidate_index = UINT32_MAX;
                trial.request_id = request->request_id;
                trial.profile_hash = p->profile_hash;
                trial.result = M3_SHD_DEFER;
                trial.reason = M3_SHD_REASON_DEFER;
                if (!have_best || better(&trial, &best)) {
                    best = trial;
                    have_best = 1;
                }
            }
        }
    }
    if (!have_best) {
        memset(out, 0, sizeof(*out));
        out->request_id = request->request_id;
        out->profile_hash = p->profile_hash;
        out->result = M3_SHD_RESULT_UNKNOWN_BOUND;
        out->reason = M3_SHD_REASON_BOUND;
        return M3_SHD_RESULT_UNKNOWN_BOUND;
    }
    *out = best;
    return best.result;
}

static int append_record(m3_shadow_plan_t *p,
                         const m3_shd_decision_record_t *r) {
    if (p->log_bytes >= sizeof(p->log)) return 0;
    int n = snprintf(p->log + p->log_bytes,
                     sizeof(p->log) - p->log_bytes,
                     "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u,%u,%u,%u,%u\n",
                     (unsigned long long)r->request_id,
                     (unsigned long long)r->snapshot_hash,
                     (unsigned long long)r->profile_hash,
                     (unsigned long long)r->dispatch_ns,
                     (unsigned long long)r->ready_ns,
                     (unsigned long long)r->consumer_need_ns,
                     (unsigned long long)r->score_ns,
                     (unsigned long long)r->uncertainty_ns,
                     (unsigned long long)r->chosen_resource_id,
                     r->candidate_index, r->occupancy_before,
                     r->occupancy_after, (unsigned)r->result,
                     (unsigned)r->reason);
    if (n < 0 || (size_t)n >= sizeof(p->log) - p->log_bytes) return 0;
    p->log_bytes += (size_t)n;
    return 1;
}

m3_shadow_plan_t *m3_shd_open(const m3_shd_topology_profile_t *topology,
                              const m3_shd_calibration_set_t *calibration,
                              m3_shd_clock_t clock,
                              char *err, size_t errsz) {
    if (!topology || !topology->resources
        || topology->n_resources == 0
        || topology->n_resources > M3_SHD_MAX_RESOURCES) {
        set_error(err, errsz, "invalid bounded topology");
        return NULL;
    }
    if (!calibration || calibration->topology != topology
        || calibration->topology_hash != topology->topology_hash
        || !m3_shd_calib_is_sound(calibration)) {
        set_error(err, errsz, "unsound or mismatched calibration");
        return NULL;
    }
    if (topology->n_copies > M3_SHD_MAX_COPIES
        || (topology->n_copies && !topology->copies)) {
        set_error(err, errsz, "invalid bounded weight-copy table");
        return NULL;
    }
    for (size_t i = 0; i < topology->n_copies; ++i) {
        const m3_shd_weight_copy_t *copy = &topology->copies[i];
        int copy_resource = resource_index(
            (const m3_shadow_plan_t *)&(m3_shadow_plan_t){
                .topology = topology }, copy->drive_id);
        if (!copy->copy_id || !copy->model_id || !copy->bytes
            || copy->available > 1 || copy_resource < 0
            || topology->resources[copy_resource].kind != M3_SHD_KIND_DRIVE) {
            set_error(err, errsz, "weight copy has invalid identity or drive");
            return NULL;
        }
        for (size_t j = 0; j < i; ++j)
            if (topology->copies[j].copy_id == copy->copy_id) {
                set_error(err, errsz, "duplicate weight-copy id");
                return NULL;
            }
        for (size_t j = 0; j < i; ++j)
            if (topology->copies[j].model_id == copy->model_id
                && topology->copies[j].tensor_id == copy->tensor_id
                && topology->copies[j].bytes != copy->bytes) {
                set_error(err, errsz, "contradictory weight-copy bytes");
                return NULL;
            }
    }
    for (size_t i = 0; i < topology->n_resources; ++i) {
        const m3_shd_resource_t *r = &topology->resources[i];
        if (!r->resource_id || r->kind >= M3_SHD_KIND_LAST) {
            set_error(err, errsz, "resource has invalid kind or id");
            return NULL;
        }
        /* Storage-kind resources (drive, controller, upstream) carry
         * bounded admission and must have a positive hard bound.  The
         * new kinds (cpu_island, gpu_island, memory_dom, link) have no
         * admission semantics and intentionally leave the bound fields
         * zero; skip the check for them. */
        if ((r->kind == M3_SHD_KIND_DRIVE || r->kind == M3_SHD_KIND_CONTROLLER
             || r->kind == M3_SHD_KIND_UPSTREAM)
            && (!r->max_inflight || !r->max_inflight_bytes)) {
            set_error(err, errsz, "storage resource has no positive hard bound");
            return NULL;
        }
        if (r->parent_id && resource_index((const m3_shadow_plan_t *)&(m3_shadow_plan_t){ .topology = topology }, r->parent_id) < 0) {
            set_error(err, errsz, "resource parent is not declared");
            return NULL;
        }
        for (size_t j = 0; j < i; ++j) {
            if (topology->resources[j].resource_id == r->resource_id) {
                set_error(err, errsz, "duplicate resource id");
                return NULL;
            }
        }
        const m3_shd_resource_profile_t *prof = profile_for(
            (const m3_shadow_plan_t *)&(m3_shadow_plan_t){
                .topology = topology, .calibration = calibration },
            r->resource_id);
        /* A missing profile is an honest partial-calibration state.  A
         * present profile was already checked by m3_shd_calib_is_sound(). */
        if (prof && prof->topology_hash != topology->topology_hash) {
            set_error(err, errsz, "mismatched resource calibration");
            return NULL;
        }
        m3_shd_resource_id_t parent = r->parent_id;
        for (size_t hops = 0; parent && hops < topology->n_resources; ++hops) {
            if (parent == r->resource_id) {
                set_error(err, errsz, "resource topology contains a cycle");
                return NULL;
            }
            int pi = resource_index(
                (const m3_shadow_plan_t *)&(m3_shadow_plan_t){
                    .topology = topology }, parent);
            if (pi < 0) {
                set_error(err, errsz, "resource parent is not declared");
                return NULL;
            }
            parent = topology->resources[pi].parent_id;
            if (parent && hops + 1 == topology->n_resources) {
                set_error(err, errsz, "resource topology exceeds bounded depth");
                return NULL;
            }
        }
        /* Slice 7a-ext: cpu_island / gpu_island / memory_dom / link rows
         * also store their kind-specific parent in `parent_mem_id` and
         * `pcie_parent_id`.  These are aliased into `parent_id` for the
         * purpose of cycle/depth walking (above), so the single
         * parent_id walk above is sufficient for all kinds. */
    }
    m3_shadow_plan_t *p = (m3_shadow_plan_t *)calloc(1, sizeof(*p));
    if (!p) {
        set_error(err, errsz, "planner allocation failed");
        return NULL;
    }
    p->topology = topology;
    p->calibration = calibration;
    p->conversion = calibration->conversion.enabled
        ? &calibration->conversion : NULL;
    p->clock = clock;
    p->next_request_id = 1;
    atomic_flag_clear(&p->guard);
    p->profile_hash = UINT64_C(1469598103934665603);
    p->profile_hash = fnv1a_u64(p->profile_hash, topology->topology_hash);
    p->profile_hash = fnv1a_u64(p->profile_hash, calibration->version);
    for (size_t i = 0; i < topology->n_resources; ++i) {
        const m3_shd_resource_t *resource = &topology->resources[i];
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->resource_id);
        p->profile_hash = fnv1a_u64(p->profile_hash, (uint64_t)resource->kind);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->parent_id);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->max_inflight);
        p->profile_hash = fnv1a_u64(p->profile_hash,
                                    resource->max_inflight_bytes);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->isa_caps);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->cores);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->parent_mem_id);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->vram_bytes);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->p2p_class);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->pcie_parent_id);
        p->profile_hash = fnv1a_u64(p->profile_hash,
                                    resource->rate_bytes_per_s);
        p->profile_hash = fnv1a_u64(p->profile_hash,
                                    resource->bandwidth_class);
        p->profile_hash = fnv1a_u64(p->profile_hash, resource->eligible);
    }
    for (size_t i = 0; i < topology->n_copies; ++i) {
        const m3_shd_weight_copy_t *copy = &topology->copies[i];
        p->profile_hash = fnv1a_u64(p->profile_hash, copy->copy_id);
        p->profile_hash = fnv1a_u64(p->profile_hash, copy->model_id);
        p->profile_hash = fnv1a_u64(p->profile_hash, copy->tensor_id);
        p->profile_hash = fnv1a_u64(p->profile_hash, copy->bytes);
        p->profile_hash = fnv1a_u64(p->profile_hash, copy->drive_id);
        p->profile_hash = fnv1a_u64(p->profile_hash, copy->available);
    }
    for (size_t i = 0; i < calibration->n_profiles; ++i) {
        const m3_shd_resource_profile_t *profile = &calibration->profiles[i];
        p->profile_hash = fnv1a_u64(p->profile_hash, profile->resource_id);
        p->profile_hash = fnv1a_u64(p->profile_hash, profile->topology_hash);
        p->profile_hash = fnv1a_u64(p->profile_hash,
                                    profile->units_per_second_explicit);
        p->profile_hash = fnv1a_u64(p->profile_hash, profile->version);
        p->profile_hash = fnv1a_u64(p->profile_hash, profile->sample_count);
        for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s)
            for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l)
                p->profile_hash = fnv1a_u64(
                    p->profile_hash, profile->startup_ns[s][l]);
        for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s)
            for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l)
                p->profile_hash = fnv1a_u64(
                    p->profile_hash, profile->rate_bytes_per_s[s][l]);
        for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s) {
            for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l) {
                p->profile_hash = fnv1a_u64(
                    p->profile_hash, profile->residual_ns[s][l]);
                p->profile_hash = fnv1a_u64(
                    p->profile_hash, profile->residual_percentile_ppm[s][l]);
            }
        }
    }
    p->profile_hash = fnv1a_u64(p->profile_hash,
                                calibration->conversion.enabled);
    p->profile_hash = fnv1a_u64(p->profile_hash,
                                calibration->max_queue_age_ns);
    if (calibration->conversion.enabled) {
        const m3_shd_conversion_profile_t *conversion =
            &calibration->conversion;
        p->profile_hash = fnv1a_u64(p->profile_hash, conversion->version);
        p->profile_hash = fnv1a_u64(p->profile_hash,
                                    conversion->sample_count);
        p->profile_hash = fnv1a_u64(p->profile_hash,
                                    conversion->topology_hash);
        p->profile_hash = fnv1a_u64(p->profile_hash,
                                    conversion->max_inflight);
        p->profile_hash = fnv1a_u64(p->profile_hash,
                                    conversion->max_inflight_bytes);
        for (size_t s = 0; s < M3_SHD_SIZE_CLASS_COUNT; ++s)
            for (size_t l = 0; l < M3_SHD_LOAD_CLASS_COUNT; ++l) {
                p->profile_hash = fnv1a_u64(
                    p->profile_hash, conversion->duration_ns[s][l]);
                p->profile_hash = fnv1a_u64(
                    p->profile_hash, conversion->residual_ns[s][l]);
                p->profile_hash = fnv1a_u64(
                    p->profile_hash,
                    conversion->residual_percentile_ppm[s][l]);
            }
    }
    return p;
}

void m3_shd_close(m3_shadow_plan_t *p) { free(p); }

m3_shd_result_t m3_shd_predict(const m3_shadow_plan_t *p,
                               const m3_shd_request_t *request,
                               const m3_shd_candidate_t *candidates,
                               size_t n_candidates,
                               m3_shd_decision_record_t *out) {
    if (!p) return M3_SHD_UNKNOWN;
    uint64_t decision_ns = planner_now(p);
    planner_lock((m3_shadow_plan_t *)p);
    m3_shd_result_t result = decide_internal(p, request, candidates,
                                             n_candidates, decision_ns, out);
    planner_unlock((m3_shadow_plan_t *)p);
    return result;
}

m3_shd_result_t m3_shd_decide(m3_shadow_plan_t *p,
                              const m3_shd_request_t *request,
                              const m3_shd_candidate_t *candidates,
                              size_t n_candidates,
                              m3_shd_decision_record_t *out) {
    if (!p) return M3_SHD_UNKNOWN;
    uint64_t decision_ns = planner_now(p);
    planner_lock(p);
    m3_shd_result_t result = decide_internal(p, request, candidates,
                                             n_candidates, decision_ns, out);
    if (out) {
        out->snapshot_hash = p->profile_hash ^ (uint64_t)p->active_count;
        append_record(p, out);
    }
    planner_unlock(p);
    return result;
}

static int record_real_start_locked(m3_shadow_plan_t *p,
                                    const m3_shd_request_t *request,
                                    const m3_shd_candidate_t *actual_path,
                                    uint64_t dispatch_ns,
                                    uint64_t *request_id_out) {
    if (p->active_count >= M3_SHD_MAX_ACTIVE_REQUESTS
        || (request->request_id && request_id_live_locked(p, request->request_id)))
        return 0;
    uint64_t assigned = 0;
    if (!prepare_request_id_locked(p, request->request_id, &assigned))
        return 0;
    m3_shd_request_t copy = *request;
    copy.request_id = assigned;
    copy.state = M3_SHD_STATE_ISSUED;
    copy.n_reserved_resources = actual_path->n_resources;
    memcpy(copy.reserved_resources, actual_path->resources,
           actual_path->n_resources * sizeof(actual_path->resources[0]));
    m3_shd_decision_record_t rec;
    int predicted = evaluate_one(p, &copy, actual_path, dispatch_ns, 0, &rec);
    for (size_t i = 0; i < M3_SHD_MAX_ACTIVE_REQUESTS; ++i) {
        if (p->active[i].used) continue;
        p->active[i].used = 1;
        p->active[i].id = assigned;
        p->active[i].request = copy;
        p->active[i].path = *actual_path;
        p->active[i].predicted_ready_ns = predicted == 1
            ? rec.ready_ns : UINT64_MAX;
        p->active_count++;
        if (request->request_id) commit_explicit_request_id_locked(p, assigned);
        if (request_id_out) *request_id_out = assigned;
        return 1;
    }
    return 0;
}

m3_shd_result_t m3_shd_decide_and_record_real_start(
    m3_shadow_plan_t *p, const m3_shd_request_t *request,
    const m3_shd_candidate_t *candidates, size_t n_candidates,
    const m3_shd_candidate_t *actual_path,
    m3_shd_decision_record_t *out, uint64_t *request_id_out,
    m3_shd_resource_occupancy_t *occupancy_out, size_t occupancy_cap) {
    if (!p || !request || !candidates || !n_candidates || !out
        || !request_id_out || n_candidates > M3_SHD_MAX_RESOURCES
        || !actual_path || !request->bytes || !request->demand
        || (occupancy_out && occupancy_cap < actual_path->n_resources)
        || !path_valid(p, actual_path))
        return M3_SHD_UNKNOWN;
    uint64_t dispatch_ns = planner_now(p);
    planner_lock(p);
    m3_shd_request_t assigned_request;
    uint64_t assigned_id = 0;
    if (!prepare_request_id_locked(p, request->request_id, &assigned_id)) {
        planner_unlock(p);
        return M3_SHD_UNKNOWN;
    }
    assigned_request = *request;
    assigned_request.request_id = assigned_id;
    m3_shd_result_t result = decide_internal(p, &assigned_request, candidates,
                                             n_candidates, dispatch_ns, out);
    if (out) {
        out->snapshot_hash = p->profile_hash ^ (uint64_t)p->active_count;
        append_record(p, out);
    }
    if (!record_real_start_locked(p, &assigned_request, actual_path,
                                  dispatch_ns, request_id_out)) {
        planner_unlock(p);
        return M3_SHD_UNKNOWN;
    }
    if (occupancy_out)
        (void)real_path_occupancy_locked(p, actual_path, occupancy_out,
                                         occupancy_cap);
    planner_unlock(p);
    return result;
}

int m3_shd_submit(m3_shadow_plan_t *p, const m3_shd_request_t *request,
                  const m3_shd_candidate_t *chosen, uint64_t *request_id_out) {
    if (!p || !request || !chosen || !request->bytes || !request->demand
        || !path_valid(p, chosen)) return 0;
    uint64_t dispatch_ns = planner_now(p);
    planner_lock(p);
    if (p->active_count >= M3_SHD_MAX_ACTIVE_REQUESTS) {
        planner_unlock(p);
        return 0;
    }

    m3_shd_request_t copy;
    if (request->request_id && active_id_locked(p, request->request_id)) {
        planner_unlock(p);
        return 0;
    }
    size_t queued_index = request->request_id
        ? queued_index_locked(p, request->request_id)
        : M3_SHD_MAX_QUEUED_REQUESTS;
    if (queued_index != M3_SHD_MAX_QUEUED_REQUESTS) {
        /* A queued request may be promoted, but only with the same immutable
         * identity/facts.  The queued copy preserves its enqueue timestamp. */
        if (!request_facts_match(&p->queued[queued_index], request)) {
            planner_unlock(p);
            return 0;
        }
        copy = p->queued[queued_index];
    } else {
        uint64_t assigned_id = 0;
        if (!prepare_request_id_locked(p, request->request_id, &assigned_id)) {
            planner_unlock(p);
            return 0;
        }
        copy = *request;
        copy.request_id = assigned_id;
    }
    copy.state = M3_SHD_STATE_ISSUED;
    copy.n_reserved_resources = chosen->n_resources;
    memcpy(copy.reserved_resources, chosen->resources,
           chosen->n_resources * sizeof(chosen->resources[0]));
    m3_shd_decision_record_t rec;
    if (evaluate_one(p, &copy, chosen, dispatch_ns, 0, &rec) != 1) {
        planner_unlock(p);
        return 0;
    }
    for (size_t i = 0; i < M3_SHD_MAX_ACTIVE_REQUESTS; ++i) {
        if (p->active[i].used) continue;
        p->active[i].used = 1;
        p->active[i].id = copy.request_id;
        p->active[i].request = copy;
        p->active[i].path = *chosen;
        p->active[i].predicted_ready_ns = rec.ready_ns;
        p->active_count++;
        if (request->request_id && queued_index == M3_SHD_MAX_QUEUED_REQUESTS)
            commit_explicit_request_id_locked(p, copy.request_id);
        remove_queued_locked(p, copy.request_id);
        if (request_id_out) *request_id_out = copy.request_id;
        planner_unlock(p);
        return 1;
    }
    planner_unlock(p);
    return 0;
}

int m3_shd_record_real_start(m3_shadow_plan_t *p,
                             const m3_shd_request_t *request,
                             const m3_shd_candidate_t *actual_path,
                             uint64_t *request_id_out) {
    if (!p || !request || !actual_path || !request->bytes
        || !request->demand || !path_valid(p, actual_path)) return 0;
    uint64_t dispatch_ns = planner_now(p);
    planner_lock(p);
    int result = record_real_start_locked(p, request, actual_path,
                                          dispatch_ns, request_id_out);
    planner_unlock(p);
    return result;
}

int m3_shd_enqueue(m3_shadow_plan_t *p, const m3_shd_request_t *request,
                   uint64_t *request_id_out) {
    if (!p || !request || !request->demand || !request->bytes) return 0;
    uint64_t enqueue_ns = request->enqueue_ns
        ? request->enqueue_ns : planner_now(p);
    planner_lock(p);
    if (p->queued_count >= M3_SHD_MAX_QUEUED_REQUESTS
        || request->bytes > M3_SHD_MAX_QUEUED_BYTES
        || p->queued_bytes > M3_SHD_MAX_QUEUED_BYTES - request->bytes) {
        planner_unlock(p);
        return 0;
    }
    uint64_t assigned_id = 0;
    if (!prepare_request_id_locked(p, request->request_id, &assigned_id)) {
        planner_unlock(p);
        return 0;
    }
    m3_shd_request_t copy = *request;
    copy.request_id = assigned_id;
    if (request->request_id)
        commit_explicit_request_id_locked(p, copy.request_id);
    if (!copy.enqueue_ns)
        copy.enqueue_ns = enqueue_ns;
    copy.state = M3_SHD_STATE_QUEUED;
    for (size_t i = 0; i < M3_SHD_MAX_QUEUED_REQUESTS; ++i) {
        if (p->queued_used[i]) continue;
        p->queued[i] = copy;
        p->queued_used[i] = 1;
        p->queued_count++;
        p->queued_bytes += copy.bytes;
        if (request_id_out) *request_id_out = copy.request_id;
        planner_unlock(p);
        return 1;
    }
    planner_unlock(p);
    return 0;
}

static int request_is_age_priority(const m3_shd_request_t *request,
                                   uint64_t now_ns, uint64_t max_age_ns) {
    return max_age_ns && now_ns >= request->enqueue_ns
        && now_ns - request->enqueue_ns >= max_age_ns;
}

static int request_queue_cmp(const m3_shd_request_t *a,
                             const m3_shd_request_t *b,
                             uint64_t now_ns, uint64_t max_age_ns) {
    int a_old = request_is_age_priority(a, now_ns, max_age_ns);
    int b_old = request_is_age_priority(b, now_ns, max_age_ns);
    if (a_old != b_old) return a_old ? -1 : 1;
    if (a_old) {
        if (a->enqueue_ns != b->enqueue_ns)
            return a->enqueue_ns < b->enqueue_ns ? -1 : 1;
        if (a->request_id != b->request_id)
            return a->request_id < b->request_id ? -1 : 1;
        return 0;
    }
    return m3_shd_request_priority_cmp(a, b);
}

int m3_shd_peek_next(const m3_shadow_plan_t *p, m3_shd_request_t *out) {
    if (!p || !out) return 0;
    uint64_t now_ns = planner_now(p);
    planner_lock((m3_shadow_plan_t *)p);
    size_t best = M3_SHD_MAX_QUEUED_REQUESTS;
    for (size_t i = 0; i < M3_SHD_MAX_QUEUED_REQUESTS; ++i) {
        if (!p->queued_used[i]) continue;
        if (best == M3_SHD_MAX_QUEUED_REQUESTS
            || request_queue_cmp(&p->queued[i], &p->queued[best], now_ns,
                                 p->calibration->max_queue_age_ns) < 0)
            best = i;
    }
    if (best == M3_SHD_MAX_QUEUED_REQUESTS) {
        planner_unlock((m3_shadow_plan_t *)p);
        return 0;
    }
    *out = p->queued[best];
    planner_unlock((m3_shadow_plan_t *)p);
    return 1;
}

int m3_shd_remove_queued(m3_shadow_plan_t *p, uint64_t request_id) {
    if (!p) return 0;
    planner_lock(p);
    int result = remove_queued_locked(p, request_id);
    planner_unlock(p);
    return result;
}

size_t m3_shd_queued_count(const m3_shadow_plan_t *p) {
    if (!p) return 0;
    planner_lock((m3_shadow_plan_t *)p);
    size_t result = p->queued_count;
    planner_unlock((m3_shadow_plan_t *)p);
    return result;
}

static int remove_active(m3_shadow_plan_t *p, uint64_t id) {
    for (size_t i = 0; i < M3_SHD_MAX_ACTIVE_REQUESTS; ++i) {
        if (p->active[i].used && p->active[i].id == id) {
            memset(&p->active[i], 0, sizeof(p->active[i]));
            p->active_count--;
            return 1;
        }
    }
    return 0;
}

int m3_shd_record_real_completion(m3_shadow_plan_t *p, uint64_t request_id) {
    if (!p) return 0;
    planner_lock(p);
    int result = remove_active(p, request_id);
    planner_unlock(p);
    return result;
}

int m3_shd_record_real_failure(m3_shadow_plan_t *p, uint64_t request_id) {
    if (!p) return 0;
    planner_lock(p);
    int result = remove_active(p, request_id);
    planner_unlock(p);
    return result;
}

size_t m3_shd_active_count(const m3_shadow_plan_t *p) {
    if (!p) return 0;
    planner_lock((m3_shadow_plan_t *)p);
    size_t result = p->active_count;
    planner_unlock((m3_shadow_plan_t *)p);
    return result;
}

uint64_t m3_shd_real_occupancy_bytes(const m3_shadow_plan_t *p) {
    uint64_t total = 0;
    if (!p) return 0;
    planner_lock((m3_shadow_plan_t *)p);
    for (size_t i = 0; i < M3_SHD_MAX_ACTIVE_REQUESTS; ++i)
        if (p->active[i].used) total = sat_add(total, p->active[i].request.bytes);
    planner_unlock((m3_shadow_plan_t *)p);
    return total;
}

static int real_resource_occupancy_locked(
    const m3_shadow_plan_t *p, m3_shd_resource_id_t resource_id,
    m3_shd_resource_occupancy_t *out) {
    if (out) *out = (m3_shd_resource_occupancy_t){resource_id, 0, 0};
    if (!p || !out || !resource_id) return 0;
    int declared = resource_index(p, resource_id) >= 0;
    if (!declared) return 0;
    for (size_t i = 0; i < M3_SHD_MAX_ACTIVE_REQUESTS; ++i) {
        const m3_shd_active_t *active = &p->active[i];
        if (!active->used) continue;
        for (size_t j = 0; j < active->path.n_resources; ++j) {
            if (active->path.resources[j] != resource_id) continue;
            if (out->inflight < UINT32_MAX) out->inflight++;
            out->inflight_bytes = sat_add(out->inflight_bytes,
                                          active->request.bytes);
            break;
        }
    }
    return 1;
}

static int real_path_occupancy_locked(
    const m3_shadow_plan_t *p, const m3_shd_candidate_t *path,
    m3_shd_resource_occupancy_t *out, size_t cap) {
    if (!p || !path || !out || path->n_resources > cap
        || path->n_resources > M3_SHD_MAX_PATH_RESOURCES)
        return 0;
    for (size_t i = 0; i < path->n_resources; ++i) {
        if (!real_resource_occupancy_locked(p, path->resources[i], &out[i]))
            return 0;
    }
    return 1;
}

int m3_shd_real_resource_occupancy(
    const m3_shadow_plan_t *p, m3_shd_resource_id_t resource_id,
    m3_shd_resource_occupancy_t *out) {
    if (out) *out = (m3_shd_resource_occupancy_t){resource_id, 0, 0};
    if (!p || !out || !resource_id) return 0;
    planner_lock((m3_shadow_plan_t *)p);
    int declared = real_resource_occupancy_locked(p, resource_id, out);
    planner_unlock((m3_shadow_plan_t *)p);
    return declared;
}

const char *m3_shd_decision_log(const m3_shadow_plan_t *p, size_t *bytes_out) {
    if (!p) {
        if (bytes_out) *bytes_out = 0;
        return NULL;
    }
    planner_lock((m3_shadow_plan_t *)p);
    if (bytes_out) *bytes_out = p->log_bytes;
    const char *log = p->log;
    planner_unlock((m3_shadow_plan_t *)p);
    return log;
}

int m3_shd_decision_log_copy(const m3_shadow_plan_t *p, char *dst,
                             size_t cap, size_t *bytes_out) {
    if (!p) {
        if (bytes_out) *bytes_out = 0;
        return 0;
    }
    planner_lock((m3_shadow_plan_t *)p);
    size_t bytes = p->log_bytes;
    if (bytes_out) *bytes_out = bytes;
    int ok = dst && cap >= bytes + 1;
    if (ok) {
        memcpy(dst, p->log, bytes);
        dst[bytes] = '\0';
    }
    planner_unlock((m3_shadow_plan_t *)p);
    return ok;
}

uint64_t m3_shd_profile_hash(const m3_shadow_plan_t *p) {
    return p ? p->profile_hash : 0;
}

void m3_shd_observer_stats(const m3_shadow_plan_t *p,
                           m3_shd_observer_stats_t *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!p) return;
    planner_lock((m3_shadow_plan_t *)p);
    *out = p->observer_stats;
    planner_unlock((m3_shadow_plan_t *)p);
}

void m3_shd_observe_lookup_begin(m3_shadow_plan_t *p,
                                 const ColiExpertKey *key,
                                 uint64_t consumer_need_ns) {
    (void)key; (void)consumer_need_ns;
    if (!p) return;
    planner_lock(p);
    p->observer_stats.lookup_begins++;
    planner_unlock(p);
}
void m3_shd_observe_lookup_end(m3_shadow_plan_t *p,
                               const ColiExpertKey *key, int result,
                               uint64_t bytes) {
    (void)key;
    if (!p) return;
    planner_lock(p);
    if (result) p->observer_stats.lookup_failures++;
    if (UINT64_MAX - p->observer_stats.lookup_bytes < bytes)
        p->observer_stats.lookup_bytes = UINT64_MAX;
    else
        p->observer_stats.lookup_bytes += bytes;
    planner_unlock(p);
}
void m3_shd_observe_release(m3_shadow_plan_t *p, const ColiExpertKey *key) {
    (void)key;
    if (!p) return;
    planner_lock(p);
    p->observer_stats.releases++;
    planner_unlock(p);
}
