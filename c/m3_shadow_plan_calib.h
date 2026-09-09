/* m3_shadow_plan_calib.h -- calibration-profile representation for the
 * MiniMax-M3 Step 7 shadow planner.
 *
 * This is commit 1 of the multi-commit Step 7 implementation per
 * docs/results/m3-execution-step7-plan-2026-09-06.md §9. It declares the
 * profile and calibration data shapes plus a small set of pure inline
 * helpers; the parser, predictor, observer, fixtures, log header, public
 * API, and Makefile wiring all land in later commits. Do NOT add the
 * per-decision log record (m3_shadow_plan_log.h, commit 2) or the public
 * planner API (m3_shadow_plan.h, commit 3) here -- those modules depend
 * on this file, not the other way round, so no forward declarations of
 * their handle types are introduced.
 *
 * Style modelled on c/expert_store.h: COLIBRI_* header guard, opaque
 * types hidden behind typedefs where appropriate, static inline helpers,
 * no malloc / free / stdio / stdlib, no global mutable state. This file
 * is header-only and depends only on <stddef.h> and <stdint.h>.
 *
 * Cross-refs to the binding contract
 * (docs/results/m3-execution-step7-invariants-2026-09-06.md):
 *   INV-7  -- M3_SHD_CALIB_VERSION pins the planner version used by the
 *             replay-determinism invariant; same (snapshot hash, profile
 *             hash, planner version) => byte-equal decision log.
 *   INV-10 -- units_per_second_explicit == 0 on any resource collapses
 *             the entire calibration to UNKNOWN through
 *             m3_shd_calib_is_sound; an invalid calibration is rejected,
 *             while a declared resource with no profile remains UNKNOWN,
 *             never a default-capacity fill.
 */
#ifndef COLIBRI_M3_SHADOW_PLAN_CALIB_H
#define COLIBRI_M3_SHADOW_PLAN_CALIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Planner version pin (INV-7 determinism). Bumping this constant is the
 * explicit signal that calibration numerics, fixture arithmetic, or the
 * decision-log byte layout have changed; replay equality across versions
 * is not asserted and is not a contract violation. */
#define M3_SHD_CALIB_VERSION 2

/* A residual is useful to the predictor only when its statistical meaning is
 * explicit.  The supported tail choices are intentionally narrow: profiles
 * store p95 or p99 in parts per million, keeping the format integer-only.
 * Zero means that the bucket has no residual/tail allowance. */
#define M3_SHD_TAIL_PERCENTILE_NONE_PPM 0U
#define M3_SHD_TAIL_PERCENTILE_P95_PPM  950000U
#define M3_SHD_TAIL_PERCENTILE_P99_PPM  990000U

/* Resource-kind taxonomy. Every node on a read or compute path -- drive,
 * controller, upstream link, CPU island, GPU island, memory domain, link --
 * is a resource; the service-profile shape is the same across kinds, only
 * the topology carries the kind and kind-specific payload. New kinds are
 * added BEFORE the LAST sentinel so bounded loops over the enum stay valid. */
typedef enum {
    M3_SHD_KIND_DRIVE         = 0,
    M3_SHD_KIND_CONTROLLER    = 1,
    M3_SHD_KIND_UPSTREAM      = 2,
    M3_SHD_KIND_CPU_ISLAND    = 3,
    M3_SHD_KIND_GPU_ISLAND    = 4,
    M3_SHD_KIND_MEMORY_DOMAIN = 5,
    M3_SHD_KIND_LINK          = 6,
    M3_SHD_KIND_LAST          /* sentinel for bounded iteration; not a real kind */
} m3_shd_resource_kind_t;

/* ISA capability bitfield for CPU_ISLAND rows. */
typedef enum {
    M3_SHD_ISA_NONE      = 0u,
    M3_SHD_ISA_AVX       = 1u << 0,
    M3_SHD_ISA_AVX2      = 1u << 1,
    M3_SHD_ISA_FMA       = 1u << 2,
    M3_SHD_ISA_AVX512    = 1u << 3,
    M3_SHD_ISA_VNNI      = 1u << 4,
    M3_SHD_ISA_AMX       = 1u << 5,
    M3_SHD_ISA_SVE       = 1u << 6,
    M3_SHD_ISA_SVE2      = 1u << 7,
    M3_SHD_ISA_I8MM      = 1u << 8
} m3_shd_isa_caps_t;

/* Memory-domain bandwidth classes. Encoded as uint32_t so the planner can
 * accept additions without breaking ABI; 0 means UNKNOWN (no measurement
 * yet -- the planner must not invent a default, per INV-10). */
typedef enum {
    M3_SHD_MEM_CLASS_UNKNOWN = 0u,
    M3_SHD_MEM_CLASS_DDR3    = 1u,
    M3_SHD_MEM_CLASS_DDR4    = 2u,
    M3_SHD_MEM_CLASS_DDR5    = 3u,
    M3_SHD_MEM_CLASS_LPDDR4X = 4u,
    M3_SHD_MEM_CLASS_LPDDR5X = 5u,
    M3_SHD_MEM_CLASS_HBM2E   = 6u,
    M3_SHD_MEM_CLASS_GDDR5X  = 7u,
    M3_SHD_MEM_CLASS_GDDR6X  = 8u
} m3_shd_mem_class_t;

/* Stable per-resource identifier. Carried in snapshot and log records so
 * successive calibration revisions can be diffed without re-parsing the
 * topology. Typed as a plain uint64_t rather than an opaque struct: this
 * commit maps no fields beyond the id value, and a future opaque refactor
 * is a one-line change at every callsite. */
typedef uint64_t m3_shd_resource_id_t;

/* Epistemic state is part of the evidence contract, not a numeric capacity
 * value.  UNKNOWN and UNAVAILABLE currently have the same conservative
 * admission effect, but UNKNOWN may become measurable later while
 * UNAVAILABLE records that this measurement path cannot supply the fact.
 * THEORETICAL is specification-derived only; NOT_APPLICABLE removes an edge
 * from consideration. */
typedef enum {
    /* Zero is deliberately conservative: memset/zero-initialization must
     * never manufacture a positive observation. */
    M3_SHD_EVIDENCE_UNKNOWN = 0,
    M3_SHD_EVIDENCE_OBSERVED,
    M3_SHD_EVIDENCE_UNAVAILABLE,
    M3_SHD_EVIDENCE_THEORETICAL,
    M3_SHD_EVIDENCE_NOT_APPLICABLE
} m3_shd_evidence_state_t;

/* The topology is deliberately explicit rather than inferred from a drive
 * count.  A resource may name one parent (drive -> controller -> upstream),
 * which is sufficient to model the shared-controller and shared-upstream
 * fixtures while keeping admission bounded and deterministic.  New kinds
 * (CPU_ISLAND, GPU_ISLAND, MEMORY_DOMAIN, LINK) carry kind-specific payloads
 * in the second half of the struct; unused fields remain zero. */
typedef struct {
    m3_shd_resource_id_t resource_id;
    m3_shd_resource_kind_t kind;
    m3_shd_resource_id_t parent_id;        /* immediate parent in the topology */
    uint32_t max_inflight;                  /* bounded admission */
    uint64_t max_inflight_bytes;
    /* Kind-specific payload.  Fields that don't apply to the kind are zero. */
    uint32_t isa_caps;                      /* CPU_ISLAND: m3_shd_isa_caps_t bitfield */
    uint32_t cores;                         /* CPU_ISLAND: physical cores in the island */
    m3_shd_resource_id_t parent_mem_id;    /* CPU_ISLAND / GPU_ISLAND: parent memory */
    uint64_t vram_bytes;                    /* GPU_ISLAND: per-GPU VRAM */
    uint16_t p2p_class;                     /* GPU_ISLAND: 0=NO_P2P, 1=PIX, 2=P2P_4G, 3=NVLINK */
    m3_shd_resource_id_t pcie_parent_id;    /* GPU_ISLAND / LINK: PCIe parent id */
    uint64_t rate_bytes_per_s;              /* LINK / MEMORY_DOMAIN: aggregate transfer rate */
    uint32_t bandwidth_class;               /* MEMORY_DOMAIN: m3_shd_mem_class_t */
    uint8_t  eligible;                      /* machine eligibility: 1 = may be used */
} m3_shd_resource_t;

/* A physically available, content-equivalent weight copy.  Candidate paths
 * carry the resources; this record carries the model/tensor identity and
 * byte range that makes two copies eligible for the same request. */
typedef struct {
    uint64_t copy_id;
    uint64_t model_id;
    uint64_t tensor_id;
    uint64_t bytes;
    m3_shd_resource_id_t drive_id;
    uint8_t available;
} m3_shd_weight_copy_t;

/* Request-size bands. The 100 MiB request size used in the plan's example
 * fixtures (docs/.../m3-execution-step7-plan-2026-09-06.md §4.1) sits
 * firmly in M3_SHD_SIZE_LARGE (> 16 MiB). UNKNOWN is reserved for "no
 * classification is honest" -- bytes == 0 (caller mistake) or values past
 * UINT64_MAX/2 (a "no length known" idiom the integrator would overflow
 * on). */
typedef enum {
    M3_SHD_SIZE_TINY    = 0, /* <= 64 KiB  */
    M3_SHD_SIZE_SMALL   = 1, /* <= 1 MiB   */
    M3_SHD_SIZE_MEDIUM  = 2, /* <= 16 MiB  */
    M3_SHD_SIZE_LARGE   = 3, /* >  16 MiB  (host of the 100 MiB fixture) */
    M3_SHD_SIZE_UNKNOWN = 4
} m3_shd_size_class_t;

/* Array dimension. Includes UNKNOWN so the array index space equals the
 * enum value space; the UNKNOWN slot is intentionally never populated by
 * the calibration loader, and any access at that index is undefined --
 * m3_shd_classify_size's UNKNOWN return is the predictor's signal to
 * short-circuit before touching the array. */
#define M3_SHD_SIZE_CLASS_COUNT 5

/* In-flight concurrency bands, mirroring 7b's bounded inflight ranges.
 * The classify thresholds are {1, 2, 4, 8} and carve the bands as:
 *   FREE  <= 1
 *   LOW   == 2
 *   MID   3..4
 *   SAT   5..7    -- saturated but not yet over-committed
 *   HIGH  >= 8    -- over-committed
 * UNKNOWN is reserved for 0 in-flight (no concurrency data -- any rate
 * would be a default capacity fill, which INV-10 forbids). */
typedef enum {
    M3_SHD_LOAD_FREE    = 0, /* 1 in-flight         */
    M3_SHD_LOAD_LOW     = 1, /* 2 in-flight         */
    M3_SHD_LOAD_MID     = 2, /* 3..4 in-flight      */
    M3_SHD_LOAD_HIGH    = 3, /* >= 8 in-flight      */
    M3_SHD_LOAD_SAT     = 4, /* 5..7 in-flight      */
    M3_SHD_LOAD_UNKNOWN = 5
} m3_shd_load_class_t;

/* Array dimension. Mirrors M3_SHD_SIZE_CLASS_COUNT rationale: array
 * index space equals enum value space; the UNKNOWN slot is unused and
 * the predictor short-circuits on UNKNOWN before indexing. */
#define M3_SHD_LOAD_CLASS_COUNT 6

/* Named thresholds for size classification. KiB/MiB are binary (1024),
 * matching the engine's tensor/byte counting convention. Kept as named
 * constants so the source of a number in m3_shd_classify_size is
 * self-evident. */
#define M3_SHD_SIZE_TINY_MAX_BYTES    ((uint64_t)64ULL  * 1024ULL)
#define M3_SHD_SIZE_SMALL_MAX_BYTES   ((uint64_t)1024ULL * 1024ULL)
#define M3_SHD_SIZE_MEDIUM_MAX_BYTES  ((uint64_t)16ULL  * 1024ULL * 1024ULL)
/* "Beyond UINT64_MAX/2" is the spec-mandated UNKNOWN sentinel: a
 * caller passing UINT64_MAX-as-bytes (the "no length known" idiom)
 * cannot be classified without overflow risk on the integrator side. */
#define M3_SHD_SIZE_UNKNOWN_MIN_BYTES ((uint64_t)(UINT64_MAX / 2))

/* Per-resource service profile. NOT opaque: the calibration loader
 * writes directly into the table and downstream consumers read it.
 * Fields are laid out in the order the spec mandates, with the
 * INV-10 honesty fields grouped first.
 *
 *   topology_hash             Every resource_profile in one topology must
 *                             carry the same hash; a mismatch anywhere
 *                             rejects the calibration in
 *                             m3_shd_calib_is_sound.
 *   units_per_second_explicit 0 means "no sound units-per-second
 *                             calibration was recorded". is_sound walks
 *                             the entire topology and treats any zero
 *                             as a calibration-collapse -- INV-10
 *                             enforcement path.
 *   version                   Profile-format version; the calibration
 *                             loader uses this to refuse profiles it
 *                             cannot interpret.
 *   sample_count              Number of recorded samples underlying this
 *                             calibration; 0 means syntactically present
 *                             but semantically empty.
 *
 * startup_ns and rate_bytes_per_s are sized M3_SHD_SIZE_CLASS_COUNT x
 * M3_SHD_LOAD_CLASS_COUNT. The *_UNKNOWN slots are never populated.
 * residual_ns is the recorded upper uncertainty margin for the same bucket;
 * it is never tuned at prediction time.  A non-zero residual must have a
 * residual_percentile_ppm must be 950000 (p95) or 990000 (p99); this prevents
 * an unlabelled or unsupported percentile from masquerading as tail evidence. */
typedef struct {
    m3_shd_resource_id_t resource_id;
    uint64_t topology_hash;
    uint8_t  units_per_second_explicit;
    uint8_t  version;
    uint32_t sample_count;
    uint64_t startup_ns     [M3_SHD_SIZE_CLASS_COUNT][M3_SHD_LOAD_CLASS_COUNT];
    uint64_t rate_bytes_per_s[M3_SHD_SIZE_CLASS_COUNT][M3_SHD_LOAD_CLASS_COUNT];
    uint64_t residual_ns     [M3_SHD_SIZE_CLASS_COUNT][M3_SHD_LOAD_CLASS_COUNT];
    uint32_t residual_percentile_ppm
                              [M3_SHD_SIZE_CLASS_COUNT][M3_SHD_LOAD_CLASS_COUNT];
} m3_shd_resource_profile_t;

/* Optional serial post-I/O conversion stage.  Duration rows are deliberately
 * sparse: an unmeasured size/load class remains UNKNOWN rather than becoming
 * an assumed zero-duration conversion. */
typedef struct {
    uint8_t enabled;
    uint8_t version;
    uint32_t sample_count;
    uint64_t topology_hash;
    uint32_t max_inflight;
    uint64_t max_inflight_bytes;
    uint64_t duration_ns[M3_SHD_SIZE_CLASS_COUNT]
                          [M3_SHD_LOAD_CLASS_COUNT];
    uint64_t residual_ns[M3_SHD_SIZE_CLASS_COUNT]
                         [M3_SHD_LOAD_CLASS_COUNT];
    uint32_t residual_percentile_ppm[M3_SHD_SIZE_CLASS_COUNT]
                                     [M3_SHD_LOAD_CLASS_COUNT];
} m3_shd_conversion_profile_t;

/* Topology-level profile: one row per resource on the read path.
 * Caller-owned. This commit declares only the shape; commit 3's
 * m3_shadow_plan.h opens the planner against a borrowed, immutable
 * instance whose lifetime outlives every snapshot. */
typedef struct {
    uint64_t topology_hash;
    size_t   n_resources;
    const m3_shd_resource_t *resources;
    size_t   n_copies;
    const m3_shd_weight_copy_t *copies;
} m3_shd_topology_profile_t;

/* Calibration set: the immutable handle every snapshot consumes.
 *
 * Ownership rule: callers own the calibration set and all arrays it points
 * to. The planner borrows the immutable set for the lifetime of its handle;
 * m3_shd_close() frees only the planner and never the caller-owned set.
 *
 *   topology_hash  Duplicate of topology->topology_hash so a single
 *                  load-time compare rules out hash drift.
 *   topology       Borrowed topology profile this calibration is keyed
 *                  against. NULL only when the planner was opened
 *                  against an explicit synthetic profile (NOT_RUN path);
 *                  m3_shd_calib_is_sound treats NULL as unsound.
 *   version        Must equal M3_SHD_CALIB_VERSION or the calibration
 *                  is unsound regardless of the underlying samples.
 *   conversion     Optional serial post-I/O profile; when enabled it must
 *                  carry the same topology hash and explicit bounds. */
typedef struct {
    uint64_t topology_hash;
    const m3_shd_topology_profile_t *topology;
    size_t n_profiles;
    const m3_shd_resource_profile_t *profiles;
    uint32_t version;
    /* Optional virtual serial conversion resource.  Zero-initialized means
     * that the calibrated path has no planner-visible conversion stage, so
     * existing storage-only profiles retain their prior behavior. */
    m3_shd_conversion_profile_t conversion;
    /* Optional starvation bound for the bounded demand queue.  Zero keeps
     * the legacy need/enqueue/request-ID ordering; a positive value makes
     * requests at least this old age-priority, in enqueue order. */
    uint64_t max_queue_age_ns;
} m3_shd_calibration_set_t;

/* Pure inline predicate: returns 1 iff the calibration is sound for
 * INV-10 honest prediction, 0 otherwise.
 *
 * The body chains the INV-10 honesty checks inline so the review can
 * verify the predicate by reading top-to-bottom; no helper functions are
 * introduced (the spec forbids them). The checks are, in order:
 *
 *   1.  the set pointer itself;
 *   2.  the borrowed topology pointer;
 *   3.  the topology carries at least one resource;
 *   4.  the topology's resources pointer is non-NULL;
 *   5.  the planner version matches the pinned M3_SHD_CALIB_VERSION;
 *   6.  EVERY present resource_profile has the pinned format version,
 *       units_per_second_explicit > 0, nonzero samples, and the matching
 *       topology hash. A single invalid row anywhere on the read path
 *       collapses the whole calibration to UNKNOWN.  A missing profile row
 *       is permitted so a topology can be partially calibrated; any path
 *       requiring that resource is UNKNOWN, never a default capacity fill.
 *       Every present profile must reference exactly one declared resource,
 *       and profile resource IDs may not be duplicated.
 *   7.  Every non-zero residual carries an explicit high-percentile label;
 *       zero residuals carry no tail label.  This keeps the uncertainty
 *       margin statistically interpretable rather than silently tunable.
 *   8.  An enabled conversion profile passes the same version/hash/sample,
 *       bound, and residual-tail checks.
 *
 * The short-circuit for-loop is the standard C idiom for "every"; an
 * inline helper would only hide the same code behind a name. */
static inline int m3_shd_calib_is_sound(const m3_shd_calibration_set_t *s) {
    if (!s
        || !s->topology
        || !s->topology->resources
        || s->topology->n_resources == 0
        || s->version != M3_SHD_CALIB_VERSION) {
        return 0;
    }
    if (s->n_profiles && !s->profiles) {
        return 0;
    }
    if (s->n_profiles > s->topology->n_resources) {
        return 0;
    }
    /* A calibration set may intentionally omit a resource profile.  The
     * predictor's profile_for() check then makes any path using that
     * resource UNKNOWN instead of fabricating capacity. */
    for (size_t i = 0; i < s->n_profiles; i++) {
        size_t resource_matches = 0;
        for (size_t ri = 0; ri < s->topology->n_resources; ++ri) {
            const m3_shd_resource_t *r = &s->topology->resources[ri];
            if (r->resource_id != s->profiles[i].resource_id) continue;
            resource_matches++;
        }
        if (resource_matches != 1) {
            return 0;
        }
        for (size_t prior = 0; prior < i; ++prior) {
            if (s->profiles[prior].resource_id
                == s->profiles[i].resource_id) {
                return 0;
            }
        }
        if (s->profiles[i].units_per_second_explicit == 0
            || s->profiles[i].sample_count == 0
            || s->profiles[i].version != M3_SHD_CALIB_VERSION
            || s->profiles[i].topology_hash != s->topology_hash) {
            return 0;
        }
        for (size_t size = 0; size < M3_SHD_SIZE_CLASS_COUNT; size++) {
            for (size_t load = 0; load < M3_SHD_LOAD_CLASS_COUNT; load++) {
                uint64_t residual = s->profiles[i].residual_ns[size][load];
                uint32_t percentile =
                    s->profiles[i].residual_percentile_ppm[size][load];
                if (residual == 0) {
                    if (percentile != M3_SHD_TAIL_PERCENTILE_NONE_PPM)
                        return 0;
                } else if (percentile != M3_SHD_TAIL_PERCENTILE_P95_PPM
                           && percentile != M3_SHD_TAIL_PERCENTILE_P99_PPM) {
                    return 0;
                }
            }
        }
    }
    if (s->conversion.enabled) {
        if (s->conversion.enabled != 1
            || s->conversion.version != M3_SHD_CALIB_VERSION
            || s->conversion.sample_count == 0
            || s->conversion.topology_hash != s->topology_hash
            || s->conversion.max_inflight == 0
            || s->conversion.max_inflight_bytes == 0) {
            return 0;
        }
        for (size_t size = 0; size < M3_SHD_SIZE_CLASS_COUNT; size++) {
            for (size_t load = 0; load < M3_SHD_LOAD_CLASS_COUNT; load++) {
                uint64_t residual = s->conversion.residual_ns[size][load];
                uint32_t percentile =
                    s->conversion.residual_percentile_ppm[size][load];
                if (residual == 0) {
                    if (percentile != M3_SHD_TAIL_PERCENTILE_NONE_PPM)
                        return 0;
                } else if (percentile != M3_SHD_TAIL_PERCENTILE_P95_PPM
                           && percentile != M3_SHD_TAIL_PERCENTILE_P99_PPM) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

/* Pure inline classifier: request size in bytes -> size_class band.
 *
 * UNKNOWN is returned for bytes == 0 (caller passed a degenerate size)
 * and for values past UINT64_MAX/2 (the "no length known" idiom the
 * integrator would overflow on). The remaining thresholds align with
 * M3_SHD_SIZE_TINY_MAX_BYTES, M3_SHD_SIZE_SMALL_MAX_BYTES and
 * M3_SHD_SIZE_MEDIUM_MAX_BYTES; anything above 16 MiB is LARGE and
 * includes the 100 MiB fixture from the plan's §4.1. */
static inline m3_shd_size_class_t m3_shd_classify_size(uint64_t bytes) {
    if (bytes == 0 || bytes >= M3_SHD_SIZE_UNKNOWN_MIN_BYTES) {
        return M3_SHD_SIZE_UNKNOWN;
    }
    if (bytes <= M3_SHD_SIZE_TINY_MAX_BYTES)   return M3_SHD_SIZE_TINY;
    if (bytes <= M3_SHD_SIZE_SMALL_MAX_BYTES)  return M3_SHD_SIZE_SMALL;
    if (bytes <= M3_SHD_SIZE_MEDIUM_MAX_BYTES) return M3_SHD_SIZE_MEDIUM;
    return M3_SHD_SIZE_LARGE;
}

/* Pure inline classifier: in-flight count -> load_class band.
 *
 * Thresholds {1, 2, 4, 8} carve the bands FREE, LOW, MID, SAT, HIGH;
 * SAT occupies 5..7 -- the saturated-but-not-yet-overloaded range
 * 7b's bounded inflight math expects. 0 is UNKNOWN: the predictor has
 * no concurrency data and any rate it could emit would be a default
 * capacity fill, which INV-10 forbids. */
static inline m3_shd_load_class_t m3_shd_classify_load(uint32_t in_flight) {
    if (in_flight == 0) return M3_SHD_LOAD_UNKNOWN;
    if (in_flight <= 1) return M3_SHD_LOAD_FREE;
    if (in_flight <= 2) return M3_SHD_LOAD_LOW;
    if (in_flight <= 4) return M3_SHD_LOAD_MID;
    if (in_flight <  8) return M3_SHD_LOAD_SAT;
    return M3_SHD_LOAD_HIGH;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_M3_SHADOW_PLAN_CALIB_H */
