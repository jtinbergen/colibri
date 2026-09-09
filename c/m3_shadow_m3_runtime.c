#include "m3_shadow_m3_runtime.h"
#include "m3_active_admission.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define M3_SHD_RUNTIME_MAX_TEXT (4u * 1024u * 1024u)
#define M3_SHD_RUNTIME_MAX_REPLICAS 16u
#define M3_SHD_RUNTIME_MAX_TRACE 4096u

typedef struct {
    uint8_t used;
    uint8_t completed;
    int32_t load_result;
    uint32_t path_count;
    uint64_t request_id;
    uint64_t model_id;
    uint64_t forward_id;
    uint64_t generation;
    uint64_t consumer_node;
    uint64_t consumer_need_ns;
    uint8_t component_mask;
    uint64_t predicted_ready_ns;
    uint64_t predicted_uncertainty_ns;
    uint64_t predicted_resource_id;
    uint64_t issue_ns;
    uint64_t ready_ns;
    int32_t layer;
    int32_t expert;
    uint64_t bytes;
    uint64_t actual_drive_id;
    uint64_t path[M3_SHD_M3_TRACE_PATH_MAX];
    uint32_t path_inflight[M3_SHD_M3_TRACE_PATH_MAX];
    uint64_t path_bytes[M3_SHD_M3_TRACE_PATH_MAX];
    uint32_t concurrency;
    uint8_t demand;
    uint8_t evidence_state;
    uint8_t confidence;
    uint8_t path_complete;
    uint64_t service_latency_ns;
    uint64_t run_id;
    uint64_t topology_hash;
    uint64_t profile_hash;
    char cache_condition[24];
} m3_shd_runtime_trace_row_t;

typedef struct {
    m3_shd_resource_t resources[M3_SHD_MAX_RESOURCES];
    m3_shd_resource_profile_t profiles[M3_SHD_MAX_RESOURCES];
    m3_shd_weight_copy_t copies[M3_SHD_MAX_COPIES];
    uint64_t replica_drive_ids[M3_SHD_RUNTIME_MAX_REPLICAS];
    m3_shd_topology_profile_t topology;
    m3_shd_calibration_set_t calibration;
    m3_shadow_plan_t *planner;
    /* Installed at the live issuance boundary, but opened untrusted until a
     * future Gate-8 profile is explicitly admitted. */
    m3_adm_dispatcher_t *admission;
    size_t replica_count;
    uint64_t configured_model_id;
    uint64_t synthetic_need_delta_ns;
    uint64_t run_id;
    char cache_condition[24];
    char log_path[1024];
    char trace_path[1024];
    m3_shd_runtime_trace_row_t trace[M3_SHD_RUNTIME_MAX_TRACE];
    size_t trace_count;
    int dump_registered;
} m3_shd_runtime_storage_t;

static m3_shd_runtime_storage_t g_runtime;
static pthread_mutex_t g_runtime_init_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_runtime_trace_mx = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int g_runtime_ready;
static _Atomic uint64_t g_runtime_forward_seq;
static _Atomic uint64_t g_runtime_request_seq;
static _Thread_local m3_shd_m3_exec_context_t g_runtime_context;

static const char *runtime_evidence_name(uint8_t state) {
    switch ((m3_shd_evidence_state_t)state) {
    case M3_SHD_EVIDENCE_OBSERVED: return "OBSERVED";
    case M3_SHD_EVIDENCE_UNKNOWN: return "UNKNOWN";
    case M3_SHD_EVIDENCE_UNAVAILABLE: return "UNAVAILABLE";
    case M3_SHD_EVIDENCE_THEORETICAL: return "THEORETICAL";
    case M3_SHD_EVIDENCE_NOT_APPLICABLE: return "NOT_APPLICABLE";
    default: return "UNKNOWN";
    }
}

static const char *runtime_confidence_name(uint8_t confidence) {
    switch (confidence) {
    case 1: return "SINGLE";
    case 2: return "REPEATED";
    default: return "UNKNOWN";
    }
}

static int runtime_trace_append_locked(char *dst, size_t cap, size_t *used,
                                       const char *text) {
    if (!dst || !used || !text) return 0;
    size_t length = strlen(text);
    if (*used > cap || length > cap - *used) return 0;
    memcpy(dst + *used, text, length);
    *used += length;
    return 1;
}

static int runtime_trace_encode_locked(char *dst, size_t cap,
                                       size_t *bytes_out) {
    static const char header[] =
        "request_id,model_id,forward_id,generation,consumer_node,layer,"
        "expert,bytes,run_id,topology_hash,profile_hash,cache_condition,"
        "component_mask,actual_drive_id,actual_path,actual_load,"
        "path_inflight,path_bytes,concurrency,request_type,evidence_state,confidence,"
        "path_complete,service_latency_ns,"
        "consumer_need_ns,"
        "predicted_ready_ns,predicted_uncertainty_ns,predicted_resource_id,"
        "issue_ns,ready_ns,"
        "load_result,completed\n";
    size_t used = 0;
    if (dst && cap) dst[0] = '\0';
    if (bytes_out) *bytes_out = 0;
    /* First pass computes the committed size without requiring a second
     * mutable buffer.  The rows are bounded and each field is integer-only. */
    for (int pass = 0; pass < 2; ++pass) {
        size_t cursor = 0;
        if (dst && pass == 1 && cap)
            dst[0] = '\0';
        if (dst && pass == 1) {
            if (!runtime_trace_append_locked(dst, cap, &cursor, header))
                return 0;
        } else {
            cursor = strlen(header);
        }
        for (size_t i = 0; i < g_runtime.trace_count; ++i) {
            const m3_shd_runtime_trace_row_t *row = &g_runtime.trace[i];
            if (!row->used) continue;
            char line[1024];
            int n = snprintf(line, sizeof(line),
                "%llu,%llu,%llu,%llu,%llu,%d,%d,%llu,%llu,%llu,%llu,%s,%u,%llu,",
                (unsigned long long)row->request_id,
                (unsigned long long)row->model_id,
                (unsigned long long)row->forward_id,
                (unsigned long long)row->generation,
                (unsigned long long)row->consumer_node,
                row->layer, row->expert,
                (unsigned long long)row->bytes,
                (unsigned long long)row->run_id,
                (unsigned long long)row->topology_hash,
                (unsigned long long)row->profile_hash,
                row->cache_condition,
                (unsigned)row->component_mask,
                (unsigned long long)row->actual_drive_id);
            if (n < 0 || (size_t)n >= sizeof(line)) return 0;
            if (dst && pass == 1) {
                if (!runtime_trace_append_locked(dst, cap, &cursor, line))
                    return 0;
            } else {
                cursor += (size_t)n;
            }
            for (uint32_t p = 0; p < row->path_count; ++p) {
                n = snprintf(line, sizeof(line), "%s%llu",
                             p ? "|" : "",
                             (unsigned long long)row->path[p]);
                if (n < 0 || (size_t)n >= sizeof(line)) return 0;
                if (dst && pass == 1) {
                    if (!runtime_trace_append_locked(dst, cap, &cursor, line))
                        return 0;
                } else {
                    cursor += (size_t)n;
                }
            }
            if (dst && pass == 1) {
                if (!runtime_trace_append_locked(dst, cap, &cursor, ","))
                    return 0;
            } else {
                cursor++;
            }
            for (uint32_t p = 0; p < row->path_count; ++p) {
                n = snprintf(line, sizeof(line), "%s%llu:%u:%llu",
                             p ? "|" : "",
                             (unsigned long long)row->path[p],
                             row->path_inflight[p],
                             (unsigned long long)row->path_bytes[p]);
                if (n < 0 || (size_t)n >= sizeof(line)) return 0;
                if (dst && pass == 1) {
                    if (!runtime_trace_append_locked(dst, cap, &cursor, line))
                        return 0;
                } else {
                    cursor += (size_t)n;
                }
            }
            if (dst && pass == 1) {
                if (!runtime_trace_append_locked(dst, cap, &cursor, ","))
                    return 0;
            } else {
                cursor++;
            }
            for (uint32_t p = 0; p < row->path_count; ++p) {
                n = snprintf(line, sizeof(line), "%s%u",
                             p ? "|" : "", row->path_inflight[p]);
                if (n < 0 || (size_t)n >= sizeof(line)) return 0;
                if (dst && pass == 1) {
                    if (!runtime_trace_append_locked(dst, cap, &cursor, line))
                        return 0;
                } else {
                    cursor += (size_t)n;
                }
            }
            if (dst && pass == 1) {
                if (!runtime_trace_append_locked(dst, cap, &cursor, ","))
                    return 0;
            } else {
                cursor++;
            }
            for (uint32_t p = 0; p < row->path_count; ++p) {
                n = snprintf(line, sizeof(line), "%s%llu",
                             p ? "|" : "",
                             (unsigned long long)row->path_bytes[p]);
                if (n < 0 || (size_t)n >= sizeof(line)) return 0;
                if (dst && pass == 1) {
                    if (!runtime_trace_append_locked(dst, cap, &cursor, line))
                        return 0;
                } else {
                    cursor += (size_t)n;
                }
            }
            n = snprintf(line, sizeof(line), ",%u", row->concurrency);
            if (n < 0 || (size_t)n >= sizeof(line)) return 0;
            if (dst && pass == 1) {
                if (!runtime_trace_append_locked(dst, cap, &cursor, line))
                    return 0;
            } else {
                cursor += (size_t)n;
            }
            n = snprintf(line, sizeof(line), ",%s,%s,%s,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d,%u\n",
                         row->demand ? "demand" : "prefetch",
                         runtime_evidence_name(row->evidence_state),
                         runtime_confidence_name(row->confidence),
                         (unsigned)row->path_complete,
                         (unsigned long long)row->service_latency_ns,
                         (unsigned long long)row->consumer_need_ns,
                         (unsigned long long)row->predicted_ready_ns,
                         (unsigned long long)row->predicted_uncertainty_ns,
                         (unsigned long long)row->predicted_resource_id,
                         (unsigned long long)row->issue_ns,
                         (unsigned long long)row->ready_ns,
                         row->load_result, (unsigned)row->completed);
            if (n < 0 || (size_t)n >= sizeof(line)) return 0;
            if (dst && pass == 1) {
                if (!runtime_trace_append_locked(dst, cap, &cursor, line))
                    return 0;
            } else {
                cursor += (size_t)n;
            }
        }
        if (bytes_out) *bytes_out = cursor;
        if (pass == 0) used = cursor;
    }
    if (dst && cap > used) dst[used] = '\0';
    return dst && cap > used;
}

static void runtime_dump_trace(void) {
    if (!g_runtime.trace_path[0] || !m3_shd_m3_runtime_enabled()) return;
    size_t bytes = 0;
    pthread_mutex_lock(&g_runtime_trace_mx);
    int enough = runtime_trace_encode_locked(NULL, 0, &bytes);
    char *text = NULL;
    if (bytes < SIZE_MAX) text = (char *)malloc(bytes + 1u);
    if (text) runtime_trace_encode_locked(text, bytes + 1u, &bytes);
    pthread_mutex_unlock(&g_runtime_trace_mx);
    if (enough || text) {
        FILE *f = fopen(g_runtime.trace_path, "wb");
        if (f) {
            if (text) fwrite(text, 1, bytes, f);
            fclose(f);
            fprintf(stderr, "[M3_SHADOW] wrote %zu trace bytes to %s\n",
                    bytes, g_runtime.trace_path);
        }
    }
    free(text);
}

static void runtime_dump_log(void) {
    if (!g_runtime.log_path[0] || !m3_shd_m3_runtime_enabled()) return;
    size_t bytes = 0;
    if (m3_shd_decision_log_copy(g_runtime.planner, NULL, 0, &bytes)
        || !bytes) return;
    char *log = (char *)malloc(bytes + 1u);
    if (!log || !m3_shd_decision_log_copy(g_runtime.planner, log,
                                         bytes + 1u, &bytes)) {
        free(log);
        return;
    }
    FILE *f = fopen(g_runtime.log_path, "wb");
    if (f) {
        fwrite(log, 1, bytes, f);
        fclose(f);
        fprintf(stderr, "[M3_SHADOW] wrote %zu decision-log bytes to %s\n",
                bytes, g_runtime.log_path);
    }
    free(log);
}

static uint64_t runtime_clock_now(void *opaque) {
    (void)opaque;
    return m3_shd_m3_runtime_now_ns();
}

uint64_t m3_shd_m3_runtime_now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    if (!QueryPerformanceFrequency(&frequency)
        || !QueryPerformanceCounter(&counter) || !frequency.QuadPart)
        return 0;
    return (uint64_t)((unsigned __int128)counter.QuadPart * UINT64_C(1000000000)
                      / (uint64_t)frequency.QuadPart);
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000)
         + (uint64_t)ts.tv_nsec;
#endif
}

static int read_bounded_file(const char *path, char **text_out) {
    if (text_out) *text_out = NULL;
    if (!path || !*path || !text_out) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long length = ftell(f);
    if (length < 0 || (unsigned long)length >= M3_SHD_RUNTIME_MAX_TEXT) {
        fclose(f); return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    char *text = (char *)malloc((size_t)length + 1u);
    if (!text) { fclose(f); return -1; }
    size_t got = fread(text, 1, (size_t)length, f);
    fclose(f);
    if (got != (size_t)length) { free(text); return -1; }
    text[got] = '\0';
    *text_out = text;
    return 0;
}

static int parse_u64_any(const char *text, uint64_t *out) {
    if (!text || !*text || !out) return -1;
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 0);
    if (errno || end == text || *end) return -1;
    *out = (uint64_t)value;
    return 0;
}

static int parse_u64(const char *text, uint64_t *out) {
    if (parse_u64_any(text, out) || !*out) return -1;
    return 0;
}

static int parse_replica_map(const char *text, uint64_t *out, size_t *count_out) {
    if (count_out) *count_out = 0;
    if (!text || !*text || !out || !count_out) return -1;
    char copy[512];
    size_t length = strlen(text);
    if (length >= sizeof(copy)) return -1;
    memcpy(copy, text, length + 1u);
    char *cursor = copy;
    size_t count = 0;
    while (*cursor) {
        char *comma = strchr(cursor, ',');
        if (comma) *comma = '\0';
        char *colon = strchr(cursor, ':');
        if (!colon || count >= M3_SHD_RUNTIME_MAX_REPLICAS) return -1;
        *colon = '\0';
        uint64_t replica = 0, drive = 0;
        if (parse_u64_any(cursor, &replica) || parse_u64(colon + 1, &drive)
            || replica >= M3_SHD_RUNTIME_MAX_REPLICAS || !drive)
            return -1;
        if (replica != count) return -1;
        out[count++] = drive;
        if (!comma) break;
        cursor = comma + 1;
        if (!*cursor) return -1;
    }
    *count_out = count;
    return count ? 0 : -1;
}

static int parse_cache_condition(const char *text, char *out, size_t cap) {
    if (!out || cap < 2) return -1;
    const char *value = text && *text ? text : "UNKNOWN";
    if (strcmp(value, "cold") && strcmp(value, "warm")
        && strcmp(value, "mixed") && strcmp(value, "unknown")
        && strcmp(value, "COLD") && strcmp(value, "WARM")
        && strcmp(value, "MIXED") && strcmp(value, "UNKNOWN")) return -1;
    size_t n = strlen(value);
    if (n >= cap) return -1;
    memcpy(out, value, n + 1u);
    for (size_t i = 0; out[i]; ++i) {
        if (out[i] >= 'a' && out[i] <= 'z')
            out[i] = (char)(out[i] - 'a' + 'A');
    }
    return 0;
}

static int runtime_init_locked(const char *text, uint64_t topology_hash,
                               const uint64_t *replica_drive_ids,
                               size_t replica_count) {
    if (atomic_load_explicit(&g_runtime_ready, memory_order_acquire)) return 1;
    if (!text || !topology_hash || !replica_drive_ids || !replica_count
        || replica_count > M3_SHD_RUNTIME_MAX_REPLICAS) return 0;
    memset(&g_runtime, 0, sizeof(g_runtime));
    char error[192] = {0};
    size_t resource_count = 0, profile_count = 0, copy_count = 0;
    if (m3_shd_parse_text(text, topology_hash,
                          g_runtime.resources, M3_SHD_MAX_RESOURCES,
                          &resource_count, g_runtime.profiles,
                          M3_SHD_MAX_RESOURCES, &profile_count,
                          g_runtime.copies, M3_SHD_MAX_COPIES, &copy_count,
                          &g_runtime.topology, &g_runtime.calibration,
                          error, sizeof(error)) != 0) {
        fprintf(stderr, "[M3_SHADOW] configuration disabled: %s\n",
                error[0] ? error : "invalid topology/profile");
        return 0;
    }
    memcpy(g_runtime.replica_drive_ids, replica_drive_ids,
           replica_count * sizeof(replica_drive_ids[0]));
    g_runtime.replica_count = replica_count;
    g_runtime.planner = m3_shd_open(&g_runtime.topology,
                                    &g_runtime.calibration,
                                    (m3_shd_clock_t){runtime_clock_now, NULL},
                                    error, sizeof(error));
    if (!g_runtime.planner) {
        fprintf(stderr, "[M3_SHADOW] planner disabled: %s\n",
                error[0] ? error : "cannot open planner");
        memset(&g_runtime, 0, sizeof(g_runtime));
        return 0;
    }
    g_runtime.admission = m3_adm_open(&g_runtime.topology, topology_hash,
                                      m3_shd_profile_hash(g_runtime.planner),
                                      0, error, sizeof(error));
    if (!g_runtime.admission) {
        /* Observation remains valid; no current configuration can activate
         * admission, so this cannot alter the existing engine path. */
        fprintf(stderr, "[M3_SHADOW] Step-8 admission seam unavailable: %s\n",
                error[0] ? error : "allocation failed");
    }
    const char *log_path = getenv("COLI_M3_SHADOW_LOG");
    if (log_path && *log_path
        && strlen(log_path) < sizeof(g_runtime.log_path)) {
        memcpy(g_runtime.log_path, log_path, strlen(log_path) + 1u);
    }
    const char *trace_path = getenv("COLI_M3_SHADOW_TRACE");
    if (trace_path && *trace_path
        && strlen(trace_path) < sizeof(g_runtime.trace_path)) {
        memcpy(g_runtime.trace_path, trace_path, strlen(trace_path) + 1u);
    }
    if ((g_runtime.log_path[0] || g_runtime.trace_path[0])
        && !g_runtime.dump_registered) {
        atexit(runtime_dump_log);
        atexit(runtime_dump_trace);
        g_runtime.dump_registered = 1;
    }
    const char *model_id_text = getenv("COLI_M3_SHADOW_MODEL_ID");
    if (model_id_text && *model_id_text
        && parse_u64(model_id_text, &g_runtime.configured_model_id)) {
        fprintf(stderr, "[M3_SHADOW] configured model id ignored: invalid value\n");
        g_runtime.configured_model_id = 0;
    }
    const char *need_delta = getenv("COLI_M3_SHADOW_SYNTHETIC_NEED_DELTA_NS");
    if (need_delta && *need_delta
        && parse_u64_any(need_delta, &g_runtime.synthetic_need_delta_ns)) {
        fprintf(stderr, "[M3_SHADOW] synthetic need delta ignored: invalid value\n");
        g_runtime.synthetic_need_delta_ns = 0;
    }
    const char *run_id_text = getenv("COLI_M3_SHADOW_RUN_ID");
    if (run_id_text && *run_id_text
        && parse_u64(run_id_text, &g_runtime.run_id)) {
        fprintf(stderr, "[M3_SHADOW] run id ignored: invalid value\n");
        g_runtime.run_id = 0;
    }
    if (!g_runtime.run_id) {
        g_runtime.run_id = m3_shd_m3_runtime_now_ns();
        if (!g_runtime.run_id) g_runtime.run_id = 1;
    }
    if (parse_cache_condition(getenv("COLI_M3_SHADOW_CACHE_CONDITION"),
                              g_runtime.cache_condition,
                              sizeof(g_runtime.cache_condition))) {
        fprintf(stderr, "[M3_SHADOW] cache condition ignored: expected "
                        "cold, warm, mixed, or unknown\n");
        strcpy(g_runtime.cache_condition, "UNKNOWN");
    }
    atomic_store_explicit(&g_runtime_request_seq, 1, memory_order_relaxed);
    atomic_store_explicit(&g_runtime_forward_seq, 1, memory_order_relaxed);
    atomic_store_explicit(&g_runtime_ready, 1, memory_order_release);
    fprintf(stderr, "[M3_SHADOW] enabled: %zu resources, %zu copies, %zu replica map entries\n",
            resource_count, copy_count, replica_count);
    return 1;
}

int m3_shd_m3_runtime_init_text(const char *text, uint64_t topology_hash,
                                const uint64_t *replica_drive_ids,
                                size_t replica_count) {
    pthread_mutex_lock(&g_runtime_init_mx);
    int result = runtime_init_locked(text, topology_hash, replica_drive_ids,
                                     replica_count);
    pthread_mutex_unlock(&g_runtime_init_mx);
    return result;
}

int m3_shd_m3_runtime_init_from_env(void) {
    const char *path = getenv("COLI_M3_SHADOW_CONFIG");
    if (!path || !*path) return 0;
    const char *hash_text = getenv("COLI_M3_SHADOW_TOPOLOGY_HASH");
    const char *map_text = getenv("COLI_M3_SHADOW_REPLICA_DRIVES");
    uint64_t topology_hash = 0;
    uint64_t replica_drive_ids[M3_SHD_RUNTIME_MAX_REPLICAS];
    size_t replica_count = 0;
    char *text = NULL;
    int result = 0;
    if (parse_u64(hash_text, &topology_hash)
        || parse_replica_map(map_text, replica_drive_ids, &replica_count)
        || read_bounded_file(path, &text) != 0) {
        fprintf(stderr, "[M3_SHADOW] configuration disabled: require readable "
                "config, topology hash, and replica map\n");
    } else {
        result = m3_shd_m3_runtime_init_text(text, topology_hash,
                                              replica_drive_ids, replica_count);
    }
    free(text);
    return result;
}

int m3_shd_m3_runtime_enabled(void) {
    return atomic_load_explicit(&g_runtime_ready, memory_order_acquire)
        && g_runtime.planner != NULL;
}

uint64_t m3_shd_m3_runtime_next_forward(void) {
    if (!m3_shd_m3_runtime_enabled()) return 0;
    uint64_t id = atomic_fetch_add_explicit(&g_runtime_forward_seq, 1,
                                            memory_order_relaxed);
    return id ? id : atomic_fetch_add_explicit(&g_runtime_forward_seq, 1,
                                                memory_order_relaxed);
}

uint64_t m3_shd_m3_runtime_model_id(uint64_t fallback) {
    if (!m3_shd_m3_runtime_enabled()) return fallback;
    return g_runtime.configured_model_id
        ? g_runtime.configured_model_id : fallback;
}

uint64_t m3_shd_m3_runtime_synthetic_need_ns(void) {
    if (!m3_shd_m3_runtime_enabled() || !g_runtime.synthetic_need_delta_ns)
        return 0;
    uint64_t now = m3_shd_m3_runtime_now_ns();
    if (!now || UINT64_MAX - now < g_runtime.synthetic_need_delta_ns)
        return 0;
    return now + g_runtime.synthetic_need_delta_ns;
}

void m3_shd_m3_runtime_context_push(
    const m3_shd_m3_exec_context_t *next,
    m3_shd_m3_exec_context_t *previous) {
    if (previous) *previous = g_runtime_context;
    g_runtime_context = next ? *next : (m3_shd_m3_exec_context_t){0};
}

void m3_shd_m3_runtime_context_pop(
    const m3_shd_m3_exec_context_t *previous) {
    g_runtime_context = previous ? *previous : (m3_shd_m3_exec_context_t){0};
}

static uint32_t runtime_trace_begin(const m3_shd_request_t *request,
                                    const m3_shd_candidate_t *actual,
                                    const m3_shd_resource_occupancy_t *occupancy,
                                    const m3_shd_decision_record_t *decision,
                                    uint64_t issue_ns,
                                    uint8_t component_mask) {
    if (!request || !actual || !decision) return M3_SHD_M3_TRACE_INDEX_NONE;
    pthread_mutex_lock(&g_runtime_trace_mx);
    if (g_runtime.trace_count >= M3_SHD_RUNTIME_MAX_TRACE) {
        pthread_mutex_unlock(&g_runtime_trace_mx);
        return M3_SHD_M3_TRACE_INDEX_NONE;
    }
    uint32_t index = (uint32_t)g_runtime.trace_count++;
    m3_shd_runtime_trace_row_t *row = &g_runtime.trace[index];
    *row = (m3_shd_runtime_trace_row_t){0};
    row->used = 1;
    row->load_result = INT32_MIN;
    row->request_id = request->request_id;
    row->run_id = g_runtime.run_id;
    row->topology_hash = g_runtime.topology.topology_hash;
    row->profile_hash = m3_shd_profile_hash(g_runtime.planner);
    memcpy(row->cache_condition, g_runtime.cache_condition,
           sizeof(row->cache_condition));
    row->cache_condition[sizeof(row->cache_condition) - 1u] = '\0';
    row->model_id = request->model_id;
    row->forward_id = request->forward_id;
    row->generation = request->generation;
    row->consumer_node = request->consumer_node;
    row->consumer_need_ns = request->consumer_need_ns;
    row->demand = request->demand ? 1 : 0;
    row->evidence_state = M3_SHD_EVIDENCE_UNKNOWN;
    row->confidence = 1; /* one correlated observation; report promotes repeats */
    row->path_complete = actual->n_resources > 0;
    row->component_mask = component_mask;
    row->predicted_ready_ns = decision->ready_ns;
    row->predicted_uncertainty_ns = decision->uncertainty_ns;
    row->predicted_resource_id = decision->chosen_resource_id;
    row->issue_ns = issue_ns;
    row->layer = request->key.layer;
    row->expert = request->key.expert;
    row->bytes = request->bytes;
    row->actual_drive_id = actual->n_resources ? actual->resources[0] : 0;
    row->path_count = actual->n_resources > M3_SHD_M3_TRACE_PATH_MAX
        ? M3_SHD_M3_TRACE_PATH_MAX : (uint32_t)actual->n_resources;
    for (uint32_t i = 0; i < row->path_count; ++i) {
        row->path[i] = actual->resources[i];
        if (occupancy) {
            row->path_inflight[i] = occupancy[i].inflight;
            row->path_bytes[i] = occupancy[i].inflight_bytes;
            if (occupancy[i].inflight > row->concurrency)
                row->concurrency = occupancy[i].inflight;
        }
    }
    pthread_mutex_unlock(&g_runtime_trace_mx);
    return index;
}

static void runtime_trace_finish(uint32_t index, int load_result,
                                 uint64_t ready_ns) {
    if (index == M3_SHD_M3_TRACE_INDEX_NONE
        || index >= M3_SHD_RUNTIME_MAX_TRACE) return;
    pthread_mutex_lock(&g_runtime_trace_mx);
    m3_shd_runtime_trace_row_t *row = &g_runtime.trace[index];
    if (row->used && !row->completed) {
        row->load_result = load_result;
        row->ready_ns = ready_ns;
        row->service_latency_ns = ready_ns >= row->issue_ns
            ? ready_ns - row->issue_ns : 0;
        if (load_result == 0 && row->path_complete
            && row->component_mask == M3_SHD_M3_COMPONENT_MASK_ALL
            && row->path_count > 0) {
            row->evidence_state = M3_SHD_EVIDENCE_OBSERVED;
        } else {
            row->evidence_state = M3_SHD_EVIDENCE_UNKNOWN;
        }
        row->completed = 1;
    }
    pthread_mutex_unlock(&g_runtime_trace_mx);
}

void m3_shd_m3_runtime_begin_load(m3_shd_m3_load_observation_t *observation) {
    if (observation) {
        *observation = (m3_shd_m3_load_observation_t){
            0, 0, M3_SHD_M3_TRACE_INDEX_NONE
        };
    }
}

int m3_shd_m3_runtime_issue(m3_shd_m3_load_observation_t *observation,
                            uint64_t model_id, ColiExpertKey key,
                            uint64_t bytes, int actual_replica,
                            int path_complete) {
    return m3_shd_m3_runtime_issue_mapped(
        observation, model_id, key, bytes, actual_replica, path_complete, 0);
}

int m3_shd_m3_runtime_issue_mapped(
    m3_shd_m3_load_observation_t *observation,
    uint64_t model_id, ColiExpertKey key, uint64_t bytes,
    int actual_replica, int path_complete, uint8_t component_mask) {
    if (!observation || observation->issued || !m3_shd_m3_runtime_enabled()
        || !g_runtime_context.valid || !g_runtime_context.model_id
        || g_runtime_context.model_id != model_id
        || !g_runtime_context.forward_id
        || !bytes || actual_replica < 0 || !path_complete
        || (size_t)actual_replica >= g_runtime.replica_count) return 0;
    m3_shd_resource_id_t drive =
        g_runtime.replica_drive_ids[(size_t)actual_replica];
    if (!drive) return 0;
    uint64_t request_id = atomic_fetch_add_explicit(&g_runtime_request_seq, 1,
                                                    memory_order_relaxed);
    if (!request_id || request_id == UINT64_MAX) return 0;
    m3_shd_m3_load_snapshot_t snapshot = {
        model_id, g_runtime_context.forward_id, g_runtime_context.generation,
        g_runtime_context.consumer_node, request_id,
        g_runtime_context.consumer_need_ns, key, bytes, drive, 1
    };
    m3_shd_request_t request;
    m3_shd_candidate_t actual;
    if (m3_shd_m3_make_request(&snapshot, &request, &actual) != 0
        || m3_shd_expand_candidate(g_runtime.planner, &actual) != 0) return 0;
    /* Shadow-only admission probe.  The dispatcher is deliberately untrusted
     * here and returns FALLBACK, preserving the established source and byte
     * sequence.  A future trusted activation must wire its ticket into the
     * real publication/completion path; this observer hook alone is not an
     * I/O publication barrier. */
    if (g_runtime.admission) {
        m3_adm_request_desc_t admission_desc = {0};
        admission_desc.request_id = request.request_id;
        admission_desc.generation = request.generation;
        admission_desc.topology_hash = g_runtime.topology.topology_hash;
        admission_desc.profile_hash = m3_shd_profile_hash(g_runtime.planner);
        admission_desc.selected_copy_id = actual.candidate_id;
        admission_desc.bytes = request.bytes;
        admission_desc.lease_id = request.request_id;
        admission_desc.lease_generation = request.generation;
        admission_desc.expected_parts = 1;
        admission_desc.demand = request.demand;
        admission_desc.path = actual;
        m3_adm_ticket_t ticket;
        m3_adm_admit_result_t admission = m3_adm_admit(
            g_runtime.admission, &admission_desc, &ticket);
        if (admission != M3_ADM_FALLBACK) {
            /* No shadow observation is recorded for an active/rejected
             * admission result.  The current production caller ignores this
             * probe result and continues its established load path; trusted
             * activation therefore requires a separate publication adapter. */
            if (admission == M3_ADM_RESERVED) {
                (void)m3_adm_cancel(g_runtime.admission, &ticket);
                (void)m3_adm_release_lease(g_runtime.admission, &ticket);
                (void)m3_adm_retire(g_runtime.admission, &ticket);
            }
            return 0;
        }
    }
    m3_shd_candidate_t candidates[M3_SHD_MAX_RESOURCES];
    size_t candidate_count = 0;
    if (!m3_shd_candidates_for_key(g_runtime.planner, model_id, key, bytes,
                                   drive, candidates,
                                   M3_SHD_MAX_RESOURCES, &candidate_count))
        return 0;
    m3_shd_decision_record_t decision;
    m3_shd_resource_occupancy_t occupancy[M3_SHD_M3_TRACE_PATH_MAX];
    uint64_t recorded_id = 0;
    if (m3_shd_decide_and_record_real_start(
            g_runtime.planner, &request, candidates, candidate_count,
            &actual, &decision, &recorded_id, occupancy,
            M3_SHD_M3_TRACE_PATH_MAX) == M3_SHD_UNKNOWN)
        return 0;
    observation->issued = 1;
    observation->request_id = recorded_id;
    request.request_id = recorded_id;
    observation->trace_index = runtime_trace_begin(
        &request, &actual, occupancy, &decision, m3_shd_m3_runtime_now_ns(),
        component_mask);
    return 1;
}

void m3_shd_m3_runtime_finish(m3_shd_m3_load_observation_t *observation,
                              int load_result) {
    if (!observation || !observation->issued
        || !m3_shd_m3_runtime_enabled()) return;
    if (load_result == 0)
        (void)m3_shd_record_real_completion(g_runtime.planner,
                                            observation->request_id);
    else
        (void)m3_shd_record_real_failure(g_runtime.planner,
                                          observation->request_id);
    runtime_trace_finish(observation->trace_index, load_result,
                         m3_shd_m3_runtime_now_ns());
    *observation = (m3_shd_m3_load_observation_t){0};
}

size_t m3_shd_m3_runtime_active_count(void) {
    return m3_shd_m3_runtime_enabled()
        ? m3_shd_active_count(g_runtime.planner) : 0;
}

int m3_shd_m3_runtime_decision_log_copy(char *dst, size_t cap,
                                        size_t *bytes_out) {
    if (!m3_shd_m3_runtime_enabled()) {
        if (bytes_out) *bytes_out = 0;
        if (dst && cap) dst[0] = '\0';
        return 0;
    }
    return m3_shd_decision_log_copy(g_runtime.planner, dst, cap, bytes_out);
}

int m3_shd_m3_runtime_trace_copy(char *dst, size_t cap,
                                 size_t *bytes_out) {
    if (!m3_shd_m3_runtime_enabled()) {
        if (bytes_out) *bytes_out = 0;
        if (dst && cap) dst[0] = '\0';
        return 0;
    }
    pthread_mutex_lock(&g_runtime_trace_mx);
    size_t required = 0;
    (void)runtime_trace_encode_locked(NULL, 0, &required);
    if (bytes_out) *bytes_out = required;
    int ok = dst && cap > required;
    if (ok) {
        (void)runtime_trace_encode_locked(dst, cap, &required);
    } else if (dst && cap) {
        dst[0] = '\0';
    }
    pthread_mutex_unlock(&g_runtime_trace_mx);
    return ok;
}
