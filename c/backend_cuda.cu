#include "backend_cuda.h"

#include "backend_gpu_compat.h"
#include "backend_cuda_dp4a.h"

/* Optional fmt=8 decode candidate (COLI_CUDA_F8_WARP=2): cuda_fp8.h maps
 * __nv_cvt_fp8_to_halfraw to an sm_89+ cvt instruction, with a bit-manip
 * fallback below 890. CUDA-only; the HIP build keeps the LUT decode. */
#if !(defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)) && defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define COLI_F8_HWCVT 1
#else
#define COLI_F8_HWCVT 0
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#ifdef COLI_CUPTI_TRACE
#include <cupti.h>
#include <cupti_activity.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef COLI_ANS
#include <dietgpu/ans/GpuANSCodec.h>
#include <dietgpu/utils/StackDeviceMemory.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <sys/stat.h>
#endif

struct RaggedKVEntry {
    const void *key;
    const float *host_l,*host_r;
    float **latent_pages,**rope_pages;
    int length,page_count,K,R;
};

#ifdef COLI_CUPTI_TRACE
/* Diagnostic-only CUPTI activity collector. It is compiled into a separate
 * DLL and remains dormant unless COLI_CUDA_CUPTI=1. The collector never adds
 * work to a CUDA stream: CUPTI receives completed activity buffers on its
 * worker thread, so this is suitable for joining against the host-side
 * resident overlap trace. */
struct ColiCuptiRecord {
    int kind;                         /* 1 kernel, 2 memcpy, 3 memcpy2, 4 API */
    uint64_t start, end, queued, submitted;
    uint32_t device, stream, correlation, cbid;
    uint64_t bytes;
    uint8_t copy_kind, src_kind, dst_kind;
    std::string name;
};

static std::mutex g_cupti_records_mu;
static std::vector<ColiCuptiRecord> g_cupti_records;
static size_t g_cupti_buffer_size = 8u << 20;
static int g_cupti_active = 0;
static uint64_t g_cupti_checkpoint_calls = 0;
static int64_t g_cupti_host_offset_ns = 0;
static std::string g_cupti_output_path;

static uint64_t cupti_trace_host_now_ns(void) {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()).count();
}

static void CUPTIAPI cupti_trace_buffer_requested(uint8_t **buffer,
                                                    size_t *size,
                                                    size_t *max_records) {
    void *p = std::malloc(g_cupti_buffer_size);
    if (!p) {
        *buffer = nullptr;
        *size = 0;
        *max_records = 0;
        return;
    }
    *buffer = (uint8_t *)p;
    *size = g_cupti_buffer_size;
    *max_records = 0;
}

static void cupti_trace_push(ColiCuptiRecord &&item) {
    std::lock_guard<std::mutex> lock(g_cupti_records_mu);
    g_cupti_records.push_back(std::move(item));
}

static void CUPTIAPI cupti_trace_buffer_completed(CUcontext, uint32_t stream_id,
                                                   uint8_t *buffer, size_t,
                                                   size_t valid_size) {
    if (buffer && valid_size) {
        uint8_t *cursor = buffer;
        CUpti_Activity *record = nullptr;
        while (cuptiActivityGetNextRecord(cursor, valid_size, &record) == CUPTI_SUCCESS) {
            if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                const CUpti_ActivityKernel9 *k =
                    reinterpret_cast<const CUpti_ActivityKernel9 *>(record);
                ColiCuptiRecord item{};
                item.kind = 1;
                item.start = k->start;
                item.end = k->end;
                item.queued = k->queued;
                item.submitted = k->submitted;
                item.device = k->deviceId;
                item.stream = k->streamId ? k->streamId : stream_id;
                item.correlation = k->correlationId;
                item.name = k->name ? k->name : "";
                cupti_trace_push(std::move(item));
            } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                const CUpti_ActivityMemcpy6 *m =
                    reinterpret_cast<const CUpti_ActivityMemcpy6 *>(record);
                ColiCuptiRecord item{};
                item.kind = 2;
                item.start = m->start;
                item.end = m->end;
                item.device = m->deviceId;
                item.stream = m->streamId ? m->streamId : stream_id;
                item.correlation = m->correlationId;
                item.bytes = m->bytes;
                item.copy_kind = m->copyKind;
                item.src_kind = m->srcKind;
                item.dst_kind = m->dstKind;
                cupti_trace_push(std::move(item));
            } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY2) {
                const CUpti_ActivityMemcpyPtoP4 *m =
                    reinterpret_cast<const CUpti_ActivityMemcpyPtoP4 *>(record);
                ColiCuptiRecord item{};
                item.kind = 3;
                item.start = m->start;
                item.end = m->end;
                item.device = m->deviceId;
                item.stream = m->streamId ? m->streamId : stream_id;
                item.correlation = m->correlationId;
                item.bytes = m->bytes;
                item.copy_kind = m->copyKind;
                item.src_kind = m->srcKind;
                item.dst_kind = m->dstKind;
                cupti_trace_push(std::move(item));
            } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME ||
                       record->kind == CUPTI_ACTIVITY_KIND_DRIVER) {
                const CUpti_ActivityAPI *a =
                    reinterpret_cast<const CUpti_ActivityAPI *>(record);
                ColiCuptiRecord item{};
                item.kind = record->kind == CUPTI_ACTIVITY_KIND_RUNTIME ? 4 : 5;
                item.start = a->start;
                item.end = a->end;
                item.correlation = a->correlationId;
                item.cbid = (uint32_t)a->cbid;
                cupti_trace_push(std::move(item));
            }
        }
    }
    std::free(buffer);
}

static int cupti_trace_result(CUptiResult rc, const char *what) {
    if (rc == CUPTI_SUCCESS) return 1;
    const char *name = nullptr;
    cuptiGetResultString(rc, &name);
    std::fprintf(stderr, "[CUDA CUPTI] %s failed: %s\n", what,
                 name ? name : "unknown");
    return 0;
}

static int cupti_trace_calibrate(void) {
    std::vector<int64_t> offsets;
    offsets.reserve(16);
    for (int i = 0; i < 16; i++) {
        uint64_t c0 = 0, c1 = 0;
        if (!cupti_trace_result(cuptiGetTimestamp(&c0), "timestamp calibration"))
            return 0;
        uint64_t h0 = cupti_trace_host_now_ns();
        uint64_t h1 = cupti_trace_host_now_ns();
        if (!cupti_trace_result(cuptiGetTimestamp(&c1), "timestamp calibration"))
            return 0;
        offsets.push_back((int64_t)((h0 + h1) / 2) - (int64_t)((c0 + c1) / 2));
    }
    std::sort(offsets.begin(), offsets.end());
    g_cupti_host_offset_ns = offsets[offsets.size() / 2];
    std::fprintf(stderr, "[CUDA CUPTI] active; host offset %lld ns\n",
                 (long long)g_cupti_host_offset_ns);
    return 1;
}

static void cupti_trace_start(void) {
    const char *enabled = std::getenv("COLI_CUDA_CUPTI");
    if (!enabled || *enabled != '1') return;
    {
        std::lock_guard<std::mutex> lock(g_cupti_records_mu);
        g_cupti_records.clear();
    }
    g_cupti_checkpoint_calls = 0;
    const char *path = std::getenv("COLI_CUDA_CUPTI_FILE");
    g_cupti_output_path = path && *path ? path : "cupti_qwen_activity.csv";
    if (!cupti_trace_result(cuptiActivityRegisterCallbacks(
                cupti_trace_buffer_requested, cupti_trace_buffer_completed),
            "register activity callbacks") ||
        !cupti_trace_result(cuptiActivityEnableLatencyTimestamps(1),
            "enable latency timestamps") ||
        !cupti_trace_result(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
            "enable concurrent kernels") ||
        !cupti_trace_result(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),
            "enable memcpy") ||
        !cupti_trace_result(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2),
            "enable peer memcpy") ||
        !cupti_trace_result(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME),
            "enable runtime API") ||
        !cupti_trace_result(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER),
            "enable driver API") ||
        !cupti_trace_calibrate()) {
        std::fprintf(stderr, "[CUDA CUPTI] disabled after initialization failure\n");
        return;
    }
    g_cupti_active = 1;
}

static void cupti_trace_write(void) {
    if (!g_cupti_active) return;
    cupti_trace_result(cuptiActivityFlushAll(1), "flush activities");
    std::vector<ColiCuptiRecord> records;
    {
        std::lock_guard<std::mutex> lock(g_cupti_records_mu);
        records = g_cupti_records;
    }
    std::sort(records.begin(), records.end(),
              [](const ColiCuptiRecord &a, const ColiCuptiRecord &b) {
                  if (a.start != b.start) return a.start < b.start;
                  return a.end < b.end;
              });
    std::ofstream out(g_cupti_output_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "[CUDA CUPTI] cannot write %s\n",
                     g_cupti_output_path.c_str());
        g_cupti_active = 0;
        return;
    }
    out << "record,kind,name,start_ns,end_ns,host_start_ns,host_end_ns,"
           "queued_ns,submitted_ns,host_queued_ns,host_submitted_ns,device,stream,correlation,cbid,bytes,"
           "copy_kind,src_kind,dst_kind\n";
    for (size_t i = 0; i < records.size(); i++) {
        const ColiCuptiRecord &r = records[i];
        const char *kind = r.kind == 1 ? "kernel" :
                           r.kind == 2 ? "memcpy" :
                           r.kind == 3 ? "memcpy2" :
                           r.kind == 4 ? "runtime" : "driver";
        out << i << ',' << kind << ",\"";
        for (char ch : r.name) {
            if (ch == '"') out << "\"\"";
            else out << ch;
        }
        out << '\"' << ',' << r.start << ',' << r.end << ','
            << (int64_t)r.start + g_cupti_host_offset_ns << ','
            << (int64_t)r.end + g_cupti_host_offset_ns << ','
            << r.queued << ',' << r.submitted << ','
            << (r.queued ? (int64_t)r.queued + g_cupti_host_offset_ns : 0) << ','
            << (r.submitted ? (int64_t)r.submitted + g_cupti_host_offset_ns : 0) << ','
            << r.device << ',' << r.stream << ',' << r.correlation << ',' << r.cbid << ','
            << r.bytes << ',' << (unsigned)r.copy_kind << ','
            << (unsigned)r.src_kind << ',' << (unsigned)r.dst_kind << '\n';
    }
    out.close();
    std::fprintf(stderr, "[CUDA CUPTI] wrote %zu records to %s\n",
                 records.size(), g_cupti_output_path.c_str());
}

static void cupti_trace_checkpoint(void) {
    if (!g_cupti_active) return;
    /* The local benchmark terminates the server with TerminateProcess after
     * receiving its response, so normal DLL shutdown is not guaranteed. Keep
     * a periodically flushed snapshot without putting a callback or query on
     * the CUDA stream. This is diagnostic-only and deliberately sparse. */
    if ((++g_cupti_checkpoint_calls & 63u) == 0)
        cupti_trace_write();
}

static void cupti_trace_stop(void) {
    if (!g_cupti_active) return;
    cupti_trace_write();
    (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY2);
    (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    (void)cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
    g_cupti_active = 0;
}
#else
static void cupti_trace_start(void) {}
static void cupti_trace_checkpoint(void) {}
static void cupti_trace_stop(void) {}
#endif

#ifndef COLI_KV_PAGE_TOKENS
#define COLI_KV_PAGE_TOKENS 64
#endif

static void ragged_kv_clear(RaggedKVEntry *e) {
    for (int i=0;i<e->page_count;i++) {
        if (e->latent_pages[i]) cudaFree(e->latent_pages[i]);
        if (e->rope_pages[i]) cudaFree(e->rope_pages[i]);
    }
    std::free(e->latent_pages);
    std::free(e->rope_pages);
    e->latent_pages=e->rope_pages=nullptr;
    e->length=e->page_count=0;
}

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int gs;                    /* quant group size; 0 = per-row scales (#334) */
    int ng;                    /* number of scale groups per row = ceil(I/gs) for fmt=4 */
    size_t scale_count;        /* floats in `scales`: O per-row, O*ng grouped */
    int tracked;
    int weights_owned;
    int scales_owned;
#ifdef COLI_ANS
    size_t archive_bytes;
    int compressed;
#endif
    RaggedKVEntry ragged[512];
    int ragged_count;
};

#ifdef COLI_ANS
struct AnsArenaChunk { uint8_t *p; size_t used,cap; };
#endif
typedef struct {
    int device;
    int compute_major,compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y,*host_kv; size_t host_x_cap,host_y_cap,host_kv_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *pipe_buf[32]; size_t pipe_cap[32];   /* scratch persistenti del resident pipeline */
    float *dn_buf[12]; size_t dn_cap[12];       /* full DeltaNet layer executor */
    cudaStream_t stream;
    cudaEvent_t ev_done; int ev_done_ok;        /* resident-group issue completion (#431 PR-C0) */
    cudaEvent_t resident_gpu_start, resident_gpu_end;
    cudaEvent_t resident_reduce_start, resident_reduce_end;
    int resident_timing_ok, resident_gpu_timing_pending, resident_reduce_timing_pending;
    uint64_t resident_gpu_host_lower_ns, resident_gpu_host_upper_ns;
    uint64_t resident_reduce_host_lower_ns, resident_reduce_host_upper_ns;
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
    int group_pending; size_t group_pending_bytes;   /* async expert-group in flight (Inc.4) */
    /* Per-phase timing events (set when COLI_CUDA_TIMING_DUMP is on). Owned
     * by the issue path; read and destroyed by the take path. */
    /* EV_COUNT_PHASE is 11: five start/end pairs plus TOTAL_POST.  Keep the
     * storage at the full count; the old [10] declaration let the final event
     * overwrite phase_ev_valid and caused timing-enabled runs to die later. */
    cudaEvent_t phase_ev[11];
    int phase_ev_valid;
    /* DP4A scratch: per-call device pointer arrays for the kernel arguments.
     * Critical: kernel arguments that are arrays of device pointers must live
     * in DEVICE memory; passing a host-stack array is undefined behavior. */
    void *d_gw_ptrs, *d_uw_ptrs, *d_dw_ptrs;
    void *d_gsc_ptrs, *d_usc_ptrs, *d_dsc_ptrs;
    size_t d_ptrs_cap;
    ColiCudaDp4aExpertMeta *d_dp4a_meta;
    size_t d_dp4a_meta_cap, d_dp4a_meta_count;
    int d_dp4a_meta_valid;
    uint64_t d_dp4a_meta_updates, d_dp4a_meta_skips;
    ColiCudaDp4aExpertMeta h_dp4a_meta[64];
    /* Optional CUDA Graph cache for the resident DP4A execution envelope.
     * The graph contains the existing kernels and device copies only; dynamic
     * metadata/weights are refreshed before replay.  A graph is keyed by the
     * fixed decode geometry and destination pointers below. */
    cudaGraph_t resident_graph[65];
    cudaGraphExec_t resident_graph_exec[65];
    int resident_graph_valid[65];
    int resident_graph_D, resident_graph_I;
    uint64_t resident_graph_captures, resident_graph_launches;
    uint64_t resident_graph_fallbacks;
#ifdef COLI_ANS
    void *ans_raw; size_t ans_raw_cap;
    void *ans_host; size_t ans_host_cap;
    int ans_copy_pending;
    dietgpu::StackDeviceMemory *ans_scratch;
    std::vector<AnsArenaChunk> *ans_chunks;
#endif
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
    int ggs,ugs,dgs;      /* per-tensor quant group size; 0 = per-row scales (#334 fmt=4) */
} GroupDesc;

/* Per-phase timing event slot indices (used by COLI_CUDA_TIMING_DUMP). Must
 * stay in sync with the phase_ev[] array in DeviceContext (11 slots). */
enum { EV_H2D_PRE = 0, EV_H2D_POST,
       EV_K0_PRE, EV_K0_POST,
       EV_K1_PRE, EV_K1_POST,
       EV_K2_PRE, EV_K2_POST,
       EV_D2H_PRE, EV_D2H_POST,
       EV_TOTAL_POST, EV_COUNT_PHASE };

/* A phase-event array has a single owner at a time: the issue path owns its
 * local array until it transfers the handles to DeviceContext, then take (or
 * shutdown) owns the context array.  Keep destruction and clearing together;
 * this makes error paths idempotent and prevents a stale handle from being
 * destroyed again after a partial timing failure. */
static void phase_events_destroy(DeviceContext *ctx) {
    if (!ctx) return;
    for (int i = 0; i < EV_COUNT_PHASE; i++) {
        if (ctx->phase_ev[i]) {
            (void)cudaEventDestroy(ctx->phase_ev[i]);
            ctx->phase_ev[i] = nullptr;
        }
    }
    ctx->phase_ev_valid = 0;
}

/* Resident-island timing is separate from the older async-group phase dump.
 * It is opt-in with COLI_TIMERS=1 and has no effect on the normal path. */
static int resident_timing_enabled(void) {
    const char *e = std::getenv("COLI_TIMERS");
    return e && *e == '1';
}
static int resident_timing_ensure(DeviceContext *ctx) {
    if (!ctx || !resident_timing_enabled()) return 0;
    if (ctx->resident_timing_ok) return 1;
    cudaEvent_t *ev[] = { &ctx->resident_gpu_start, &ctx->resident_gpu_end,
                          &ctx->resident_reduce_start, &ctx->resident_reduce_end };
    int made = 0;
    for (int i = 0; i < 4; i++) {
        if (cudaEventCreate(ev[i]) != cudaSuccess) {
            for (int j = 0; j < made; j++) cudaEventDestroy(*ev[j]);
            ctx->resident_gpu_start = ctx->resident_gpu_end = nullptr;
            ctx->resident_reduce_start = ctx->resident_reduce_end = nullptr;
            return 0;
        }
        made++;
    }
    ctx->resident_timing_ok = 1;
    return 1;
}
static void resident_timing_destroy(DeviceContext *ctx) {
    if (!ctx) return;
    if (ctx->resident_gpu_start) cudaEventDestroy(ctx->resident_gpu_start);
    if (ctx->resident_gpu_end) cudaEventDestroy(ctx->resident_gpu_end);
    if (ctx->resident_reduce_start) cudaEventDestroy(ctx->resident_reduce_start);
    if (ctx->resident_reduce_end) cudaEventDestroy(ctx->resident_reduce_end);
    ctx->resident_gpu_start = ctx->resident_gpu_end = nullptr;
    ctx->resident_reduce_start = ctx->resident_reduce_end = nullptr;
    ctx->resident_timing_ok = 0;
    ctx->resident_gpu_timing_pending = 0;
    ctx->resident_reduce_timing_pending = 0;
    ctx->resident_gpu_host_lower_ns = ctx->resident_gpu_host_upper_ns = 0;
    ctx->resident_reduce_host_lower_ns = ctx->resident_reduce_host_upper_ns = 0;
}

static int resident_graph_enabled(void) {
    const char *e = std::getenv("COLI_CUDA_GRAPH");
    return e && *e == '1';
}

static void resident_graph_destroy(DeviceContext *ctx) {
    if (!ctx) return;
    for (int i = 0; i < 65; i++) {
        if (ctx->resident_graph_exec[i]) {
            (void)cudaGraphExecDestroy(ctx->resident_graph_exec[i]);
            ctx->resident_graph_exec[i] = nullptr;
        }
        if (ctx->resident_graph[i]) {
            (void)cudaGraphDestroy(ctx->resident_graph[i]);
            ctx->resident_graph[i] = nullptr;
        }
        ctx->resident_graph_valid[i] = 0;
    }
    ctx->resident_graph_D = ctx->resident_graph_I = 0;
}

static void resident_graph_slot_destroy(DeviceContext *ctx, int count) {
    if (!ctx || count < 1 || count > 64) return;
    if (ctx->resident_graph_exec[count]) {
        (void)cudaGraphExecDestroy(ctx->resident_graph_exec[count]);
        ctx->resident_graph_exec[count] = nullptr;
    }
    if (ctx->resident_graph[count]) {
        (void)cudaGraphDestroy(ctx->resident_graph[count]);
        ctx->resident_graph[count] = nullptr;
    }
    ctx->resident_graph_valid[count] = 0;
}

/* CUDA event timestamps are on a device clock and cannot be directly
 * subtracted from the host monotonic timestamps used by qwen36.c.  The
 * timeline mode therefore records a host-clock lower bound at issue and an
 * upper bound after the existing stream synchronization.  It does not query
 * or callback from the CUDA stream: both mechanisms materially perturb this
 * small Pascal workload and would invalidate the overlap measurement. */
static uint64_t resident_host_clock_ns(void) {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
static int resident_timeline_enabled(void) {
    const char *e = std::getenv("COLI_CUDA_TIMELINE");
    return e && *e == '1';
}

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static uint64_t g_device_group_calls[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_experts[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_rows[COLI_CUDA_MAX_DEVICES];
static double g_device_group_h2d_ms[COLI_CUDA_MAX_DEVICES];
static double g_device_group_kernel_ms[COLI_CUDA_MAX_DEVICES];
static double g_device_group_d2h_ms[COLI_CUDA_MAX_DEVICES];
static std::mutex g_group_stats_mu;
#ifdef COLI_ANS
static FILE *g_ans_sidecar;
static int g_ans_sidecar_pack;
#if defined(__linux__)
static int g_ans_direct_fd=-1;
static off_t g_ans_direct_off;
#endif
static uint64_t g_ans_load_records;
static double g_ans_header_s,g_ans_read_s,g_ans_stage_s,g_ans_enqueue_s;
static int g_ans_profile_printed;
static double ans_now_s(){
    using clock=std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
#endif

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    (void)cudaGetLastError();   /* consume the sticky error: a failed call must
                                   not poison the next launch's error check */
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!cuda_ok(cudaSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

/* fmt=6 (E8/IQ3) geometry, mirroring quant.h. A super-block packs 256 weights
 * into 98 bytes: 64 codebook indices, 8 words of (4x7 signs + 4-bit sub-scale),
 * and one fp16 super-scale. Scales live INSIDE the block, so fmt=6 tensors carry
 * no separate scale array (#452). */
#define COLI_E8_QK      256
#define COLI_E8_SUB      32
#define COLI_E8_BBYTES   98

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2 || fmt == 4) return (size_t)(I + 1) / 2;   /* fmt=4: same packed int4 */
    if (fmt == 3) return (size_t)(I + 3) / 4;
    if (fmt == 4) return (size_t)(I + 1) / 2;   /* grouped int4: nibbles like fmt 2 */
    if (fmt == 7) return (size_t)(I + 1) / 2;   /* MXFP4: e2m1 nibbles, 2 per byte */
    if (fmt == 6) return (size_t)(((int64_t)I + COLI_E8_QK - 1) / COLI_E8_QK) * COLI_E8_BBYTES;
    if (fmt == 8) return (size_t)I;             /* fp8-e4m3: raw bytes, layout of fmt=1 */
    return 0;
}

/* The E8 codebook, uploaded once per device from quant.h's e8_grid so the table
 * has a single source of truth and cannot drift from the CPU decoder. */
__constant__ uint8_t c_e8_grid[256][4];

/* The fmt=8 e4m3 decode table, uploaded once per device from quant.h's
 * E4M3_LUT — same single-source-of-truth arrangement as c_e8_grid. Uploads of
 * fmt=8 tensors are refused until it is published (g_fp8_lut_ready): a kernel
 * reading the zero-initialized table would compute silent zeros, the exact
 * failure mode this format's dispatch work exists to prevent. */
__constant__ float c_e4m3[256];
static int g_fp8_lut_ready;

/* A super-block is 98 bytes, so nothing inside it is guaranteed 4- or 2-byte
 * aligned: assemble the words byte-wise instead of dereferencing. */
__device__ __forceinline__ uint32_t e8_ld_u32(const uint8_t *p){
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
/* Mirrors e8_fp16_to_f32 rather than calling __half2float, so the two decoders
 * cannot disagree on subnormals. */
__device__ __forceinline__ float e8_fp16(const uint8_t *p){
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1]<<8);
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1F, man=h&0x3FF, bits;
    if (!exp)         bits = man ? (sign|((127u-15u+1u-1u)<<23)|(man<<13)) : sign;
    else if (exp==31) bits = sign|0x7F800000u|(man<<13);
    else              bits = sign|((exp+112u)<<23)|(man<<13);
    float f; memcpy(&f,&bits,4); return f;
}
/* Expand one 32-weight sub-block; mirrors e8_expand_sub in quant.h. */
__device__ __forceinline__ void e8_expand_sub_dev(const uint8_t *blk, int ib, float d, float *out){
    uint32_t word = e8_ld_u32(blk + COLI_E8_QK/4 + ib*4);
    float db = d * (0.5f + (float)((word>>28)&0xF)) * 0.5f;
    const uint8_t *idx = blk + ib*8;
    for (int l=0;l<4;l++){
        uint32_t seven=(word>>(7*l))&0x7F;
        const uint8_t *g0=c_e8_grid[idx[l*2+0]], *g1=c_e8_grid[idx[l*2+1]];
        int par=0;
        for (int j=0;j<8;j++){
            int neg = j<7 ? (int)((seven>>j)&1) : 0;
            if (j<7) par^=neg; else neg=par;        /* odd parity closes the block */
            float mag = (j<4 ? (float)g0[j] : (float)g1[j-4]) * 0.5f;
            out[l*8+j] = neg ? -mag*db : mag*db;
        }
    }
}

/* ---- MXFP4 (OCP microscaling FP4), fmt=7 -----------------------------------
 * Same layout the CPU path decodes in quant.h's matmul_mxfp4, and the same two
 * tricks, so the two agree bit for bit:
 *
 *   packed [O, I/2]  u8 — e2m1 nibbles, LOW nibble = even column, bit3 = sign,
 *                         bits 0..2 index {0,.5,1,1.5,2,3,4,6}
 *   scales [O, I/32] u8 — ue8m0 exponent per 32-column group, w = v * 2^(s-127)
 *
 * The exponent is decoded as a bit pattern rather than exp2f: (uint32)s << 23
 * reinterpreted as float IS 2^(s-127) for s in [1,254], and reproduces the CPU
 * path's documented edge behaviour exactly -- s=0 gives +0 and s=255 gives +inf
 * on both sides. Using exp2f here would agree for the normal range and diverge
 * at the ends, which is precisely where a silent mismatch would hide.
 *
 * The LUT holds DOUBLED values so every entry is an exact small integer; the
 * compensating 0.5f rides along in mx4_scale_dev, as it does on the CPU. */
/* Decoded arithmetically rather than from a __constant__ table: a file-scope
 * __constant__ array with static linkage is initialised per translation unit,
 * and this kernel is also compiled into the HIP build and the DLL, where that
 * silently yields garbage. The magnitude is 2^(exp-1) * 0.5 for exp in 1..3 and
 * 0 for exp 0, which is exactly the OCP e2m1 table {0,.5,1,1.5,2,3,4,6}. */
__device__ static inline float mx4_decode(int n) {
    int mant = n & 1, exp = (n >> 1) & 3;
    float mag = exp ? ldexpf(1.0f + 0.5f * (float)mant, exp - 1) : 0.5f * (float)mant;
    return (n & 8) ? -mag : mag;
}

__device__ static inline float mx4_scale_dev(uint8_t s) {
    union { uint32_t u; float f; } b;
    b.u = static_cast<uint32_t>(s) << 23;
    return b.f;
}

/* e2m1 nibble at column i of a packed row. */
__device__ static inline float mx4_weight_at(const uint8_t *q, int i) {
    uint8_t v = q[i >> 1];
    return mx4_decode((i & 1) ? (v >> 4) : (v & 15));
}

/* Generic per-element weight decode for the kernels that need one (the absorb
 * kernels and the generic grouped-expert path). fmt=3 (int2) is now an EXPLICIT
 * branch and the fall-through is a refusal.
 *
 * It used to be the other way round: int2 was the fall-through, so every format
 * this function does not decode -- fmt=5 (int3-g64), fmt=6 (E8/IQ3), fmt=8
 * (fp8-e4m3), and anything added later -- was read as 2-bit values and returned
 * numbers. Meanwhile the CPU functions doing the same job on the same tensor,
 * qt_addrow and qt_matvec_rows (colibri.c), both exit(1) naming the function and
 * the fmt. Two backends, identical unsupported input, one refusing and one
 * fabricating: that asymmetry is the defect, independent of any particular
 * format's arrival.
 *
 * WHY __trap() AND NOT A DIAGNOSTIC. This is device code inside a running
 * kernel; there is no stderr to name the tensor on and no way to unwind. __trap
 * aborts the kernel and poisons the context, so the next cuda_ok() call on the
 * host reports a failure instead of the caller consuming fabricated values --
 * the same "stop rather than misread" outcome as the CPU's exit(1), reached the
 * only way device code can reach it.
 *
 * IT IS A BACKSTOP, NOT THE PRIMARY GATE -- and the gates above it are several
 * DIFFERENT checks, not one. Naming them exactly, because "every launcher checks
 * coli_cuda_weight_at_supported" would be false and a reader will verify it:
 *   - UPLOAD is the widest gate. coli_cuda_tensor_upload{,_g} refuse when
 *     row_bytes(fmt,I) == 0, which is every fmt this file has no row stride for --
 *     fmt=5 (int3-g64), every negative fmt, every unknown fmt. Those can never
 *     become a ColiCudaTensor at all, so the uploadable set is {0,1,2,3,4,6,7,8}.
 *   - quant_matmul dispatches 6, 7, 4 and 8 in explicit branches of its own, so
 *     the generic else that calls this function sees only 0/1/2/3 out of that set.
 *   - the generic grouped-expert path refuses gf>3 || uf>3 || df>3 on the host.
 *   - the absorb call sites (the other caller of this function) are the ones
 *     gated by coli_cuda_weight_at_supported, via absorb_fmt_ok below -- one
 *     caller, not "every launcher".
 * Each of those is sufficient on its own for the sites it covers; this trap
 * exists because none of them is a property of THIS function. Reaching this line
 * means a launch site got past its own gate, which is a bug in that gate. */
__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2 || fmt == 4) {                               /* fmt=4: same nibble layout */
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    if (fmt == 3) {                                           /* int2 */
        uint8_t v = q[i >> 2];
        return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
    }
    __trap();
    return 0.0f;   /* not reached: __trap() does not return */
}

/* Scale for output `row`, input element `k`. fmt=4 (grouped int4) stores ng
 * scales per row at scales[row*ng + k/gs]; every other quantized format has
 * one scale per row at scales[row]. Mirrors quant_matmul's fmt==4 branch so the
 * attention absorb kernels apply per-group scales instead of the per-row
 * (fmt=2) semantic that crashed #298's g64 kv_b. */
__device__ static float absorb_scale(const float *wscale, int fmt, int gs, int ng, int row, int k) {
    if (!fmt) return 1.f;
    if (fmt != 4) return wscale[row];
    int g = k / gs; if (g >= ng) g = ng - 1;   /* tail of the last (partial) group */
    return wscale[(size_t)row * ng + g];
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

/* Convert a batch of separately allocated execution-cache weight tensors in
 * one launch. The pointer table itself is only a temporary device-side
 * argument; the tensors remain owned by the caller/cache. */
__global__ static void offset_to_signed_s4_ptrs(uint8_t *const *ptrs,
                                                size_t bytes,int count){
    size_t ix=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t total=bytes*(size_t)count;
    if(ix<total) ptrs[ix/bytes][ix%bytes]^=0x88;
}

/* ---- fmt=8 (fp8-e4m3) warp decode/accumulate helpers -----------------------
 * The fmt=8 dot products are compared against the CPU reference (quant.h
 * matmul_fp8) by the cross-tier parity tests, so this TU must never be built
 * with fast-math: nvcc's --use_fast_math implies -ftz=true, which flushes the
 * scale*subnormal contributions the denormal-scale test pins down. (-ftz has
 * no macro of its own to test; the Makefile pins -ftz=false on the nvcc line.) */
#if defined(__USE_FAST_MATH__) || defined(__FAST_MATH__)
#error "backend_cuda.cu: fmt=8 kernels forbid fast-math builds (FTZ breaks CPU parity)"
#endif

/* Fixed-order 5-shuffle reduce over a 32-lane logical warp. Width pinned to 32
 * so a HIP wave64 device does not fold two output rows into one reduction. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define f8_shfl_down(v,o) __shfl_down((v),(o),32)
#else
#define f8_shfl_down(v,o) __shfl_down_sync(0xffffffffu,(v),(o),32)
#endif

/* Decode one e4m3 byte. Default: a __shared__ copy of c_e4m3 — per-thread
 * data-dependent indices are the __constant__ cache's documented worst case
 * (accesses to different addresses within a warp serialize), while shared
 * memory takes them at full rate. HW=1 (COLI_CUDA_F8_WARP=2, CUDA-only): the
 * cuda_fp8.h route. Every finite e4m3 value is exactly representable in f16,
 * so the two paths should agree bit for bit — but the on-device 256-value
 * sweep is the authority, and the LUT stays the default until the sweep
 * certifies the cvt path on the target silicon. */
template<int HW>
__device__ __forceinline__ float f8_dec(const float *slut, uint8_t b){
#if COLI_F8_HWCVT
    if (HW) return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
#endif
    return slut[b];
}

/* One 128-column scale block, one warp: 32 lanes x 4 bytes is the exact
 * FP8_BLOCK fit. Weights are decoded once and reused across ns activation
 * rows (weights are the traffic; activations come from cache), then each
 * row's per-lane products are tree-reduced in fixed order — the block partial
 * lands in LANE 0 of gp[]/up_[] (other lanes hold partials). vec picks the
 * LOAD width only (uchar4/float4 on aligned full blocks, guarded bytes on
 * tails and unaligned bases — e.g. zero-copy views: correct, slower); both
 * load paths feed ONE shared product/accumulate sequence below, and the
 * vec/byte parity test asserts their bit-equality on identical data. Tail
 * padding contributes +0.0f terms, which can at most flip a -0.0 partial to
 * +0.0. DUAL folds a second weight row (gate+up) against the same
 * activations without re-reading x. The per-row arithmetic never depends on
 * ns or s, so S-tiling cannot reorder a dot (B5). */
template<int HW,int DUAL>
__device__ __forceinline__ void f8w_block(const float *slut,const uint8_t *gr,
        const uint8_t *ur,const float *xs,int xstride,int ns,int base,int len,
        int vec,float *gp,float *up_){
    int lane=threadIdx.x&31,i0=base+lane*4,full=vec&&len==128;
    float gw[4],uw[4]={0,0,0,0};
    if(full){
        uchar4 q=*(const uchar4*)(gr+i0);
        gw[0]=f8_dec<HW>(slut,q.x);gw[1]=f8_dec<HW>(slut,q.y);
        gw[2]=f8_dec<HW>(slut,q.z);gw[3]=f8_dec<HW>(slut,q.w);
        if(DUAL){uchar4 r=*(const uchar4*)(ur+i0);
            uw[0]=f8_dec<HW>(slut,r.x);uw[1]=f8_dec<HW>(slut,r.y);
            uw[2]=f8_dec<HW>(slut,r.z);uw[3]=f8_dec<HW>(slut,r.w);}
    }else for(int k=0;k<4;k++){
        int in=i0+k<base+len;
        gw[k]=in?f8_dec<HW>(slut,gr[i0+k]):0.f;
        if(DUAL)uw[k]=in?f8_dec<HW>(slut,ur[i0+k]):0.f;
    }
    for(int s=0;s<ns;s++){
        const float *xr=xs+(size_t)s*xstride;
        float xv[4];
        if(full){
            float4 xf=*(const float4*)(xr+i0);
            xv[0]=xf.x;xv[1]=xf.y;xv[2]=xf.z;xv[3]=xf.w;
        }else for(int k=0;k<4;k++) xv[k]=i0+k<base+len?xr[i0+k]:0.f;
        float g=0,u=0;
        for(int k=0;k<4;k++){g+=xv[k]*gw[k];if(DUAL)u+=xv[k]*uw[k];}
        for(int off=16;off;off>>=1){g+=f8_shfl_down(g,off);if(DUAL)u+=f8_shfl_down(u,off);}
        gp[s]=g;if(DUAL)up_[s]=u;
    }
}

/* Dense fmt=8 rework: quant_matmul's fmt=8 branch as its OWN kernel, so
 * COLI_CUDA_F8_WARP=0 restores the fully original dense behavior too (the
 * host picks in quant_matmul_launch). Same grid/block contract as
 * quant_matmul: one 256-thread block per (o,s). Accumulation mirrors the CPU
 * reference: f32 partial per 128-block (f8w_block), the scale applied ONCE
 * per partial, double across blocks. Warps stride the block axis; the
 * cross-warp double sum runs in fixed warp order, so the reduction order is
 * a pure function of the dims. Decode reads a shared copy of c_e4m3
 * (data-dependent __constant__ indices serialize); the cvt candidate stays
 * grouped-path-only until certified. NaN bytes decode to NaN and propagate,
 * same policy as the CPU path. */
__global__ static void quant_matmul_f8w(float *y,const float *x,const void *weights,
                                        const float *scales,int S,int I,int O){
    int o=blockIdx.x,s=blockIdx.y;
    const float *xs=x+(size_t)s*I;
    __shared__ float slut[256];
    __shared__ double dsum[32];
    for(int i=threadIdx.x;i<256;i+=blockDim.x) slut[i]=c_e4m3[i];
    __syncthreads();
    const uint8_t *wrow=(const uint8_t*)weights+(size_t)o*I;
    const float *scl=scales+(size_t)(o>>7)*(size_t)((I+127)>>7);
    int warp=threadIdx.x>>5,nw=blockDim.x>>5,nblk=(I+127)>>7;
    int vec=!(I&3)&&!((size_t)wrow&3)&&!((size_t)xs&15);
    double a=0;
    for(int bi=warp;bi<nblk;bi+=nw){
        int base=bi<<7,len=I-base<128?I-base:128;
        float p;
        f8w_block<0,0>(slut,wrow,nullptr,xs,0,1,base,len,vec,&p,nullptr);
        if(!(threadIdx.x&31)) a+=(double)p*scl[bi];
    }
    if(!(threadIdx.x&31)) dsum[warp]=a;
    __syncthreads();
    if(!threadIdx.x){
        for(int w=1;w<nw;w++) a+=dsum[w];
        y[(size_t)s*O+o]=(float)a;
    }
    (void)S;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb, int gs, int ng) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    if (fmt == 6) {
        /* E8/IQ3: decode is per 32-weight sub-block, so threads stride over
         * sub-blocks rather than elements -- expanding once per 32 weights
         * instead of redoing the word/parity work for every element. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        int nsub = (I + COLI_E8_SUB - 1) / COLI_E8_SUB;
        for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
            const uint8_t *blk = wrow + (size_t)(sb / (COLI_E8_QK/COLI_E8_SUB)) * COLI_E8_BBYTES;
            float w[COLI_E8_SUB];
            e8_expand_sub_dev(blk, sb % (COLI_E8_QK/COLI_E8_SUB), e8_fp16(blk+96), w);
            int off = sb*COLI_E8_SUB, n = I-off < COLI_E8_SUB ? I-off : COLI_E8_SUB;
            for (int k=0;k<n;k++) sum += xs[off+k]*w[k];
        }
    } else if (fmt == 7) {
        /* MXFP4: one ue8m0 exponent per 32 columns. The SCALAR CPU reference
         * (quant.h matmul_mxfp4's scalar loop -- its AVX2 sibling applies the
         * scale per 8-lane FMA instead and disagrees with it at s=255)
         * accumulates each 32-group UNSCALED and multiplies the group
         * subtotal by the scale once. For scales in [0,254] -- every value a
         * real checkpoint contains -- the scale is a finite power of two,
         * scaling each element or the subtotal is exact either way, so
         * threads stride over columns exactly as before and outputs stay
         * bit-identical with prior builds. s=255 decodes to +inf (the
         * documented edge), and there the application point is visible:
         * per-element scaling turns every zero product into 0*inf = NaN,
         * where the scalar reference's finite-subtotal-times-inf keeps the
         * group's sign. Rows carrying a 255 scale byte therefore take the
         * per-group path mirroring that scalar loop; the row's +-inf/NaN
         * classification does not depend on summation order unless several
         * large-finite (s~254) group results overflow only in aggregate --
         * out-of-domain either way -- so the tree reduce below needs no
         * change. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const uint8_t *scl = reinterpret_cast<const uint8_t *>(scales) + (size_t)o * ng;
        __shared__ int scale_inf;
        if (!threadIdx.x) scale_inf = 0;
        __syncthreads();
        for (int g = threadIdx.x; g < ng; g += blockDim.x)
            if (scl[g] == 255) scale_inf = 1;
        __syncthreads();
        if (!scale_inf) {
            for (int i = threadIdx.x; i < I; i += blockDim.x) {
                int g = i >> 5;
                if (g >= ng) g = ng - 1;
                sum += xs[i] * mx4_weight_at(wrow, i) * mx4_scale_dev(scl[g]);
            }
        } else {
            for (int g = threadIdx.x; g < ng; g += blockDim.x) {
                /* same element->group mapping as the clamp above: the last
                 * group takes every remaining column, a group past the data
                 * contributes nothing */
                int base = g << 5, end = base + 32;
                if (g == ng - 1 || end > I) end = I;
                if (base >= end) continue;
                float ga = 0.0f;
                for (int i = base; i < end; i++)
                    ga += xs[i] * mx4_weight_at(wrow, i);
                sum += ga * mx4_scale_dev(scl[g]);
            }
        }
    } else if (fmt == 4) {
        /* Grouped int4: one f32 scale per gs elements along I (ng groups per row).
         * Scale layout: scales[o*ng + g]. Each thread strides through I, applying
         * the appropriate group scale as it crosses group boundaries. This matches
         * the CPU matmul_i4_grouped accumulation exactly. */
        const float *scl = scales + (size_t)o * ng;
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            int g = i / gs;
            if (g >= ng) g = ng - 1;  /* tail elements in the last (partial) group */
            sum += xs[i] * weight_at(weights, fmt, row, i) * scl[g];
        }
    } else if (fmt == 8) {
        /* fp8-e4m3 (matmul_fp8): one byte per weight (layout of fmt=1), one f32
         * scale per 128x128 BLOCK of [O,I] — scales[(o/128)*ceil(I/128) + i/128].
         * The block edge is a fixed property of the format (FP8_BLOCK), so the
         * geometry derives from I alone and gs/ng are ignored: this branch is
         * correct no matter which call site launched it. NaN bytes decode to NaN
         * through the LUT and propagate, same policy as the CPU path. This is
         * the ORIGINAL dense path, kept for COLI_CUDA_F8_WARP=0; the default
         * routes fmt=8 to quant_matmul_f8w instead (quant_matmul_launch). */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const float *scl = scales + (size_t)(o >> 7) * (size_t)((I + 127) >> 7);
        for (int i = threadIdx.x; i < I; i += blockDim.x)
            sum += xs[i] * c_e4m3[wrow[i]] * scl[i >> 7];
    } else {
        for (int i = threadIdx.x; i < I; i += blockDim.x)
            sum += xs[i] * weight_at(weights, fmt, row, i);
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        /* fmt 4/6/7/8 already applied their scaling inside the loop: 4 and 7 are
         * per-group (one scale per gs / per 32 columns), 6 carries it in the
         * block header, 8 reads a per-128 block scale alongside the weights.
         * Only the per-row formats get the trailing multiply --
         * and for fmt=7 `scales` points at ue8m0 BYTES, so reading it as float
         * here does not merely double-scale, it reads garbage. */
        y[(size_t)s * O + o] = (fmt && fmt != 4 && fmt != 6 && fmt != 7 && fmt != 8) ? partial[0] * scales[o] : partial[0];
}

/* Dense-int8 decode GEMV with the same four 8-lane accumulation streams as
 * qwen36.c's AVX2/FMA matmul_q().  This is not a new arithmetic format: it is
 * a semantics-preserving execution variant used by the coarse dense batch.
 * One warp owns one output row, and the final scalar reduction mirrors the
 * CPU vector reduction.  Qwen dense dimensions are 32-aligned; unusual tails
 * stay on quant_matmul(). */
__global__ static void dense_i8_matmul_cpu_order(float *y,const float *x,
                                                const int8_t *weights,
                                                const float *scales,int I,int O){
    int o=blockIdx.x, lane=threadIdx.x;
    if(o>=O || lane>=32) return;
    const int8_t *w=weights+(size_t)o*I;
    int start=(lane&7)+((lane>>3)*8);
    float acc=0.f;
    for(int i=start;i<I;i+=32)
        acc=fmaf(x[i],(float)w[i],acc);
    __shared__ float p[32];
    p[lane]=acc; __syncthreads();
    if(lane==0){
        float b0=(p[0]+p[8])+(p[16]+p[24]);
        float b1=(p[1]+p[9])+(p[17]+p[25]);
        float b2=(p[2]+p[10])+(p[18]+p[26]);
        float b3=(p[3]+p[11])+(p[19]+p[27]);
        float b4=(p[4]+p[12])+(p[20]+p[28]);
        float b5=(p[5]+p[13])+(p[21]+p[29]);
        float b6=(p[6]+p[14])+(p[22]+p[30]);
        float b7=(p[7]+p[15])+(p[23]+p[31]);
        float s0=(b0+b4)+(b2+b6);
        float s1=(b1+b5)+(b3+b7);
        y[o]=(s0+s1)*scales[o];
    }
}

/* One-thread-per-output form of the same Qwen dense-int8 reduction contract.
 *
 * The warp-per-output kernel above mirrors the CPU tree, but it still pays for
 * 32 CUDA threads and a shared-memory rendezvous for every output row.  That
 * is a poor execution shape for decode GEMV: the useful work is small, while
 * the launch contains thousands of mostly idle lanes.  Keep the four logical
 * eight-lane FMA streams explicitly in one thread instead.  __fmaf_rn and
 * __fadd_rn make the rounding points explicit, and the final additions are in
 * the same order as matmul_q()'s AVX2 reduction.  This is an execution-shape
 * change, not a new quantization or arithmetic scheme.
 *
 * It is selected only for 32-aligned inputs.  The historical kernel remains
 * the fallback for unusual tails and for an A/B comparison. */
__global__ static void dense_i8_matmul_exact(float *y,const float *x,
                                             const int8_t *weights,
                                             const float *scales,int I,int O){
    int o=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=O) return;
    const int8_t *w=weights+(size_t)o*I;
    float p0=0.f,p1=0.f,p2=0.f,p3=0.f,p4=0.f,p5=0.f,p6=0.f,p7=0.f;
    float p8=0.f,p9=0.f,p10=0.f,p11=0.f,p12=0.f,p13=0.f,p14=0.f,p15=0.f;
    float p16=0.f,p17=0.f,p18=0.f,p19=0.f,p20=0.f,p21=0.f,p22=0.f,p23=0.f;
    float p24=0.f,p25=0.f,p26=0.f,p27=0.f,p28=0.f,p29=0.f,p30=0.f,p31=0.f;
    for(int i=0;i<I;i+=32){
        p0 =__fmaf_rn(x[i+0], (float)w[i+0], p0);
        p1 =__fmaf_rn(x[i+1], (float)w[i+1], p1);
        p2 =__fmaf_rn(x[i+2], (float)w[i+2], p2);
        p3 =__fmaf_rn(x[i+3], (float)w[i+3], p3);
        p4 =__fmaf_rn(x[i+4], (float)w[i+4], p4);
        p5 =__fmaf_rn(x[i+5], (float)w[i+5], p5);
        p6 =__fmaf_rn(x[i+6], (float)w[i+6], p6);
        p7 =__fmaf_rn(x[i+7], (float)w[i+7], p7);
        p8 =__fmaf_rn(x[i+8], (float)w[i+8], p8);
        p9 =__fmaf_rn(x[i+9], (float)w[i+9], p9);
        p10=__fmaf_rn(x[i+10],(float)w[i+10],p10);
        p11=__fmaf_rn(x[i+11],(float)w[i+11],p11);
        p12=__fmaf_rn(x[i+12],(float)w[i+12],p12);
        p13=__fmaf_rn(x[i+13],(float)w[i+13],p13);
        p14=__fmaf_rn(x[i+14],(float)w[i+14],p14);
        p15=__fmaf_rn(x[i+15],(float)w[i+15],p15);
        p16=__fmaf_rn(x[i+16],(float)w[i+16],p16);
        p17=__fmaf_rn(x[i+17],(float)w[i+17],p17);
        p18=__fmaf_rn(x[i+18],(float)w[i+18],p18);
        p19=__fmaf_rn(x[i+19],(float)w[i+19],p19);
        p20=__fmaf_rn(x[i+20],(float)w[i+20],p20);
        p21=__fmaf_rn(x[i+21],(float)w[i+21],p21);
        p22=__fmaf_rn(x[i+22],(float)w[i+22],p22);
        p23=__fmaf_rn(x[i+23],(float)w[i+23],p23);
        p24=__fmaf_rn(x[i+24],(float)w[i+24],p24);
        p25=__fmaf_rn(x[i+25],(float)w[i+25],p25);
        p26=__fmaf_rn(x[i+26],(float)w[i+26],p26);
        p27=__fmaf_rn(x[i+27],(float)w[i+27],p27);
        p28=__fmaf_rn(x[i+28],(float)w[i+28],p28);
        p29=__fmaf_rn(x[i+29],(float)w[i+29],p29);
        p30=__fmaf_rn(x[i+30],(float)w[i+30],p30);
        p31=__fmaf_rn(x[i+31],(float)w[i+31],p31);
    }
    float b0=__fadd_rn(__fadd_rn(p0,p8), __fadd_rn(p16,p24));
    float b1=__fadd_rn(__fadd_rn(p1,p9), __fadd_rn(p17,p25));
    float b2=__fadd_rn(__fadd_rn(p2,p10),__fadd_rn(p18,p26));
    float b3=__fadd_rn(__fadd_rn(p3,p11),__fadd_rn(p19,p27));
    float b4=__fadd_rn(__fadd_rn(p4,p12),__fadd_rn(p20,p28));
    float b5=__fadd_rn(__fadd_rn(p5,p13),__fadd_rn(p21,p29));
    float b6=__fadd_rn(__fadd_rn(p6,p14),__fadd_rn(p22,p30));
    float b7=__fadd_rn(__fadd_rn(p7,p15),__fadd_rn(p23,p31));
    float s0=__fadd_rn(__fadd_rn(b0,b4),__fadd_rn(b2,b6));
    float s1=__fadd_rn(__fadd_rn(b1,b5),__fadd_rn(b3,b7));
    y[o]=__fmul_rn(__fadd_rn(s0,s1),scales[o]);
}

/* Full DeltaNet island kernels. These deliberately keep the scalar loop
 * ordering of qwen36.c for the recurrent state; the point of this prototype is
 * to remove host boundaries, not to invent a different reduction tree. */
__global__ static void dn_f32_ba_kernel(float *b,float *a,const float *bw,
                                        const float *aw,const float *x,
                                        int hidden,int vheads){
    int h=blockIdx.x*blockDim.x+threadIdx.x;
    if(h>=vheads) return;
    const float *br=bw+(size_t)h*hidden, *ar=aw+(size_t)h*hidden;
    float sb=0.f, sa=0.f;
    for(int i=0;i<hidden;i++){
        sb += x[i]*br[i];
        sa += x[i]*ar[i];
    }
    b[h]=sb; a[h]=sa;
}

__global__ static void dn_conv_kernel(float *conv_out,float *ring,
                                      const float *conv,const float *qkv,
                                      int conv_dim,int convk){
    for(int cc=blockIdx.x*blockDim.x+threadIdx.x;cc<conv_dim;cc+=gridDim.x*blockDim.x){
        const float *w=conv+(size_t)cc*convk;
        float *rg=ring+(size_t)cc*(convk-1);
        float acc=0.f;
        for(int kk=0;kk<convk-1;kk++) acc += w[kk]*rg[kk];
        acc += w[convk-1]*qkv[cc];
        conv_out[cc]=acc/(1.f+expf(-acc));
        for(int kk=0;kk<convk-2;kk++) rg[kk]=rg[kk+1];
        rg[convk-2]=qkv[cc];
    }
}

__device__ static float dn_softplus(float z){
    return z>20.f ? z : log1pf(expf(z));
}

__global__ static void dn_state_kernel(float *outv,float *q,float *k,
                                       float *kv_all,float *delta_all,
                                       float *rec,const float *conv_out,
                                       const float *b,const float *a,
                                       const float *dtbias,const float *alog,
                                       int vheads,int kheads,int kdim,int vdim,
                                       int conv_dim){
    int h=blockIdx.x;
    if(h>=vheads) return;
    int rep=vheads/kheads, vk_idx=h/rep, key_dim_tot=kheads*kdim;
    float *qd=q+(size_t)h*kdim, *kd=k+(size_t)h*kdim;
    const float *qin=conv_out+(size_t)vk_idx*kdim;
    const float *kin=conv_out+key_dim_tot+(size_t)vk_idx*kdim;
    /* The old implementation used one thread for the complete head.  Keep
     * the two norm reductions serial (and therefore numerically identical),
     * but let independent vector elements execute in parallel. */
    for(int d=threadIdx.x;d<kdim;d+=blockDim.x){ qd[d]=qin[d]; kd[d]=kin[d]; }
    __shared__ double nq_s, nk_s;
    __shared__ float beta_s, egh_s;
    __syncthreads();
    if(threadIdx.x==0){
        double sq=1e-6;
        for(int d=0;d<kdim;d++) sq+=(double)qd[d]*qd[d];
        nq_s=sqrt(sq);
        double sk=1e-6;
        for(int d=0;d<kdim;d++) sk+=(double)kd[d]*kd[d];
        nk_s=sqrt(sk);
        beta_s=1.f/(1.f+expf(-b[h]));
        float gg=-expf(alog[h])*dn_softplus(a[h]+dtbias[h]);
        egh_s=expf(gg);
    }
    __syncthreads();
    float scale=1.f/sqrtf((float)kdim);
    for(int d=threadIdx.x;d<kdim;d+=blockDim.x){
        qd[d]=(float)((double)qd[d]/(double)nq_s*scale);
        kd[d]=(float)((double)kd[d]/(double)nk_s);
    }
    __syncthreads();
    float beta=beta_s, egh=egh_s;
    float *Sh=rec+(size_t)h*kdim*vdim;
    float *kvl=kv_all+(size_t)h*vdim, *dl=delta_all+(size_t)h*vdim;
    const float *vd=conv_out+2*key_dim_tot+(size_t)h*vdim;
    float *ov=outv+(size_t)h*vdim;
    for(int vv=threadIdx.x;vv<vdim;vv+=blockDim.x){
        /* Every loop over kk is still serial for one output element, so the
         * FP32 accumulation order remains the same as the scalar kernel. */
        for(int kk=0;kk<kdim;kk++) Sh[(size_t)kk*vdim+vv]*=egh;
        float kvv=0.f;
        for(int kk=0;kk<kdim;kk++) kvv+=kd[kk]*Sh[(size_t)kk*vdim+vv];
        kvl[vv]=kvv;
        float dlv=(vd[vv]-kvv)*beta;
        dl[vv]=dlv;
        for(int kk=0;kk<kdim;kk++)
            Sh[(size_t)kk*vdim+vv]+=kd[kk]*dlv;
        float ovv=0.f;
        for(int kk=0;kk<kdim;kk++) ovv+=qd[kk]*Sh[(size_t)kk*vdim+vv];
        ov[vv]=ovv;
    }
}

/* Reference state kernel for the coarse-layer A/B.  It is intentionally kept
 * as the exact one-thread-per-head implementation used before the occupancy
 * experiment, so a state-kernel-only comparison does not depend on the host
 * control path or on the norm kernel. */
__global__ static void dn_state_kernel_scalar(float *outv,float *q,float *k,
                                              float *kv_all,float *delta_all,
                                              float *rec,const float *conv_out,
                                              const float *b,const float *a,
                                              const float *dtbias,const float *alog,
                                              int vheads,int kheads,int kdim,int vdim,
                                              int conv_dim){
    int h=blockIdx.x;
    if(h>=vheads || threadIdx.x) return;
    int rep=vheads/kheads, vk_idx=h/rep, key_dim_tot=kheads*kdim;
    float *qd=q+(size_t)h*kdim, *kd=k+(size_t)h*kdim;
    const float *qin=conv_out+(size_t)vk_idx*kdim;
    const float *kin=conv_out+key_dim_tot+(size_t)vk_idx*kdim;
    for(int d=0;d<kdim;d++){ qd[d]=qin[d]; kd[d]=kin[d]; }
    double sq=1e-6;
    for(int d=0;d<kdim;d++) sq+=(double)qd[d]*qd[d];
    double nq=sqrt(sq);
    float scale=1.f/sqrtf((float)kdim);
    for(int d=0;d<kdim;d++) qd[d]=(float)((double)qd[d]/nq*scale);
    double sk=1e-6;
    for(int d=0;d<kdim;d++) sk+=(double)kd[d]*kd[d];
    double nk=sqrt(sk);
    for(int d=0;d<kdim;d++) kd[d]=(float)((double)kd[d]/nk);
    float beta=1.f/(1.f+expf(-b[h]));
    float gg=-expf(alog[h])*dn_softplus(a[h]+dtbias[h]);
    float egh=expf(gg);
    float *Sh=rec+(size_t)h*kdim*vdim;
    for(int t=0;t<kdim*vdim;t++) Sh[t]*=egh;
    float *kvl=kv_all+(size_t)h*vdim, *dl=delta_all+(size_t)h*vdim;
    const float *vd=conv_out+2*key_dim_tot+(size_t)h*vdim;
    for(int vv=0;vv<vdim;vv++) kvl[vv]=0.f;
    for(int kk=0;kk<kdim;kk++){
        float kkd=kd[kk]; const float *Sr=Sh+(size_t)kk*vdim;
        for(int vv=0;vv<vdim;vv++) kvl[vv]+=kkd*Sr[vv];
    }
    for(int vv=0;vv<vdim;vv++) dl[vv]=(vd[vv]-kvl[vv])*beta;
    for(int kk=0;kk<kdim;kk++){
        float kkd=kd[kk]; float *Sr=Sh+(size_t)kk*vdim;
        for(int vv=0;vv<vdim;vv++) Sr[vv]+=kkd*dl[vv];
    }
    float *ov=outv+(size_t)h*vdim; const float *qd0=qd;
    for(int vv=0;vv<vdim;vv++) ov[vv]=0.f;
    for(int kk=0;kk<kdim;kk++){
        float qkd=qd0[kk]; const float *Sr=Sh+(size_t)kk*vdim;
        for(int vv=0;vv<vdim;vv++) ov[vv]+=qkd*Sr[vv];
    }
}

__global__ static void dn_norm_kernel(float *outr,const float *outv,
                                      const float *z,const float *norm,
                                      int vheads,int vdim,float eps){
    int h=blockIdx.x;
    if(h>=vheads) return;
    const float *o=outv+(size_t)h*vdim, *zr=z+(size_t)h*vdim;
    __shared__ float r_s;
    if(threadIdx.x==0){
        double ms=0.;
        for(int d=0;d<vdim;d++) ms+=(double)o[d]*o[d];
        r_s=1.f/sqrtf((float)(ms/vdim)+eps);
    }
    __syncthreads();
    float r=r_s;
    float *dst=outr+(size_t)h*vdim;
    for(int d=threadIdx.x;d<vdim;d+=blockDim.x){
        float val=o[d]*r*norm[d];
        dst[d]=val*zr[d]/(1.f+expf(-zr[d]));
    }
}

__global__ static void dn_norm_kernel_scalar(float *outr,const float *outv,
                                             const float *z,const float *norm,
                                             int vheads,int vdim,float eps){
    int h=blockIdx.x;
    if(h>=vheads || threadIdx.x) return;
    const float *o=outv+(size_t)h*vdim, *zr=z+(size_t)h*vdim;
    double ms=0.;
    for(int d=0;d<vdim;d++) ms+=(double)o[d]*o[d];
    float r=1.f/sqrtf((float)(ms/vdim)+eps);
    float *dst=outr+(size_t)h*vdim;
    for(int d=0;d<vdim;d++){
        float val=o[d]*r*norm[d];
        dst[d]=val*zr[d]/(1.f+expf(-zr[d]));
    }
}

/* fmt=6 activation rotation, y = Q^T x for Q = D*H/sqrt(n) (#452). One block per
 * row; the power-of-two block is staged in shared memory, capping n at 4096
 * floats -- which covers every block GLM produces (6144 -> 2048+4096, 1536 ->
 * 512+1024). The sign stream is regenerated in-kernel from the same xorshift64*
 * that quant.h's e8_signs uses, so no rotation data is stored or uploaded.
 *
 * Placement note: all routed experts of a layer share one gate/up input, so that
 * rotation belongs to the CALLER (once per layer). This kernel exists for the
 * down projection, whose input is the per-expert silu(gate)*up product and so
 * cannot be shared -- mirroring colibri.c's split at moe(). */
__global__ static void e8_rot_rows_kernel(float *rows, int dim, int off, int n){
    extern __shared__ float sh[];
    __shared__ uint8_t sbits[4096/8];
    if (!threadIdx.x) {
        uint64_t s = 417u + (uint64_t)n;
        for (int i=0;i<(n+7)/8;i++){
            s^=s>>12; s^=s<<25; s^=s>>27;
            sbits[i] = (uint8_t)((s*2685821657736338717ULL)>>56);
        }
    }
    __syncthreads();
    float *row = rows + (size_t)blockIdx.x*dim + off;
    for (int i=threadIdx.x;i<n;i+=blockDim.x){
        float v=row[i];
        sh[i] = (sbits[i>>3]>>(i&7)&1) ? -v : v;
    }
    __syncthreads();
    for (int len=1;len<n;len<<=1){
        for (int j=threadIdx.x;j<n/2;j+=blockDim.x){
            int i = (j/len)*(len<<1) + (j%len);
            float u=sh[i], v=sh[i+len];
            sh[i]=u+v; sh[i+len]=u-v;
        }
        __syncthreads();
    }
    float sc=rsqrtf((float)n);
    for (int i=threadIdx.x;i<n;i+=blockDim.x) row[i]=sh[i]*sc;
}

/* Rotate nr rows in place, tiling non-power-of-two dims block-diagonally exactly
 * as e8_rot_rows does. Returns 0 if a block exceeds the shared-memory cap. */
static int e8_rot_rows_dev(float *rows, int nr, int dim, cudaStream_t stream){
    int off = 0;
    while (off < dim) {
        int rem = dim-off, b = rem & (-rem);
        while (b > 4096) b >>= 1;
        e8_rot_rows_kernel<<<(unsigned)nr, 256, (size_t)b*sizeof(float), stream>>>(rows, dim, off, b);
        if (cudaGetLastError() != cudaSuccess) return 0;
        off += b;
    }
    return 1;
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* Four warps share one A tile and compute 16x64 outputs.  This matters for
 * prefill: the first prototype reloaded/converter A once per 16 output cols. */
__global__ static void w4a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);           /* [Ntile,Ktile] == B col-major */
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* Gate and up use the same input.  Eight warps compute both 16x64 projections
 * while sharing the FP32->FP16 conversion of A. */
__global__ static void w4a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;size_t rb=(size_t)(K+1)/2;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    __syncwarp();   /* i due fratelli sopra la portano; stessa lettura cross-lane di out[warp]
                     * subito sotto -> stesso requisito di visibilita' (racecheck: 3 hazard qui).
                     * EN: the two sibling kernels carry this; same cross-lane read follows. */
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

/* Native fmt=6 expert groups.  One block owns one (expert,row,output) and
 * expands each 32-weight E8 sub-block once for both gate/up projections. */
__global__ static void grouped_hidden_e8_dual(float *gate,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    size_t rb=row_bytes(6,D);
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*rb;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    int nsub=(D+COLI_E8_SUB-1)/COLI_E8_SUB;
    for(int sb=threadIdx.x;sb<nsub;sb+=blockDim.x){
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *gb=gr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        const uint8_t *ub=ur+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float gw[COLI_E8_SUB],uw[COLI_E8_SUB];
        e8_expand_sub_dev(gb,ib,e8_fp16(gb+96),gw);
        e8_expand_sub_dev(ub,ib,e8_fp16(ub+96),uw);
        int off=sb*COLI_E8_SUB,n=D-off<COLI_E8_SUB?D-off:COLI_E8_SUB;
        for(int k=0;k<n;k++){float v=xs[off+k];ga+=v*gw[k];ua+=v*uw[k];}
    }
    __shared__ float gp[256],up[256];gp[threadIdx.x]=ga;up[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];up[threadIdx.x]+=up[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){float g=gp[0];gate[(size_t)(d.offset+s)*I+o]=(g/(1.f+expf(-g)))*up[0];}
}

__global__ static void grouped_down_e8(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    size_t rb=row_bytes(6,I);const uint8_t *wr=(const uint8_t*)d.d+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    int nsub=(I+COLI_E8_SUB-1)/COLI_E8_SUB;
    for(int sb=threadIdx.x;sb<nsub;sb+=blockDim.x){
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *blk=wr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float w[COLI_E8_SUB];e8_expand_sub_dev(blk,ib,e8_fp16(blk+96),w);
        int off=sb*COLI_E8_SUB,n=I-off<COLI_E8_SUB?I-off:COLI_E8_SUB;
        for(int k=0;k<n;k++)sum+=xs[off+k]*w[k];
    }
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* Fused epilogue: silu(gate)*up lands here instead of a third kernel —
     * the exact silu_mul expression on the exact same inputs, so bit-identical,
     * and the up[] round-trip through global memory disappears. up stays a
     * param so the launch sites keep their signature. */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0]*d.gs[o],u=upv[0]*d.us[o];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

/* fmt=4 grouped-int4 variants (#334): identical structure to the w4 kernels,
 * but the scale varies along the input dimension — sc[o*ng + i/gs], applied
 * per element inside the accumulation (gs is even, so a packed byte never
 * straddles a group). gs<=0 degrades to per-row (ng=1), so mixed fmt2/fmt4
 * groups run correctly through this one kernel family. */
__global__ static void grouped_hidden_g4_dual(float *gate,float *up,const float *x,
                                              const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    int ggs=d.ggs>0?d.ggs:D, ugs=d.ugs>0?d.ugs:D;
    const float *gsc=d.gs+(size_t)o*(size_t)((D+ggs-1)/ggs);
    const float *usc=d.us+(size_t)o*(size_t)((D+ugs-1)/ugs);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;float gv=gsc[i/ggs],uv=usc[i/ugs];
        ga+=xs[i]*g0*gv;ua+=xs[i]*u0*uv;
        if(i+1<D){ga+=xs[i+1]*g1*gv;ua+=xs[i+1]*u1*uv;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* same epilogue fusion as the w4 dual above (per-group scales already
     * applied inside the accumulation, so silu runs on the raw sums) */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}
__global__ static void grouped_down_g4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    int dgs=d.dgs>0?d.dgs:I;
    const float *dsc=d.ds+(size_t)o*(size_t)((I+dgs-1)/dgs);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;float sv=dsc[i/dgs];
        sum+=xs[i]*a*sv;if(i+1<I)sum+=xs[i+1]*z*sv;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

/* fmt=8 fp8-e4m3 variants: same structure as the g4 kernels, but one byte per
 * weight (decoded through c_e4m3) and the scale is per 128x128 BLOCK of the
 * member's [O,I] matrix — sc[(o/128)*ceil(I/128) + i/128]. The block edge is a
 * property of the format, so the geometry derives from the dims alone; these
 * kernels require every member to be fmt=8 (no ride-along: a per-row member's
 * scales are [O], which this indexing would read out of bounds). */
__global__ static void grouped_hidden_f8_dual(float *gate,float *up,const float *x,
                                              const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*D;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*D;
    int nblk=(D+127)>>7;
    const float *gsc=d.gs+(size_t)(o>>7)*nblk;
    const float *usc=d.us+(size_t)(o>>7)*nblk;
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x){float xv=xs[i];int b=i>>7;
        ga+=xv*c_e4m3[gr[i]]*gsc[b];ua+=xv*c_e4m3[ur[i]]*usc[b];}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* same fused epilogue as the w4/g4 duals: scales applied in the
     * accumulation, silu(gate)*up lands in gate[], up[] is never written */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}
__global__ static void grouped_down_f8(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*I;
    int nblk=(I+127)>>7;
    const float *dsc=d.ds+(size_t)(o>>7)*nblk;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x)sum+=xs[i]*c_e4m3[row[i]]*dsc[i>>7];
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

/* fmt=8 warp rework (COLI_CUDA_F8_WARP, default on): one WARP owns one
 * (expert,output-row) pair and walks the row block-major through f8w_block —
 * weights stream once per tile of up to 4 activation rows instead of once per
 * (o,s) 256-thread block, and the accumulation is the CPU reference's (f32
 * block partial, scale once per block, double across blocks) instead of the
 * old flat per-element-scale f32 sum. Same descriptor surface as the old
 * kernels; grid (ceil(O/8), count): 8 warps per 256-thread block, rows looped
 * in-kernel. The old kernels stay compiled and selectable (=0) as the field
 * escape hatch. */
template<int HW>
__global__ static void grouped_hidden_f8w_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    __shared__ float slut[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)slut[i]=c_e4m3[i];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int o=blockIdx.x*(blockDim.x>>5)+warp,c=blockIdx.y;GroupDesc d=desc[c];
    if(o>=I)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*D;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*D;
    int nblk=(D+127)>>7;
    const float *gsc=d.gs+(size_t)(o>>7)*nblk;
    const float *usc=d.us+(size_t)(o>>7)*nblk;
    int vec=!(D&3)&&!(((size_t)gr|(size_t)ur)&3)&&!((size_t)x&15);
    for(int s0=0;s0<d.rows;s0+=4){
        int ns=d.rows-s0<4?d.rows-s0:4;
        const float *xs=x+(size_t)(d.offset+s0)*D;
        double ga[4]={0,0,0,0},ua[4]={0,0,0,0};
        for(int bi=0;bi<nblk;bi++){
            int base=bi<<7,len=D-base<128?D-base:128;
            float gp[4],up_[4];
            f8w_block<HW,1>(slut,gr,ur,xs,D,ns,base,len,vec,gp,up_);
            if(!lane){float sg=gsc[bi],su=usc[bi];
                for(int s=0;s<ns;s++){ga[s]+=(double)gp[s]*sg;ua[s]+=(double)up_[s]*su;}}
        }
        /* same fused epilogue as the old dual: silu(gate)*up lands in gate[],
         * up[] is never written */
        if(!lane)for(int s=0;s<ns;s++){
            float g=(float)ga[s],u=(float)ua[s];
            gate[(size_t)(d.offset+s0+s)*I+o]=(g/(1.f+expf(-g)))*u;
        }
    }
    (void)up;
}
template<int HW>
__global__ static void grouped_down_f8w(float *y,const float *x,const GroupDesc *desc,int D,int I){
    __shared__ float slut[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)slut[i]=c_e4m3[i];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int o=blockIdx.x*(blockDim.x>>5)+warp,c=blockIdx.y;GroupDesc d=desc[c];
    if(o>=D)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*I;
    int nblk=(I+127)>>7;
    const float *dsc=d.ds+(size_t)(o>>7)*nblk;
    int vec=!(I&3)&&!((size_t)row&3)&&!((size_t)x&15);
    for(int s0=0;s0<d.rows;s0+=4){
        int ns=d.rows-s0<4?d.rows-s0:4;
        const float *xs=x+(size_t)(d.offset+s0)*I;
        double a[4]={0,0,0,0};
        for(int bi=0;bi<nblk;bi++){
            int base=bi<<7,len=I-base<128?I-base:128;
            float p[4];
            f8w_block<HW,0>(slut,row,nullptr,xs,I,ns,base,len,vec,p,nullptr);
            if(!lane){float sd=dsc[bi];
                for(int s=0;s<ns;s++)a[s]+=(double)p[s]*sd;}
        }
        if(!lane)for(int s=0;s<ns;s++)
            y[(size_t)(d.offset+s0+s)*D+o]=(float)a[s];
    }
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale,
                                                int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

__global__ static void attention_absorb_batch_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale,
        int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K;
        const float *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    __syncthreads(); /* every warp must read red[0] above before red[] is reused below */
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

/* Independent device-resident KV sequence per row. lengths selects the valid
 * prefix; latent/rope point at paged caches updated by the host wrapper. */
__global__ static void attention_absorb_ragged_kernel(float *ctx,const float *q,
        const float *const *latent,const float *const *rope,const int *lengths,
        const void *weights,const float *wscale,int fmt,int S,int H,int Q,int R,
        int V,int K,int T,int page_stride,float scale,int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=lengths[s],rbase=h*(Q+V);
    if(s>=S||nt<1||nt>T)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;int pg=t/COLI_KV_PAGE_TOKENS,pt=t%COLI_KV_PAGE_TOKENS;
        const float *lt=latent[(size_t)s*page_stride+pg]+(size_t)pt*K;
        const float *rt=rope[(size_t)s*page_stride+pg]+(size_t)pt*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    __syncthreads(); /* every warp must read red[0] above before red[] is reused below */
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++){
        const float *lt=latent[(size_t)s*page_stride+t/COLI_KV_PAGE_TOKENS]+
                        (size_t)(t%COLI_KV_PAGE_TOKENS)*K;
        a+=scores[t]*lt[k];}cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*
            absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

__global__ static void ragged_kv_append(float *const *latent,float *const *rope,
        const float *packed,const int *old_len,const int *add,const int *offset,
        int K,int R,int page_stride){
    int s=blockIdx.x,n=add[s],base=offset[s];
    for(int t=0;t<n;t++){
        int pos=old_len[s]+t,pg=pos/COLI_KV_PAGE_TOKENS,pt=pos%COLI_KV_PAGE_TOKENS;
        float *lp=latent[(size_t)s*page_stride+pg]+(size_t)pt*K;
        float *rp=rope[(size_t)s*page_stride+pg]+(size_t)pt*R;
        for(int k=threadIdx.x;k<K;k+=blockDim.x)lp[k]=packed[base+(size_t)t*K+k];
        for(int r=threadIdx.x;r<R;r+=blockDim.x)rp[r]=packed[base+(size_t)n*K+(size_t)t*R+r];
    }
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

#ifdef COLI_ANS
static void *ans_arena_alloc(DeviceContext *ctx,size_t bytes){
    bytes=(bytes+255)&~size_t(255);
    if(!ctx->ans_chunks)ctx->ans_chunks=new std::vector<AnsArenaChunk>;
    if(ctx->ans_chunks->empty()||ctx->ans_chunks->back().cap-ctx->ans_chunks->back().used<bytes){
        size_t cap=256ull<<20;if(cap<bytes)cap=bytes;
        uint8_t *p=nullptr;if(!cuda_ok(cudaMalloc(&p,cap),"ANS arena chunk"))return nullptr;
        ctx->ans_chunks->push_back({p,0,cap});
    }
    AnsArenaChunk &c=ctx->ans_chunks->back();void *p=c.p+c.used;c.used+=bytes;return p;
}
static int ans_host_reserve(DeviceContext *ctx,size_t bytes){
    if(ctx->ans_copy_pending){
        if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"ANS sidecar upload synchronize"))return 0;
        ctx->ans_copy_pending=0;
    }
    if(ctx->ans_host_cap>=bytes)return 1;
    if(ctx->ans_host)cudaFreeHost(ctx->ans_host);
    ctx->ans_host=nullptr;ctx->ans_host_cap=0;
    if(!cuda_ok(cudaMallocHost(&ctx->ans_host,bytes),"ANS pinned staging allocation"))return 0;
    ctx->ans_host_cap=bytes;return 1;
}
static int prepare_group_weights(DeviceContext *ctx,
        ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,
        ColiCudaTensor *const *downs,int count,GroupDesc *host){
    int n=0; size_t total=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *q[3]={gates[c],ups[c],downs[c]};
        for(int k=0;k<3;k++) if(q[k]->compressed){n++;total+=q[k]->weight_bytes;}
    }
    if(!n) return 1;
    if(!g_ans_profile_printed&&std::getenv("COLI_ANS_PROFILE")){
        g_ans_profile_printed=1;
        std::fprintf(stderr,
            "[ANS] load profile: %llu records | header %.2fs | read %.2fs | "
            "staging/alloc %.2fs | enqueue %.2fs\n",
            (unsigned long long)g_ans_load_records,g_ans_header_s,g_ans_read_s,
            g_ans_stage_s,g_ans_enqueue_s);
    }
    if(!ctx->ans_scratch||!reserve_bytes(&ctx->ans_raw,&ctx->ans_raw_cap,total)) return 0;
    std::vector<const void*> in; in.reserve(n);
    std::vector<void*> out; out.reserve(n);
    std::vector<uint32_t> cap; cap.reserve(n);
    size_t off=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *q[3]={gates[c],ups[c],downs[c]};
        const void **dst[3]={&host[c].g,&host[c].u,&host[c].d};
        for(int k=0;k<3;k++) if(q[k]->compressed){
            void *raw=(uint8_t*)ctx->ans_raw+off;
            in.push_back(q[k]->weights);out.push_back(raw);cap.push_back((uint32_t)q[k]->weight_bytes);
            *dst[k]=raw;off+=q[k]->weight_bytes;
        }
    }
    dietgpu::ANSCodecConfig config(11,false);
    dietgpu::ansDecodeBatchPointer(*ctx->ans_scratch,config,(uint32_t)n,in.data(),out.data(),
                                   cap.data(),nullptr,nullptr,ctx->stream);
    return cuda_ok(cudaGetLastError(),"ANS expert decode launch");
}
#else
static int prepare_group_weights(DeviceContext *,ColiCudaTensor *const *,
        ColiCudaTensor *const *,ColiCudaTensor *const *,int,GroupDesc *){return 1;}
#endif

/* Publish quant.h's E8 codebook to every configured device. __constant__ memory
 * is per-device, so this walks the contexts; the engine calls it once after init
 * rather than the backend carrying a second copy of the table that could drift
 * from the CPU decoder's (#452). Safe to call before any fmt=6 upload only. */
extern "C" int coli_cuda_e8_set_grid(const void *grid) {
    if (!grid || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(cudaMemcpyToSymbol(c_e8_grid, grid, sizeof(c_e8_grid)), "E8 codebook upload"))
            return 0;
    }
    return 1;
}

/* Publish quant.h's E4M3_LUT the same way — one source of truth for the fmt=8
 * decode on CPU and GPU. Until this succeeds, fmt=8 uploads are refused. */
extern "C" int coli_cuda_fp8_set_lut(const float *lut) {
    if (!lut || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(cudaMemcpyToSymbol(c_e4m3, lut, sizeof(c_e4m3)), "e4m3 LUT upload"))
            return 0;
    }
    g_fp8_lut_ready = 1;
    return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    /* #509: the ROCm runtime (comgr, MIOpen, roctracer) reads $TEMP as a temp-dir
     * path. A stray numeric TEMP (the engine's legacy sampling alias) makes comgr's
     * lazy init fail inside the first stream create -- SIGSEGV in the error-unwind
     * on gfx1100, clean hipErrorOutOfMemory on gfx1030. The engine has already
     * parsed g_temp by the time we get here, so a TEMP that is not a real directory
     * is safe to drop before the first ROCm call; a genuine temp-dir is preserved. */
    {
        const char *t = std::getenv("TEMP");
        struct stat st;
        /* Same test on both hosts; only the CRT spelling differs. The MSVC CRT
         * (Windows hipcc's host pass) has no S_ISDIR and no unsetenv — it spells
         * the directory bit _S_IFDIR/_S_IFMT and clears a variable by assigning
         * an empty value. This is an OS/CRT difference, NOT a vendor one, so it
         * stays a _WIN32 branch and adds no CUDA-vs-HIP conditional. */
#ifdef _WIN32
        if (t && *t && (stat(t, &st) != 0 ||
                        (st.st_mode & _S_IFMT) != _S_IFDIR)) _putenv_s("TEMP", "");
#else
        if (t && *t && (stat(t, &st) != 0 || !S_ISDIR(st.st_mode))) unsetenv("TEMP");
#endif
    }
#endif
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    cupti_trace_start();
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        ctx->compute_major=prop.major;ctx->compute_minor=prop.minor;
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
#ifdef COLI_ANS
        if(std::getenv("CUDA_RAW_EXPERTS")){
            ctx->ans_scratch = new dietgpu::StackDeviceMemory(device, 1ull << 30);
            ctx->ans_chunks = new std::vector<AnsArenaChunk>;
        }
#endif
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" int coli_cuda_available_device_count(void) {
    int available = 0;
    if (cudaGetDeviceCount(&available) != cudaSuccess) return 0;
    return available;
}

extern "C" void coli_cuda_shutdown(void) {
    /* Flush before tearing down CUDA contexts so the final resident kernels,
     * peer copies, and host API records are present in the diagnostic file. */
    cupti_trace_stop();
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        resident_graph_destroy(ctx);
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        for(int b=0;b<32;b++) if(ctx->pipe_buf[b]) cudaFree(ctx->pipe_buf[b]);
        for(int b=0;b<12;b++) if(ctx->dn_buf[b]) cudaFree(ctx->dn_buf[b]);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->host_kv) cudaFreeHost(ctx->host_kv);
        phase_events_destroy(ctx);
        resident_timing_destroy(ctx);
        if (ctx->d_gw_ptrs)  cudaFree(ctx->d_gw_ptrs);
        if (ctx->d_uw_ptrs)  cudaFree(ctx->d_uw_ptrs);
        if (ctx->d_dw_ptrs)  cudaFree(ctx->d_dw_ptrs);
        if (ctx->d_gsc_ptrs) cudaFree(ctx->d_gsc_ptrs);
        if (ctx->d_usc_ptrs) cudaFree(ctx->d_usc_ptrs);
        if (ctx->d_dsc_ptrs) cudaFree(ctx->d_dsc_ptrs);
        if (ctx->d_dp4a_meta) cudaFree(ctx->d_dp4a_meta);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
#ifdef COLI_ANS
        if(ctx->ans_copy_pending)cudaStreamSynchronize(ctx->stream);
        if(ctx->ans_host)cudaFreeHost(ctx->ans_host);
        if (ctx->ans_raw) cudaFree(ctx->ans_raw);
        if(ctx->ans_chunks){for(auto &c:*ctx->ans_chunks)cudaFree(c.p);delete ctx->ans_chunks;}
        delete ctx->ans_scratch;
        ctx->ans_scratch=nullptr;ctx->ans_chunks=nullptr;ctx->ans_raw=nullptr;ctx->ans_raw_cap=0;
        ctx->ans_host=nullptr;ctx->ans_host_cap=0;ctx->ans_copy_pending=0;
#endif
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->host_x=ctx->host_y=ctx->host_kv=nullptr;ctx->stream=nullptr;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=ctx->host_kv_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
        ctx->d_gw_ptrs=ctx->d_uw_ptrs=ctx->d_dw_ptrs=nullptr;
        ctx->d_gsc_ptrs=ctx->d_usc_ptrs=ctx->d_dsc_ptrs=nullptr;
        ctx->d_ptrs_cap=0;
        ctx->d_dp4a_meta=nullptr;
        ctx->d_dp4a_meta_cap=ctx->d_dp4a_meta_count=0;
        ctx->d_dp4a_meta_valid=0;
        if (std::getenv("COLI_CUDA_META_STATS"))
            std::fprintf(stderr, "[dp4a-meta] device=%d updates=%llu skips=%llu hot_capability_queries=0\n",
                         ctx->device,
                         (unsigned long long)ctx->d_dp4a_meta_updates,
                         (unsigned long long)ctx->d_dp4a_meta_skips);
    }
    g_nctx = 0;
#ifdef COLI_ANS
    if(g_ans_sidecar){std::fclose(g_ans_sidecar);g_ans_sidecar=nullptr;}
#if defined(__linux__)
    if(g_ans_direct_fd>=0){close(g_ans_direct_fd);g_ans_direct_fd=-1;g_ans_direct_off=0;}
#endif
#endif
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

/* #653: 1 when the device shares physical memory with the host (Grace-Blackwell /
 * GB10, Jetson, integrated GPUs). On these the expert tier and the RAM cache draw
 * from the same pool, so the RAM budget must account for the tier; on a discrete GPU
 * VRAM is a separate pool and this returns 0. */
extern "C" int coli_cuda_device_integrated(int device) {
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) return 0;
    return prop.integrated ? 1 : 0;
}

extern "C" int coli_cuda_device_pci(int device, int *domain, int *bus,
                                     int *dev, int *function) {
    if (!domain || !bus || !dev || !function || device < 0) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    (void)device;
    return 0;
#else
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device PCI properties")) return 0;
    *domain = prop.pciDomainID;
    *bus = prop.pciBusID;
    *dev = prop.pciDeviceID;
    *function = 0; /* CUDA exposes the PCI device, not the function number. */
    return 1;
#endif
}

extern "C" int coli_cuda_peer_access(int dst_device,int src_device) {
    if (dst_device < 0 || src_device < 0) return 0;
    if (dst_device == src_device) return 1;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, dst_device, src_device) != cudaSuccess) return 0;
    return can != 0;
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" void coli_cuda_group_stats_device(
    int device, uint64_t *calls, uint64_t *experts, uint64_t *rows,
    double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    std::lock_guard<std::mutex> lock(g_group_stats_mu);
    int index=-1;
    for(int i=0;i<g_nctx;i++) if(g_ctx[i].device==device){ index=i; break; }
    if(calls) *calls=index<0?0:g_device_group_calls[index];
    if(experts) *experts=index<0?0:g_device_group_experts[index];
    if(rows) *rows=index<0?0:g_device_group_rows[index];
    if(h2d_ms) *h2d_ms=index<0?0:g_device_group_h2d_ms[index];
    if(kernel_ms) *kernel_ms=index<0?0:g_device_group_kernel_ms[index];
    if(d2h_ms) *d2h_ms=index<0?0:g_device_group_d2h_ms[index];
}

/* group size for the NEXT upload on this thread (fmt=4): routed through a
 * thread_local so the widely-wired upload signature (and the Windows DLL ABI)
 * stays untouched. pin_load uploads in parallel, hence thread_local. */
static thread_local int g_upload_gs = 0;
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs);
extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    if (!tensor) return 0;
    if (*tensor) {
        /* Cached device copy: usable even when the caller's host pointers are
         * gone. CUDA_RELEASE_HOST slots null their host pointers after upload,
         * and with the old order (!weights checked first) every later matmul
         * on such a slot failed here — the GPU tier silently never computed
         * for host-released slab experts. */
        ColiCudaTensor *t = *tensor;
        int want_gs = (fmt==4 && g_upload_gs>0) ? g_upload_gs : 0;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device && t->gs == want_gs;
    }
    DeviceContext *ctx = find_ctx(device);
    if (!weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    /* fmt=6 keeps its scales inside each 98-byte block, so it is the one
     * quantized format that legitimately arrives with scales == NULL. */
    if (!rb || (fmt && fmt != 6 && !scales)) return 0;
    if (fmt == 8 && !g_fp8_lut_ready) return 0;   /* kernels would read a zero LUT */
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    t->gs = (fmt==4 && g_upload_gs>0) ? g_upload_gs : 0;
    t->ng = t->gs ? (I + t->gs - 1) / t->gs : 1;
    t->scale_count = t->gs ? (size_t)O * (size_t)t->ng : (size_t)O;
    if (fmt == 8) {   /* per-128x128-block scales: [ceil(O/128), ceil(I/128)] */
        t->ng = (I + 127) / 128;
        t->scale_count = (size_t)((O + 127) / 128) * (size_t)t->ng;
    }
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    /* Ownership is a fact of the allocation, not the copy: set it BEFORE the
     * memcpy, or a failed H2D upload frees the tensor while weights_owned is
     * still 0 and free()'s ownership gate leaks the device buffer. */
    t->weights_owned=1;
    if (!cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2||fmt==4){ /* same nibble layout: offset-binary -> signed in place */
        offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt && fmt != 6) {
        if (!cuda_ok(cudaMalloc(&t->scales, t->scale_count * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, t->scale_count * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
        t->scales_owned=1;
    }
    if (fmt == 6) t->scale_count = 0;      /* in-block scales: nothing separate to track */
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + ((fmt && fmt != 6) ? t->scale_count * sizeof(float) : 0);
    *tensor = t;
    return 1;
}
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs){
    g_upload_gs = gs>0 ? gs : 0;
    int r = coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device);
    g_upload_gs = 0;
    return r;
}

#ifdef COLI_ANS
struct AnsSidecarHeader {
    uint32_t magic,raw_bytes,archive_bytes,fmt,I,O;
};
static FILE *ans_sidecar(void){
    if(g_ans_sidecar) return g_ans_sidecar;
    const char *path=std::getenv("COLI_ANS_SIDECAR");
    if(!path||!*path) return nullptr;
    g_ans_sidecar_pack=std::getenv("COLI_ANS_PACK")&&std::atoi(std::getenv("COLI_ANS_PACK"));
    g_ans_sidecar=std::fopen(path,g_ans_sidecar_pack?"wb":"rb");
#if defined(__linux__)
    if(g_ans_sidecar&&!g_ans_sidecar_pack&&std::getenv("COLI_ANS_DIRECT")&&
       std::atoi(std::getenv("COLI_ANS_DIRECT"))){
        g_ans_direct_fd=open(path,O_RDONLY|O_DIRECT);
        if(g_ans_direct_fd<0)std::fprintf(stderr,"[ANS] O_DIRECT unavailable; using buffered sidecar\n");
    }
    if(g_ans_sidecar&&!g_ans_sidecar_pack&&g_ans_direct_fd<0)
        posix_fadvise(fileno(g_ans_sidecar),0,0,POSIX_FADV_SEQUENTIAL);
#endif
    return g_ans_sidecar;
}
extern "C" int coli_cuda_tensor_upload_compressed(ColiCudaTensor **tensor,
        const void *weights,const float *scales,int fmt,int I,int O,int device){
    if(fmt!=2 || !tensor || *tensor) return 0;  /* prototype: per-row int4 experts only */
    FILE *sidecar=ans_sidecar();
    if(sidecar&&!g_ans_sidecar_pack){
        AnsSidecarHeader h{};
        size_t expected=((size_t)I+1)/2*(size_t)O;
        double t0=ans_now_s();
#if defined(__linux__)
        size_t direct_delta=0,direct_bytes=0,direct_data=0;
        if(g_ans_direct_fd>=0){
            alignas(4096) uint8_t first[8192];
            off_t start=g_ans_direct_off&~off_t(4095);
            direct_delta=(size_t)(g_ans_direct_off-start);
            ssize_t got=pread(g_ans_direct_fd,first,sizeof(first),start);
            if(got<(ssize_t)(direct_delta+sizeof(h))){
                std::fprintf(stderr,"[ANS] direct header read failed at %lld: got %lld errno %d\n",
                    (long long)g_ans_direct_off,(long long)got,errno);
                return 0;
            }
            std::memcpy(&h,first+direct_delta,sizeof(h));
        }else
#endif
        if(std::fread(&h,sizeof(h),1,sidecar)!=1)return 0;
        if(h.magic!=0x31534e41u||
           h.fmt!=(uint32_t)fmt||h.I!=(uint32_t)I||h.O!=(uint32_t)O||
           expected>UINT32_MAX||h.raw_bytes!=(uint32_t)expected||
           !h.archive_bytes||h.archive_bytes>dietgpu::getMaxCompressedSize(h.raw_bytes)){
            std::fprintf(stderr,"[ANS] invalid or mismatched sidecar record\n");
            return 0;
        }
        g_ans_header_s+=ans_now_s()-t0;
        DeviceContext *ctx=find_ctx(device); if(!ctx||!select_ctx(ctx)) return 0;
        ColiCudaTensor *t=(ColiCudaTensor*)std::calloc(1,sizeof(*t)); if(!t)return 0;
        t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->weight_bytes=h.raw_bytes;
        t->scale_count=(size_t)O;t->archive_bytes=h.archive_bytes;t->compressed=1;
        t0=ans_now_s();
        t->weights=ans_arena_alloc(ctx,h.archive_bytes);
        size_t scale_bytes=(size_t)O*sizeof(float),scale_off;
#if defined(__linux__)
        if(g_ans_direct_fd>=0){
            size_t record_bytes=sizeof(h)+(size_t)h.archive_bytes;
            direct_data=direct_delta+sizeof(h);
            direct_bytes=(direct_delta+record_bytes+4095)&~size_t(4095);
            scale_off=(direct_bytes+255)&~size_t(255);
        }else
#endif
            scale_off=(h.archive_bytes+255)&~size_t(255);
        if(!t->weights||!ans_host_reserve(ctx,scale_off+scale_bytes)||
           !cuda_ok(cudaMalloc(&t->scales,scale_bytes),"ANS sidecar scales")){
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_stage_s+=ans_now_s()-t0;
        t0=ans_now_s();
#if defined(__linux__)
        if(g_ans_direct_fd>=0){
            off_t start=g_ans_direct_off&~off_t(4095);
            ssize_t got=pread(g_ans_direct_fd,ctx->ans_host,direct_bytes,start);
            if(got<(ssize_t)(direct_data+h.archive_bytes)){
                std::fprintf(stderr,
                    "[ANS] direct record read failed at %lld: need %zu got %lld errno %d\n",
                    (long long)g_ans_direct_off,direct_data+h.archive_bytes,
                    (long long)got,errno);
                coli_cuda_tensor_free(t);return 0;
            }
            g_ans_direct_off+=(off_t)sizeof(h)+(off_t)h.archive_bytes;
        }else
#endif
        if(std::fread(ctx->ans_host,h.archive_bytes,1,sidecar)!=1){
            std::fprintf(stderr,"[ANS] truncated sidecar record\n");
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_read_s+=ans_now_s()-t0;
        std::memcpy((uint8_t*)ctx->ans_host+scale_off,scales,scale_bytes);
        t0=ans_now_s();
        void *archive_src=
#if defined(__linux__)
            g_ans_direct_fd>=0?(uint8_t*)ctx->ans_host+direct_data:
#endif
            ctx->ans_host;
        if(!cuda_ok(cudaMemcpyAsync(t->weights,archive_src,h.archive_bytes,
                                   cudaMemcpyHostToDevice,ctx->stream),"ANS sidecar upload")||
           !cuda_ok(cudaMemcpyAsync(t->scales,(uint8_t*)ctx->ans_host+scale_off,scale_bytes,
                                   cudaMemcpyHostToDevice,ctx->stream),"ANS sidecar scale upload")){
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_enqueue_s+=ans_now_s()-t0;g_ans_load_records++;
        ctx->ans_copy_pending=1;
        t->tracked=1;ctx->tensor_count++;ctx->tensor_bytes+=h.archive_bytes+(size_t)O*sizeof(float);
        *tensor=t;return 1;
    }
    if(!sidecar||!g_ans_sidecar_pack) return 0;
    if(!coli_cuda_tensor_upload(tensor,weights,scales,fmt,I,O,device)) return 0;
    ColiCudaTensor *t=*tensor;
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->ans_scratch||!select_ctx(ctx)){ coli_cuda_tensor_free(t);*tensor=nullptr;return 0; }
    uint32_t raw=(uint32_t)t->weight_bytes;
    uint32_t bound=dietgpu::getMaxCompressedSize(raw), *dsize=nullptr;
    void *tmp=nullptr;
    if(!cuda_ok(cudaMalloc(&tmp,bound),"ANS archive allocation")||
       !cuda_ok(cudaMalloc(&dsize,sizeof(*dsize)),"ANS size allocation")){
        if(tmp)cudaFree(tmp);if(dsize)cudaFree(dsize);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;
    }
    dietgpu::ANSCodecConfig config(11,false);
    /* tensor_upload converted offset-binary nibbles on the legacy stream.
     * ctx->stream is explicitly non-blocking, so it does not inherit the
     * legacy-stream dependency. Finish that one-time conversion before the
     * encoder reads the bytes. */
    if(!cuda_ok(cudaStreamSynchronize(0),"ANS source conversion synchronize")){
        cudaFree(tmp);cudaFree(dsize);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;
    }
    dietgpu::ansEncodeBatchStride(*ctx->ans_scratch,config,1,t->weights,raw,raw,nullptr,
                                  tmp,bound,dsize,ctx->stream);
    uint32_t used=0;
    int ok=cuda_ok(cudaMemcpyAsync(&used,dsize,sizeof(used),cudaMemcpyDeviceToHost,ctx->stream),
                   "ANS size download")&&
           cuda_ok(cudaStreamSynchronize(ctx->stream),"ANS encode synchronize")&&used>0&&used<raw;
    cudaFree(dsize);
    if(!ok){cudaFree(tmp);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;}
    if(g_ans_sidecar_pack){
        std::vector<uint8_t> archive(used);
        AnsSidecarHeader h{0x31534e41u,raw,used,(uint32_t)fmt,(uint32_t)I,(uint32_t)O};
        ok=cuda_ok(cudaMemcpy(archive.data(),tmp,used,cudaMemcpyDeviceToHost),"ANS sidecar download")&&
           std::fwrite(&h,sizeof(h),1,sidecar)==1&&
           std::fwrite(archive.data(),archive.size(),1,sidecar)==1;
        cudaFree(tmp);coli_cuda_tensor_free(t);*tensor=nullptr;
        if(!ok)return 0;
        t=(ColiCudaTensor*)std::calloc(1,sizeof(*t));if(!t)return 0;
        t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->weight_bytes=raw;
        t->archive_bytes=used;t->compressed=1;*tensor=t;
        return 1;
    }
    return 0;
}
#endif

extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                          const void *weights,
                                          const float *scales) {
    if (!tensor || !weights || (tensor->fmt && tensor->fmt != 6 && !scales)) return 0;
#ifdef COLI_ANS
    if(tensor->compressed) return 0;
#endif
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaMemcpy(tensor->weights,weights,tensor->weight_bytes,
                            cudaMemcpyHostToDevice),"tensor refresh")) return 0;
    if(tensor->fmt==2||tensor->fmt==4){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight refresh")) return 0;
    }
    /* fmt=6 has no scale buffer at all (scales live in-block, scale_count 0), and
     * the fallback below would otherwise copy O floats out of a NULL host pointer. */
    return !tensor->fmt || tensor->fmt==6 || cuda_ok(cudaMemcpy(tensor->scales,scales,
        (tensor->scale_count?tensor->scale_count:(size_t)tensor->O)*sizeof(float),
        cudaMemcpyHostToDevice),"scale refresh");
}

extern "C" int coli_cuda_expert_update_async(ColiCudaTensor *gate,
                                               ColiCudaTensor *up,
                                               ColiCudaTensor *down,
                                               const void *weights,
                                               const float *scales) {
    if(!gate || !up || !down || !weights || !scales ||
       gate->fmt!=4 || up->fmt!=4 || down->fmt!=4 ||
       gate->device!=up->device || gate->device!=down->device ||
       gate->weight_bytes!=up->weight_bytes || gate->weight_bytes!=down->weight_bytes ||
       gate->gs<=0 || up->gs!=gate->gs || down->gs!=gate->gs) return 0;
#ifdef COLI_ANS
    if(gate->compressed || up->compressed || down->compressed) return 0;
#endif
    DeviceContext *ctx=find_ctx(gate->device);
    if(!select_ctx(ctx)) return 0;
    const size_t wb=gate->weight_bytes;
    const size_t sg=gate->scale_count, su=up->scale_count, sd=down->scale_count;
    const uint8_t *w=(const uint8_t*)weights;
    if(!cuda_ok(cudaMemcpyAsync(gate->weights,w,wb,cudaMemcpyHostToDevice,ctx->stream),
                "async expert gate refresh") ||
       !cuda_ok(cudaMemcpyAsync(up->weights,w+wb,wb,cudaMemcpyHostToDevice,ctx->stream),
                "async expert up refresh") ||
       !cuda_ok(cudaMemcpyAsync(down->weights,w+2*wb,wb,cudaMemcpyHostToDevice,ctx->stream),
                "async expert down refresh")) return 0;
    offset_to_signed_s4<<<(unsigned)((wb+255)/256),256,0,ctx->stream>>>(
        (uint8_t*)gate->weights,wb);
    if(!cuda_ok(cudaGetLastError(),"async expert gate conversion")) return 0;
    offset_to_signed_s4<<<(unsigned)((wb+255)/256),256,0,ctx->stream>>>(
        (uint8_t*)up->weights,wb);
    if(!cuda_ok(cudaGetLastError(),"async expert up conversion")) return 0;
    offset_to_signed_s4<<<(unsigned)((wb+255)/256),256,0,ctx->stream>>>(
        (uint8_t*)down->weights,wb);
    if(!cuda_ok(cudaGetLastError(),"async expert down conversion")) return 0;
    const float *s=scales;
    if(!cuda_ok(cudaMemcpyAsync(gate->scales,s,sg*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),
                "async expert gate scales") ||
       !cuda_ok(cudaMemcpyAsync(up->scales,s+sg,su*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),
                "async expert up scales") ||
       !cuda_ok(cudaMemcpyAsync(down->scales,s+sg+su,sd*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),
                "async expert down scales")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_update_batch_async(
        ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
        ColiCudaTensor *const *downs, const void *const *weights,
        const float *const *scales, int count) {
    if (!gates || !ups || !downs || !weights || !scales || count < 1 || count > 64)
        return 0;
    const int debug = std::getenv("COLI_STAGE_BATCH_DEBUG") &&
                      std::atoi(std::getenv("COLI_STAGE_BATCH_DEBUG"));
    ColiCudaTensor *first=gates[0];
    if (!first || first->fmt!=4 || first->gs<=0) {
        if(debug) std::fprintf(stderr,"[qtier-stage-debug] invalid first tensor\n");
        return 0;
    }
    DeviceContext *ctx=find_ctx(first->device);
    if (!select_ctx(ctx)) {
        if(debug) std::fprintf(stderr,"[qtier-stage-debug] no device context %d\n",first->device);
        return 0;
    }
    const size_t wb=first->weight_bytes;
    const size_t sg=first->scale_count;
    if (!wb || !sg) {
        if(debug) std::fprintf(stderr,"[qtier-stage-debug] invalid sizes wb=%zu sg=%zu\n",wb,sg);
        return 0;
    }
    const size_t su=ups[0] ? ups[0]->scale_count : 0;
    const size_t sd=downs[0] ? downs[0]->scale_count : 0;
    if (!su || !sd) {
        if(debug) std::fprintf(stderr,"[qtier-stage-debug] invalid scale sizes su=%zu sd=%zu\n",su,sd);
        return 0;
    }
    uint8_t *dst[192];
    for (int i=0;i<count;i++) {
        ColiCudaTensor *g=gates[i],*u=ups[i],*d=downs[i];
        if (!g || !u || !d || !weights[i] || !scales[i] ||
            g->fmt!=4 || u->fmt!=4 || d->fmt!=4 ||
            g->device!=first->device || u->device!=first->device || d->device!=first->device ||
            g->weight_bytes!=wb || u->weight_bytes!=wb || d->weight_bytes!=wb ||
            g->scale_count!=sg || u->scale_count!=su || d->scale_count!=sd ||
            g->gs!=first->gs || u->gs!=first->gs || d->gs!=first->gs) {
            if(debug) std::fprintf(stderr,
                "[qtier-stage-debug] reject i=%d fmt=%d/%d/%d dev=%d/%d/%d wb=%zu/%zu/%zu sc=%zu/%zu/%zu gs=%d/%d/%d\n",
                i,g?g->fmt:-1,u?u->fmt:-1,d?d->fmt:-1,
                g?g->device:-1,u?u->device:-1,d?d->device:-1,
                g?g->weight_bytes:0,u?u->weight_bytes:0,d?d->weight_bytes:0,
                g?g->scale_count:0,u?u->scale_count:0,d?d->scale_count:0,
                g?g->gs:0,u?u->gs:0,d?d->gs:0);
            return 0;
        }
        dst[3*i+0]=(uint8_t*)g->weights;
        dst[3*i+1]=(uint8_t*)u->weights;
        dst[3*i+2]=(uint8_t*)d->weights;
    }
    const size_t ptr_bytes=3*(size_t)count*sizeof(void*);
    if (!ctx->d_gw_ptrs || ctx->d_ptrs_cap < ptr_bytes) {
        if (ctx->d_gw_ptrs) cudaFree(ctx->d_gw_ptrs);
        if (!cuda_ok(cudaMalloc(&ctx->d_gw_ptrs,ptr_bytes),"batch update pointer table")) {
            ctx->d_gw_ptrs=nullptr; ctx->d_ptrs_cap=0; return 0;
        }
        ctx->d_ptrs_cap=ptr_bytes;
    }
    if(debug) std::fprintf(stderr,"[qtier-stage-debug] batch count=%d device=%d wb=%zu sg=%zu su=%zu sd=%zu\n",
                          count,first->device,wb,sg,su,sd);
    for (int i=0;i<count;i++) {
        const uint8_t *w=(const uint8_t*)weights[i];
        const float *s=scales[i];
        if (!cuda_ok(cudaMemcpyAsync(gates[i]->weights,w,wb,cudaMemcpyHostToDevice,ctx->stream),
                     "batch gate refresh") ||
            !cuda_ok(cudaMemcpyAsync(ups[i]->weights,w+wb,wb,cudaMemcpyHostToDevice,ctx->stream),
                     "batch up refresh") ||
            !cuda_ok(cudaMemcpyAsync(downs[i]->weights,w+2*wb,wb,cudaMemcpyHostToDevice,ctx->stream),
                     "batch down refresh") ||
            !cuda_ok(cudaMemcpyAsync(gates[i]->scales,s,sg*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),
                     "batch gate scales") ||
            !cuda_ok(cudaMemcpyAsync(ups[i]->scales,s+sg,su*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),
                     "batch up scales") ||
            !cuda_ok(cudaMemcpyAsync(downs[i]->scales,s+sg+su,sd*sizeof(float),cudaMemcpyHostToDevice,ctx->stream),
                     "batch down scales")) return 0;
    }
    if(debug) std::fprintf(stderr,"[qtier-stage-debug] copies queued count=%d\n",count);
    if (!cuda_ok(cudaMemcpyAsync(ctx->d_gw_ptrs,dst,ptr_bytes,
                                 cudaMemcpyHostToDevice,ctx->stream),
                 "batch conversion pointer table")) return 0;
    size_t total=3*wb*(size_t)count;
    offset_to_signed_s4_ptrs<<<(unsigned)((total+255)/256),256,0,ctx->stream>>>(
        (uint8_t *const *)ctx->d_gw_ptrs,wb,3*count);
    int ok=cuda_ok(cudaGetLastError(),"batch int4 conversion");
    if(debug) std::fprintf(stderr,"[qtier-stage-debug] conversion queued ok=%d\n",ok);
    return ok;
}

/* Test hook: COLI_GPU_FAIL_AFTER=N makes every GPU COMPUTE entry point report
 * failure after N successful calls (N=0: every call fails), exercising the
 * engine's CPU fallbacks and host-rematerialization end-to-end without real
 * hardware faults. Uploads/queries are not gated. Unset: no effect. */
static long g_gpu_calls;
static int fault_injected(void) {
    const char *fa = std::getenv("COLI_GPU_FAIL_AFTER");
    return fa && g_gpu_calls++ >= std::atol(fa);
}

/* COLI_CUDA_F8_WARP mode, re-read per dispatch like its siblings. Strict
 * parse: a non-numeric value selects the DEFAULT, not atoi's silent 0.
 * Default 1 (warp kernels) on CUDA; 0 (original kernels) on HIP — the warp
 * kernels' wave64 width-32 shuffle sub-grouping has never been validated on
 * AMD silicon, and a scoring instrument must not default onto an untested
 * reduction. Opt in explicitly with =1/=2 once hip-test certifies it. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define COLI_F8_DEFAULT 0
#else
#define COLI_F8_DEFAULT 1
#endif
static int f8_warp_mode(void) {
    const char *e = std::getenv("COLI_CUDA_F8_WARP");
    if (!e || !*e) return COLI_F8_DEFAULT;
    char *end; long v = std::strtol(e, &end, 10);
    return *end ? COLI_F8_DEFAULT : (int)v;
}

/* One launch site for the dense matvec so fmt=8 honors the same toggle as
 * f8_group_launch: mode 0 runs the original quant_matmul branch (fully
 * original behavior), anything else the warp/shared-LUT rework. */
static void quant_matmul_launch(float *y, const float *x, const void *w,
        const float *sc, int fmt, int S, int I, int O, size_t rb, int gs, int ng) {
    dim3 grid((unsigned)O, (unsigned)S);
    if (fmt == 8 && f8_warp_mode())
        quant_matmul_f8w<<<grid, 256>>>(y, x, w, sc, S, I, O);
    else
        quant_matmul<<<grid, 256>>>(y, x, w, sc, fmt, S, I, O, rb, gs, ng);
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device, int gs) {
    if (fault_injected()) return 0;
    /* fmt=4 carries [O, ceil(I/gs)] scales: without the group size the plain
     * upload truncates the buffer to O floats and quant_matmul divides by
     * gs==0. Callers must come through the gs>0 path (upload_g) or stay on
     * the CPU (#298, #334). */
    if (fmt == 4 && gs <= 0) return 0;
    if (S < 1) return 0;
    if (gs > 0) { if (!coli_cuda_tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs)) return 0; }
    else        { if (!coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0; }
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    quant_matmul_launch(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb, t->gs, t->ng);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

/* MXFP4 matmul, stateless. Separate from coli_cuda_matmul on purpose: that one
 * takes scales as const float* and caches an uploaded tensor, while MXFP4
 * scales are ue8m0 BYTES -- passing them through the float* parameter would
 * compile and silently reinterpret the buffer. Kimi K3's routed experts stream
 * (a fill-once tier at decode), so there is nothing to cache here anyway; the
 * weights go up with the call.
 *
 * Returns 0 and leaves y untouched on any failure, which is the contract the
 * engine's GPU paths already use to fall back to CPU. */
extern "C" int coli_cuda_matmul_mxfp4(float *y, const float *x,
                                      const uint8_t *q4, const uint8_t *e8s,
                                      int S, int I, int O) {
    if (fault_injected()) return 0;
    if (S < 1 || I < 1 || O < 1 || !y || !x || !q4 || !e8s) return 0;
    DeviceContext *ctx = find_ctx(0);
    if (!select_ctx(ctx)) return 0;

    size_t rb = (size_t)(I + 1) / 2, ng = (size_t)(I + 31) / 32;
    size_t wb = (size_t)O * rb, sb = (size_t)O * ng;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);

    uint8_t *dw = nullptr, *ds = nullptr;
    if (!cuda_ok(cudaMalloc(&dw, wb), "mxfp4 weight alloc")) return 0;
    if (!cuda_ok(cudaMalloc(&ds, sb), "mxfp4 scale alloc")) { cudaFree(dw); return 0; }

    int ok = reserve(&ctx->x, &ctx->x_cap, xb) && reserve(&ctx->y, &ctx->y_cap, yb) &&
             cuda_ok(cudaMemcpy(dw, q4, wb, cudaMemcpyHostToDevice), "mxfp4 weight upload") &&
             cuda_ok(cudaMemcpy(ds, e8s, sb, cudaMemcpyHostToDevice), "mxfp4 scale upload") &&
             cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "mxfp4 input upload");
    if (ok) {
        dim3 grid((unsigned)O, (unsigned)S);
        quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, dw, reinterpret_cast<const float *>(ds),
                                    7, S, I, O, rb, 32, (int)ng);
        ok = cuda_ok(cudaGetLastError(), "mxfp4 launch") &&
             cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "mxfp4 output download");
    }
    cudaFree(dw);
    cudaFree(ds);
    return ok;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (fault_injected()) return 0;
    /* same reason as coli_cuda_matmul: fmt=4 without recorded group info would
     * misread the scales (and divide by gs==0 in the kernel). */
    if (gate && ((gate->fmt == 4 && gate->gs <= 0) ||
                 (up && up->fmt == 4 && up->gs <= 0) ||
                 (down && down->fmt == 4 && down->gs <= 0))) return 0;
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    quant_matmul_launch(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D),gate->gs,gate->ng);
    quant_matmul_launch(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D),up->gs,up->ng);
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    /* fmt=6: the down projection stores W@Q, so its input needs Q^T applied. This
     * one is per-expert (the silu product is not shared), unlike the gate/up input
     * rotation, which the caller does once per layer -- same split as moe(). */
    if (down->fmt == 6 && !e8_rot_rows_dev(ctx->gate, S, I, 0)) return 0;
    quant_matmul_launch(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I),down->gs,down->ng);
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if (fault_injected()) return 0;
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||!COLI_GPU_HAS_WMMA||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "shared w4a16 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    w4a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    w4a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"shared w4a16 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "shared w4a16 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"shared w4a16 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

/* Single launch site for the fmt=8 group kernels, shared by the sync and the
 * issue/take dispatches so both stay one-line call sites (rebase-tolerant
 * against #935's e8_group_launch refactor of the adjacent branch).
 * COLI_CUDA_F8_WARP (docs/ENVIRONMENT.md, parsed by f8_warp_mode): 1 = warp
 * kernels (the CUDA default; HIP defaults to 0), 0 = the original per-(o,s)
 * kernels (field escape hatch, dense path included via quant_matmul_launch),
 * 2 = warp kernels decoding through cuda_fp8.h — a real cvt instruction only
 * on sm_89+, the header's bit-manip emulation below that, and plain =1
 * behavior where cuda_fp8.h is absent (HIP); experimental until the
 * 256-value sweep certifies it bit-identical on the target silicon. */
static void f8_group_launch(DeviceContext *ctx,GroupDesc *dev,int I,int D,
                            int max_rows,int count){
    int mode=f8_warp_mode();
    if(mode){
        dim3 hg((unsigned)((I+7)/8),(unsigned)count),og((unsigned)((D+7)/8),(unsigned)count);
#if COLI_F8_HWCVT
        if(mode==2){
            grouped_hidden_f8w_dual<1><<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
            grouped_down_f8w<1><<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
            return;
        }
#endif
        grouped_hidden_f8w_dual<0><<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        grouped_down_f8w<0><<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        return;
    }
    dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
    grouped_hidden_f8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
    grouped_down_f8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
}

/* Refresh the compact metadata tuple used by the resident/synchronous DP4A
 * kernels.  The expert addresses are dynamic when routing or LFRU changes the
 * selected tuple, but they are stable while that tuple is in flight.  Keep a
 * single device allocation for the full supported count so no cudaFree/cudaMalloc
 * can invalidate work queued on the resident stream. */
static int dp4a_meta_prepare(DeviceContext *ctx,
                             ColiCudaTensor *const *gates,
                             ColiCudaTensor *const *ups,
                             ColiCudaTensor *const *downs,
                             int count, cudaStream_t stream) {
    if (!ctx || !gates || !ups || !downs || count <= 0 || count > 64)
        return 0;
    if (!ctx->d_dp4a_meta) {
        if (!cuda_ok(cudaMalloc((void **)&ctx->d_dp4a_meta,
                                64 * sizeof(ColiCudaDp4aExpertMeta)),
                         "dp4a metadata alloc"))
            return 0;
        ctx->d_dp4a_meta_cap = 64;
        ctx->d_dp4a_meta_count = 0;
        ctx->d_dp4a_meta_valid = 0;
    }

    ColiCudaDp4aExpertMeta host[64] = {};
    for (int c = 0; c < count; c++) {
        host[c].gw  = (const uint8_t *)gates[c]->weights;
        host[c].uw  = (const uint8_t *)ups[c]->weights;
        host[c].dw  = (const uint8_t *)downs[c]->weights;
        host[c].gsc = gates[c]->scales;
        host[c].usc = ups[c]->scales;
        host[c].dsc = downs[c]->scales;
    }
    int same = ctx->d_dp4a_meta_valid && count <= (int)ctx->d_dp4a_meta_count &&
               std::memcmp(host, ctx->h_dp4a_meta,
                           (size_t)count * sizeof(ColiCudaDp4aExpertMeta)) == 0;
    if (same) {
        ctx->d_dp4a_meta_skips++;
        return 1;
    }

    if (!cuda_ok(cudaMemcpyAsync(ctx->d_dp4a_meta, host,
                                 (size_t)count * sizeof(ColiCudaDp4aExpertMeta),
                                 cudaMemcpyHostToDevice, stream),
                 "dp4a metadata update"))
        return 0;
    std::memcpy(ctx->h_dp4a_meta, host,
                (size_t)count * sizeof(ColiCudaDp4aExpertMeta));
    ctx->d_dp4a_meta_count = (size_t)count;
    ctx->d_dp4a_meta_valid = 1;
    ctx->d_dp4a_meta_updates++;
    return 1;
}

/* Shared SM61 DP4A S=1 dispatcher. The input row is common to every expert;
 * gate/up produces count fused rows, which are then quantized once each for
 * the batched down projection. Keeping this in one helper prevents the sync
 * and issue paths from drifting back to per-expert launch loops. */
static int dp4a_group_launch(DeviceContext *ctx,
                             ColiCudaTensor *const *gates,
                             ColiCudaTensor *const *ups,
                             ColiCudaTensor *const *downs,
                             const int *rows, int count, int D, int I,
                             const float *input, int input_device) {
    if (!ctx || !gates || !ups || !downs || !rows || count <= 0 || count > 64)
        return 0;
    if (!input) return 0;
    if (D % 64 != 0 || I % 64 != 0 || D % 4 != 0) return 0;
    for (int c = 0; c < count; c++) {
        ColiCudaTensor *g = gates[c], *u = ups[c], *d = downs[c];
        int ggs = g && g->gs > 0 ? g->gs : 0;
        int ugs = u && u->gs > 0 ? u->gs : 0;
        int dgs = d && d->gs > 0 ? d->gs : 0;
        if (!g || !u || !d || rows[c] != 1 || g->fmt != 4 || u->fmt != 4 ||
            d->fmt != 4 || ggs != 64 || ugs != 64 || dgs != 64)
            return 0;
    }
    /* The host API supplies one concatenated row per expert and must reject
     * distinct rows because the batched DP4A kernels intentionally quantize
     * one shared row.  The resident island already has that one row on the
     * device; its caller has replicated the row before entering this helper. */
    if (!input_device)
        for (int c = 1; c < count; c++)
            if (std::memcmp(input, input + (size_t)c * D,
                            (size_t)D * sizeof(float)) != 0)
                return 0;

    /* The old implementation submitted six independent H2D updates for the
     * six pointer arrays.  The resident path now has one persistent device
     * metadata table; this is at most one compact refresh when the selected
     * routed tuple changes, and zero copies when the tuple is unchanged. */
    if (!dp4a_meta_prepare(ctx, gates, ups, downs, count, ctx->stream)) return 0;

    size_t qrows = (size_t)count * (size_t)I;
    size_t qb = (size_t)D > qrows ? (size_t)D : qrows;
    size_t qng = (size_t)((D + 63) / 64);
    size_t down_ng = (size_t)((I + 63) / 64);
    if ((size_t)count * down_ng > qng) qng = (size_t)count * down_ng;
    if (!reserve_bytes((void **)&ctx->qx, &ctx->qx_cap, qb) ||
        !reserve(&ctx->qscale, &ctx->qscale_cap, qng * sizeof(float))) return 0;
    /* `input` may alias an already device-resident layer row.  The regular
     * host/synchronous callers pass ctx->x; the resident coarse path can pass
     * its home-device row and avoid a redundant P2P copy. */
    const float *input_dev = input_device ? input : ctx->x;
    if (!coli_cuda_dp4a_quantize_row_g_s(input_dev, (int8_t *)ctx->qx,
                                         ctx->qscale, D, 64, ctx->stream)) return 0;
    if (!coli_cuda_dp4a_gate_up_meta_s(
            (const int8_t *)ctx->qx, ctx->qscale, ctx->d_dp4a_meta,
            ctx->gate, count, I, D, 64, ctx->stream)) return 0;
    if (!coli_cuda_dp4a_quantize_rows_g_s(
            ctx->gate, (int8_t *)ctx->qx, ctx->qscale,
            count, I, 64, ctx->stream)) return 0;
    return coli_cuda_dp4a_down_meta_s(
        (const int8_t *)ctx->qx, ctx->qscale, ctx->d_dp4a_meta,
        ctx->y, count, D, I, 64, ctx->stream);
}

static int expert_group_impl(ColiCudaTensor *const *gates,
                             ColiCudaTensor *const *ups,
                             ColiCudaTensor *const *downs,
                             const int *rows, int count,
                             float *y, const float *x,
                             int pin_small_batch) {
    if(!getenv("COLI_CUDA_DP4A_QUIET"))
        std::fprintf(stderr, "[dp4a] expert_group_impl called: count=%d D=%d I=%d\n",
                     count, gates&&gates[0]?gates[0]->I:0, gates&&gates[0]?gates[0]->O:0);
    if (fault_injected()) return 0;
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    int all_s4=1,all_q4=1,any_g4=0,any_e8=0,all_e8=1,any_f8=0,all_f8=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        any_f8|=g->fmt==8||u->fmt==8||d->fmt==8;
        all_f8&=g->fmt==8&&u->fmt==8&&d->fmt==8;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    /* Mixed E8/FP8 groups cannot use a homogeneous grouped kernel. */
    if((any_e8&&!all_e8)||(any_f8&&!all_f8)){
        int off=0;
        for(int c=0;c<count;c++){
            if(!coli_cuda_expert_mlp(gates[c],ups[c],downs[c],
                    y+(size_t)off*D,x+(size_t)off*D,rows[c])) return 0;
            off+=rows[c];
        }
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
        return 1;
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")){
        for(int j=0;j<i;j++) cudaEventDestroy(ev[j]); profile=0; break; }   /* (#B8) don't leak the events already created */
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    /* grouped_s4_wmma's body needs __CUDA_ARCH__>=750: on builds where the
     * WMMA kernels are compiled out (COLI_HIP_NO_WMMA) the launch would
     * succeed with an EMPTY kernel and the output buffer would silently keep
     * stale data. Gate the branch like TC_W4A16 below does. */
    tc=tc&&!pin_small_batch&&COLI_GPU_HAS_WMMA&&all_s4&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=rows[c]>=tc_min;
    if(all_e8){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_e8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        grouped_down_e8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_f8){
        /* fp8-e4m3 groups: silu fused in the dual epilogue, like the w4/g4 duals. */
        f8_group_launch(ctx,dev,I,D,max_rows,count);
    }else if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(!pin_small_batch&&all_s4&&COLI_GPU_HAS_WMMA&&ctx->compute_major>=7&&getenv("COLI_CUDA_TC_W4A16")&&
             atoi(getenv("COLI_CUDA_TC_W4A16"))&&
             [&]{ int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
                  for(int c=0;c<count;c++) if(rows[c]>=tc16_min) return 1;
                  return 0; }()){
        /* At least one expert has enough rows for a Tensor Core tile. Groups
         * where EVERY expert is below the threshold (decode: r=1) fall through
         * to the grouped-W4 path below — 3 launches for the whole group instead
         * of 4 per expert (#431: the launch flood measured at ~981 micro-kernels
         * per token came from decode riding this branch's per-expert fallback). */
        /* W4A16 Tensor Core per gruppo: attivazioni fp16 per tile (lossless al
         * contrario del path W4A4), un lancio per expert dentro lo stream —
         * l'overhead di lancio e' trascurabile rispetto ai GEMM. */
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                dim3 hg16((unsigned)((I+63)/64),(unsigned)((r+15)/16));
                dim3 og16((unsigned)((D+63)/64),(unsigned)((r+15)/16));
                w4a16_gate_up<<<hg16,256,0,ctx->stream>>>(g16,u16,x16,
                    (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                w4a16_matmul<<<og16,128,0,ctx->stream>>>(y16,g16,
                    (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
                    host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
                    host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
                    host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
            }
            off16+=r;
        }
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{   /* non-dual path has no fused epilogue: silu stays a kernel here */
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_q4&&any_g4){
        /* grouped-int4 (fmt=4) present: per-group scales (#334). fmt=2 members
         * ride along as the ng=1 special case. silu fused in the dual epilogue. */
        if(!getenv("COLI_CUDA_DP4A_QUIET"))
            std::fprintf(stderr, "[dp4a] reached g4 branch: count=%d D=%d I=%d max_rows=%d\n",
                         count, D, I, max_rows);
        /* DP4A opt-in: if the device is SM61 and COLI_CUDA_DP4A is set, route
         * the per-row INT8 GEMV path through the DP4A kernels defined in
         * backend_cuda_dp4a.cu. The activation is shared across all output
         * rows and experts, so it is quantized ONCE per layer per token into
         * ctx->qx / ctx->qscale. The activation scale is per-group (gs=64),
         * aligned with the weight group size. The W4 path remains the default
         * and the fallback when the gate is unset. */
        int dp4a_dispatched = 0;
        if(getenv("COLI_CUDA_DP4A")&&atoi(getenv("COLI_CUDA_DP4A"))&&
           ctx->compute_major==6&&ctx->compute_minor==1){
            int all_rows_1 = 1;
            for(int c=0;c<count;c++) if(rows[c] != 1){ all_rows_1 = 0; break; }
            if(all_rows_1){
                dp4a_dispatched = dp4a_group_launch(
                    ctx, gates, ups, downs, rows, count, D, I, x, 0);
                if(dp4a_dispatched && !getenv("COLI_CUDA_DP4A_QUIET")){
                    std::fprintf(stderr, "[dp4a] dispatched %d experts on D=%d I=%d\n",
                                 count, D, I);
                    std::fflush(stderr);
                }
            }
        }
        if(!dp4a_dispatched){
            dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
            grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
            grouped_down_g4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        }
    }else{
        /* generic path decodes fmt 0/1/2/3 only — refuse everything else rather
         * than whitelist known offenders: a fmt=4 group that slipped the gates
         * above (odd gs) must NOT be silently decoded as int2 (#334), and any
         * group/block-scaled format that gains CUDA tensors later (fmt=5, fmt=8)
         * carries scale geometry this kernel's epilogue does not apply.
         * STRICTER than weight_at's own admissible set on purpose: fmt=4 belongs
         * on the g4 path above, and returning 0 here keeps the CORRECT CPU
         * fallback, which is the outcome we want — weight_at's device-side
         * __trap backstop is for a launch that got past a gate like this one,
         * not a substitute for having the gate. */
        for(int c=0;c<count;c++)
            if(host[c].gf>3||host[c].uf>3||host[c].df>3) return 0;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          int index=(int)(ctx-g_ctx);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c;
          g_device_group_h2d_ms[index]+=a;
          g_device_group_kernel_ms[index]+=b;
          g_device_group_d2h_ms[index]+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      int index=(int)(ctx-g_ctx);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_device_group_calls[index]++; g_device_group_experts[index]+=(uint64_t)count;
      g_device_group_rows[index]+=(uint64_t)total; }
    return 1;
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    return expert_group_impl(gates,ups,downs,rows,count,y,x,0);
}

extern "C" int coli_cuda_expert_group_pinned(ColiCudaTensor *const *gates,
                                               ColiCudaTensor *const *ups,
                                               ColiCudaTensor *const *downs,
                                               const int *rows, int count,
                                               float *y, const float *x,
                                               int pin_small_batch) {
    return expert_group_impl(gates,ups,downs,rows,count,y,x,pin_small_batch);
}

/* ---- Async expert group (Inc.4): issue/take split of coli_cuda_expert_group ----
 * The measured cost of the sync call at decode is ~0.45 ms/call of HOST-side wait
 * (stream sync + staging), vs ~0.18 ms of actual GPU work — 70% tax, paid ~5x per
 * layer because a token's 8 experts scatter across devices. issue() stages and
 * launches on the device stream and returns immediately; take() syncs and hands
 * back the pinned result rows. One issue may be outstanding per device; moe()
 * takes at each layer end, which also orders the next layer's reuse of the ctx
 * scratch buffers. Small batches only (decode/spec): bigger totals keep the sync
 * path with its TC variants. Numerics are the sync path's small-batch kernels,
 * so greedy output is byte-identical by construction. */
static int expert_group_issue_impl(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              const float *x,
                                              int allow_dp4a) {
    if (!gates || !ups || !downs || !rows || !x || count < 1 || count > 64) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0,all_s4=1,any_e8=0,all_e8=1,
        all_q4=1,any_g4=0,any_f8=0,all_f8=1;
    GroupDesc host[64];
    /* Per-phase timing events (used by COLI_CUDA_TIMING_DUMP, see the take
     * function for the matching reader). Declared at function scope so the
     * post-dispatch stashing can see them. */
    const int phase_on = getenv("COLI_CUDA_TIMING_DUMP") && atoi(getenv("COLI_CUDA_TIMING_DUMP"));
    cudaEvent_t evp[EV_COUNT_PHASE] = {};
    auto phase_cleanup = [&]() {
        if (!phase_on) return;
        for (int i = 0; i < EV_COUNT_PHASE; i++) {
            if (evp[i]) cudaEventDestroy(evp[i]);
            evp[i] = nullptr;
        }
    };
#define PHASE_RECORD(slot) \
    do { \
        if (phase_on && !cuda_ok(cudaEventRecord(evp[(slot)], ctx->stream), \
                                 "timing event record")) { \
            phase_cleanup(); \
            return 0; \
        } \
    } while (0)
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_f8|=g->fmt==8||u->fmt==8||d->fmt==8;
        all_f8&=g->fmt==8&&u->fmt==8&&d->fmt==8;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    if(any_e8&&!all_e8) return 0;
    if(any_f8&&!all_f8) return 0;   /* mixed FP8: no homogeneous kernel, sync path has the per-expert loop */
    if(total>64) return 0;                      /* bounded prefill contract */
    DeviceContext *ctx=find_ctx(device); if(!ctx||ctx->group_pending||!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)) return 0;
    if (phase_on) {
        for (int i = 0; i < EV_COUNT_PHASE; i++) {
            if (cudaEventCreate(&evp[i]) != cudaSuccess) {
                phase_cleanup();
                return 0;
            }
        }
    }
    std::memcpy(ctx->host_x,x,xb);
    PHASE_RECORD(EV_H2D_PRE);
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue descriptors")||
       !cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue upload")) { phase_cleanup(); return 0; }
    if (phase_on) {
        PHASE_RECORD(EV_H2D_POST);
        PHASE_RECORD(EV_K0_PRE);
    }
    if(all_e8){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_e8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream)){ phase_cleanup(); return 0; }
        PHASE_RECORD(EV_K0_POST);
        PHASE_RECORD(EV_K1_PRE);
        grouped_down_e8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        PHASE_RECORD(EV_K1_POST);
        PHASE_RECORD(EV_K2_PRE);
        PHASE_RECORD(EV_K2_POST);
    }else if(all_f8){
        /* fp8-e4m3 groups on the async decode path: same launch helper as the
         * sync dispatch, silu fused in the dual epilogue. */
        f8_group_launch(ctx,(GroupDesc*)ctx->group_desc,I,D,max_rows,count);
        PHASE_RECORD(EV_K0_POST);
        PHASE_RECORD(EV_K1_PRE);
        PHASE_RECORD(EV_K1_POST);
        PHASE_RECORD(EV_K2_PRE);
        PHASE_RECORD(EV_K2_POST);
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual) grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else {
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(
                ctx->gate,ctx->up,(size_t)total*I);
        }
        PHASE_RECORD(EV_K0_POST);
        PHASE_RECORD(EV_K1_PRE);
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        PHASE_RECORD(EV_K1_POST);
        PHASE_RECORD(EV_K2_PRE);
        PHASE_RECORD(EV_K2_POST);
} else if(all_q4&&any_g4){
        /* grouped int4 (fmt=4) present in the async decode path: per-group
         * scales via the #334 kernels (fmt=2 members ride along as ng=1). The
         * previous fallback ran quant_matmul with gs=0,ng=1, which silently
         * applied one per-row scale to a grouped container -> wrong output. */

        /* DP4A opt-in (async): the qwen36 tier dispatches routed experts through
         * expert_group_issue, so the gate must be replicated here. The 4-launch
         * batched path:
         *   L1: quantize the SHARED gate/up input (D floats -> per-group INT8)
         *   L2: dp4a_gate_up_all (grid=(O_gu, count)) — one launch covers all
         *       count experts' gate+up projections, writes silu(g)*u to gate[]
         *   L3: quantize the count silu*up rows once into INT8 groups
         *   L4: dp4a_down_all (grid=(O_d/4, count)) — one launch covers all
         *       count experts' down projections using those rows
         * Total: 4 launches, vs W4's 2 launches. The quantizers are amortized
         * across all output rows and experts. */
        int dp4a_async_dispatched = 0;
        if(allow_dp4a && getenv("COLI_CUDA_DP4A")&&atoi(getenv("COLI_CUDA_DP4A"))&&
           ctx->compute_major==6&&ctx->compute_minor==1){
            extern int coli_cuda_dp4a_quantize_row_g_s(const float*,int8_t*,float*,int,int,cudaStream_t);
            extern int coli_cuda_dp4a_quantize_rows_g_s(const float*,int8_t*,float*,int,int,int,cudaStream_t);
            extern int coli_cuda_dp4a_gate_up_all_s(const int8_t*,const float*,
                const uint8_t*const*,const uint8_t*const*,const float*const*,
                const float*const*,float*,int,int,int,int,cudaStream_t);
            extern int coli_cuda_dp4a_gate_up_rows_s(const int8_t*,const float*,
                const uint8_t*const*,const uint8_t*const*,const float*const*,
                const float*const*,float*,int,int,int,int,cudaStream_t);
            extern int coli_cuda_dp4a_down_all_s(const int8_t*,const float*,
                const uint8_t*const*,const float*const*,float*,int,int,int,int,cudaStream_t);
            /* S>1 CRITICAL REJECTION. The DP4A kernels (gate_up_all,
             * down_all) are S=1-only: they assume total == count (one
             * input row per expert, K activations). The async path accepts
             * rows[c] up to 8. Silently accepting S>1 would quantize only
             * the first row, write one output row per expert, and produce
             * garbage for the others. Reject early. */
            int all_rows_1 = 1;
            for(int c=0;c<count;c++) if(rows[c]!=1){ all_rows_1=0; break; }
            if(all_rows_1 || allow_dp4a==2){
                /* Verify every expert meets the batched-path contract:
                 * fmt=4, gs=64 (per-expert), O%4==0 for down. Anything else
                 * falls back to W4. */
                int ok = 1;
                for(int c=0;c<count;c++){
                    ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
                    int cgs=g->gs>0?g->gs:64;
                    if(cgs!=64||g->fmt!=4||u->fmt!=4||d->fmt!=4) ok=0;
                }
                if(ok && (D % 4 != 0 || D % 64 != 0 || I % 64 != 0)) ok = 0;
                /* Verify u->gs and d->gs also equal 64 (mixed geometry would
                 * interpret scales under the wrong layout). g is checked
                 * below in the per-c loop; mirror it for u and d here. */
                if(ok){
                    for(int c=0;c<count;c++){
                        int ugs=ups[c]->gs>0?ups[c]->gs:64;
                        int dgs=downs[c]->gs>0?downs[c]->gs:64;
                        if(ugs!=64||dgs!=64){ ok=0; break; }
                    }
                }
                if(ok){
                    /* Build per-expert pointer arrays. The arrays must live in
                     * DEVICE memory (passing a host-stack array of device
                     * pointers to a kernel is undefined — the GPU would
                     * dereference host virtual addresses). Allocate device
                     * scratch once and reuse across calls. */
                    size_t ptrs_bytes = (size_t)count * sizeof(void*);
                    if(ctx->d_ptrs_cap < ptrs_bytes){
                        if(ctx->d_gw_ptrs) cudaFree(ctx->d_gw_ptrs);
                        if(ctx->d_uw_ptrs) cudaFree(ctx->d_uw_ptrs);
                        if(ctx->d_dw_ptrs) cudaFree(ctx->d_dw_ptrs);
                        if(ctx->d_gsc_ptrs) cudaFree(ctx->d_gsc_ptrs);
                        if(ctx->d_usc_ptrs) cudaFree(ctx->d_usc_ptrs);
                        if(ctx->d_dsc_ptrs) cudaFree(ctx->d_dsc_ptrs);
                        cudaMalloc(&ctx->d_gw_ptrs,  ptrs_bytes);
                        cudaMalloc(&ctx->d_uw_ptrs,  ptrs_bytes);
                        cudaMalloc(&ctx->d_dw_ptrs,  ptrs_bytes);
                        cudaMalloc(&ctx->d_gsc_ptrs, ptrs_bytes);
                        cudaMalloc(&ctx->d_usc_ptrs, ptrs_bytes);
                        cudaMalloc(&ctx->d_dsc_ptrs, ptrs_bytes);
                        ctx->d_ptrs_cap = ptrs_bytes;
                    }
                    /* Stage the per-expert device pointers into pinned host
                     * arrays, then H2D them. Direct H2D of host-stack
                     * pointers works on CUDA >= 11, but copying via the
                     * pointer arrays is explicit and safe. */
                    const uint8_t *h_gw[64], *h_uw[64], *h_dw[64];
                    const float   *h_gsc[64], *h_usc[64], *h_dsc[64];
                    for(int c=0;c<count;c++){
                        h_gw[c]  = (const uint8_t*)gates[c]->weights;
                        h_uw[c]  = (const uint8_t*)ups[c]->weights;
                        h_dw[c]  = (const uint8_t*)downs[c]->weights;
                        h_gsc[c] = gates[c]->scales;
                        h_usc[c] = ups[c]->scales;
                        h_dsc[c] = downs[c]->scales;
                    }
                    cudaMemcpyAsync(ctx->d_gw_ptrs,  h_gw,  ptrs_bytes, cudaMemcpyHostToDevice, ctx->stream);
                    cudaMemcpyAsync(ctx->d_uw_ptrs,  h_uw,  ptrs_bytes, cudaMemcpyHostToDevice, ctx->stream);
                    cudaMemcpyAsync(ctx->d_dw_ptrs,  h_dw,  ptrs_bytes, cudaMemcpyHostToDevice, ctx->stream);
                    cudaMemcpyAsync(ctx->d_gsc_ptrs, h_gsc, ptrs_bytes, cudaMemcpyHostToDevice, ctx->stream);
                    cudaMemcpyAsync(ctx->d_usc_ptrs, h_usc, ptrs_bytes, cudaMemcpyHostToDevice, ctx->stream);
                    cudaMemcpyAsync(ctx->d_dsc_ptrs, h_dsc, ptrs_bytes, cudaMemcpyHostToDevice, ctx->stream);
                    /* Reuse the A8 scratch for the gate input and then the
                     * count down-input rows. For qwen36 this is 4096 bytes
                     * plus 256 bytes of scales, still negligible. */
                    size_t qrows = (size_t)count * (size_t)I;
                    size_t qb = (size_t)D > qrows ? (size_t)D : qrows;
                    size_t qng = (size_t)((D + 63) / 64);
                    size_t down_ng = (size_t)((I + 63) / 64);
                    if ((size_t)count * down_ng > qng) qng = (size_t)count * down_ng;
                    size_t qscb = qng * sizeof(float);
                    if(!reserve_bytes((void**)&ctx->qx, &ctx->qx_cap, qb) ||
                       !reserve(&ctx->qscale, &ctx->qscale_cap, qscb)){
                        ok = 0;
                    } else {
                        if(!getenv("COLI_CUDA_DP4A_QUIET")){
                            std::fprintf(stderr, "[dp4a] L1 quantize input\n");
                            std::fflush(stderr);
                        }
                        int qok = allow_dp4a==2
                            ? coli_cuda_dp4a_quantize_rows_g_s(ctx->x, (int8_t*)ctx->qx, ctx->qscale, count, D, 64, ctx->stream)
                            : coli_cuda_dp4a_quantize_row_g_s(ctx->x, (int8_t*)ctx->qx, ctx->qscale, D, 64, ctx->stream);
                        if(!qok){
                            ok = 0;
                        } else {
                            PHASE_RECORD(EV_K0_POST);
                            if(!getenv("COLI_CUDA_DP4A_QUIET")){
                                std::fprintf(stderr, "[dp4a] L2 gate+up all (%d experts)\n", count);
                                std::fflush(stderr);
                            }
                            PHASE_RECORD(EV_K1_PRE);
                            int guok = allow_dp4a==2
                                ? coli_cuda_dp4a_gate_up_rows_s(
                                    (const int8_t*)ctx->qx, ctx->qscale,
                                    (const uint8_t * const *)ctx->d_gw_ptrs,
                                    (const uint8_t * const *)ctx->d_uw_ptrs,
                                    (const float * const *)ctx->d_gsc_ptrs,
                                    (const float * const *)ctx->d_usc_ptrs,
                                    ctx->gate, count, I, D, 64, ctx->stream)
                                : coli_cuda_dp4a_gate_up_all_s(
                                    (const int8_t*)ctx->qx, ctx->qscale,
                                    (const uint8_t * const *)ctx->d_gw_ptrs,
                                    (const uint8_t * const *)ctx->d_uw_ptrs,
                                    (const float * const *)ctx->d_gsc_ptrs,
                                    (const float * const *)ctx->d_usc_ptrs,
                                    ctx->gate, count, I, D, 64, ctx->stream);
                            if(!guok){
                                ok = 0;
                            } else {
                                PHASE_RECORD(EV_K1_POST);
                                if(!getenv("COLI_CUDA_DP4A_QUIET")){
                                    std::fprintf(stderr, "[dp4a] L3 quantize down inputs (%d rows)\n", count);
                                    std::fflush(stderr);
                                }
                                /* For S=1 decode, total==count. The down_all
                                 * writes count*O FP32 values to ctx->y. */
                                PHASE_RECORD(EV_K2_PRE);
                                if(!coli_cuda_dp4a_quantize_rows_g_s(
                                        ctx->gate, (int8_t*)ctx->qx, ctx->qscale,
                                        count, I, 64, ctx->stream)){
                                    ok = 0;
                                } else if(!coli_cuda_dp4a_down_all_s(
                                        (const int8_t*)ctx->qx, ctx->qscale,
                                        (const uint8_t * const *)ctx->d_dw_ptrs,
                                        (const float * const *)ctx->d_dsc_ptrs,
                                        ctx->y, count, D, I, 64, ctx->stream)){
                                    ok = 0;
                                }
                                PHASE_RECORD(EV_K2_POST);
                            }
                        }
                    }
                }
                dp4a_async_dispatched = ok;
            }
        }
        if(!dp4a_async_dispatched){
            GroupDesc *dev=(GroupDesc*)ctx->group_desc;
            dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
            dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
            /* silu is fused in the dual kernel's epilogue (like the sync path):
             * an extra silu_mul here would re-apply it against the never-written
             * ctx->up buffer. */
            grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
            PHASE_RECORD(EV_K0_POST);
            PHASE_RECORD(EV_K1_PRE);
            grouped_down_g4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
            PHASE_RECORD(EV_K1_POST);
            PHASE_RECORD(EV_K2_PRE);
            PHASE_RECORD(EV_K2_POST);
        }
     } else {
        /* Fallback runs quant_matmul with gs=0,ng=1 — per-row-scale semantics.
         * That is only correct for fmt 0/1/2/3: refuse group/block-scaled
         * members (fmt=4 with odd gs today; fmt=5/8 if they ever gain CUDA
         * tensors) instead of silently mis-scaling them, mirroring the sync
         * path's refusal (#334). fmt=6 cannot reach here (any_e8 gates above). */
        for(int c=0;c<count;c++)
            if(host[c].gf>3||host[c].uf>3||host[c].df>3){ phase_cleanup(); return 0; }
        for(int c=0;c<count;c++){
        int r=rows[c];
        float *g16=ctx->gate+(size_t)host[c].offset*I,*u16=ctx->up+(size_t)host[c].offset*I;
        float *x16=ctx->x+(size_t)host[c].offset*D,*y16=ctx->y+(size_t)host[c].offset*D;
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
            host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
            host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
        silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
        quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
            host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
        }
        if (phase_on) {
            PHASE_RECORD(EV_K0_POST);
            PHASE_RECORD(EV_K1_PRE);
            PHASE_RECORD(EV_K1_POST);
            PHASE_RECORD(EV_K2_PRE);
            PHASE_RECORD(EV_K2_POST);
        }
    }
    PHASE_RECORD(EV_D2H_PRE);
    if(!cuda_ok(cudaGetLastError(),"expert group issue launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                "expert group issue download")) { phase_cleanup(); return 0; }
    if (phase_on) {
        PHASE_RECORD(EV_D2H_POST);
        /* Final event AFTER the D2H enqueue. We don't sync here; the
         * event reads happen on the take path. */
        PHASE_RECORD(EV_TOTAL_POST);
        /* Hand the events to the take path. If a previous issue was still
         * pending (shouldn't be — async engine is serial), destroy those
         * first. */
        phase_events_destroy(ctx);
        for (int i = 0; i < EV_COUNT_PHASE; i++) {
            ctx->phase_ev[i] = evp[i];
            evp[i] = nullptr;
        }
        ctx->phase_ev_valid = 1;
    }
    ctx->group_pending=1; ctx->group_pending_bytes=xb;
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      int index=(int)(ctx-g_ctx);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_device_group_calls[index]++; g_device_group_experts[index]+=(uint64_t)count;
      g_device_group_rows[index]+=(uint64_t)total; }
    return 1;
#undef PHASE_RECORD
}

extern "C" int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              const float *x) {
    return expert_group_issue_impl(gates, ups, downs, rows, count, x, 1);
}

extern "C" int coli_cuda_expert_group_issue_batch(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              const float *x) {
    return expert_group_issue_impl(gates, ups, downs, rows, count, x, 2);
}

extern "C" const float *coli_cuda_expert_group_take(int device) {
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->group_pending) return nullptr;
    ctx->group_pending=0;
    if(!select_ctx(ctx)) return nullptr;
    if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group take")) return nullptr;
    /* Phase timing dump: read elapsed times from events stashed by issue,
     * append a CSV row to COLI_CUDA_TIMING_FILE (or stderr if unset). */
    if(ctx->phase_ev_valid){
        float t_h2d=0, t_k0=0, t_k1=0, t_k2=0, t_d2h=0, t_total=0;
        int have_all = 1;
        for (int i = 0; i < EV_COUNT_PHASE; i++)
            if (!ctx->phase_ev[i]) { have_all = 0; break; }
        int elapsed_ok = have_all;
        if (elapsed_ok) elapsed_ok &= cuda_ok(cudaEventElapsedTime(&t_h2d, ctx->phase_ev[EV_H2D_PRE], ctx->phase_ev[EV_H2D_POST]), "timing h2d");
        if (elapsed_ok) elapsed_ok &= cuda_ok(cudaEventElapsedTime(&t_k0, ctx->phase_ev[EV_K0_PRE], ctx->phase_ev[EV_K0_POST]), "timing k0");
        if (elapsed_ok) elapsed_ok &= cuda_ok(cudaEventElapsedTime(&t_k1, ctx->phase_ev[EV_K1_PRE], ctx->phase_ev[EV_K1_POST]), "timing k1");
        if (elapsed_ok) elapsed_ok &= cuda_ok(cudaEventElapsedTime(&t_k2, ctx->phase_ev[EV_K2_PRE], ctx->phase_ev[EV_K2_POST]), "timing k2");
        if (elapsed_ok) elapsed_ok &= cuda_ok(cudaEventElapsedTime(&t_d2h, ctx->phase_ev[EV_D2H_PRE], ctx->phase_ev[EV_D2H_POST]), "timing d2h");
        if (elapsed_ok) elapsed_ok &= cuda_ok(cudaEventElapsedTime(&t_total, ctx->phase_ev[EV_H2D_PRE], ctx->phase_ev[EV_TOTAL_POST]), "timing total");
        if (elapsed_ok) {
            const char *path = std::getenv("COLI_CUDA_TIMING_FILE");
            FILE *fp = path ? std::fopen(path, "a") : stderr;
            if(fp){
                /* cudaEventElapsedTime returns milliseconds (float). */
                std::fprintf(fp, "h2d_ms=%.4f k0_ms=%.4f k1_ms=%.4f k2_ms=%.4f d2h_ms=%.4f total_ms=%.4f\n",
                             t_h2d, t_k0, t_k1, t_k2, t_d2h, t_total);
                if(path) std::fclose(fp);
            }
        }
        phase_events_destroy(ctx);
    }
    return ctx->host_y;
}


/* The absorb kernels decode `w` through weight_at + absorb_scale, which know
 * per-row and fmt=4 group scales only. Refuse anything else (fmt=5/6/8) rather
 * than mis-decode it — the caller keeps its CPU attention path. (`proj`
 * tensors are exempt: they run through quant_matmul, which dispatches every
 * format it uploads.) A dedicated block-scale absorb for fmt=8 is follow-up
 * work, same shape as routing fmt=4 through the grouped kernels was.
 *
 * The admissible set is weight_at's own, taken from the shared predicate rather
 * than restated as `fmt <= 4`: this gate and weight_at's device-side backstop
 * must not be able to drift apart, and the old inequality also admitted
 * NEGATIVE fmt values, which weight_at would then have fallen through on. Same
 * truth table for every fmt a container can actually carry (0..8), so no
 * existing container changes behaviour here. */
static int absorb_fmt_ok(const ColiCudaTensor *w){
    return w && coli_cuda_weight_at_supported(w->fmt);
}

extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,int S,int H,int Q,int R,int V,
        int K,int T,float scale){
    if(!absorb_fmt_ok(w)||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention batch launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_ragged(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!absorb_fmt_ok(w)||!proj||!out||!q||!keys||!latent||!rope||!lengths||S<1||S>512||T<1||T>8192||
       H<1||Q<1||R<1||V<1||K<1||K>512||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);
    if(!select_ctx(dc))return 0;
    int *old=(int*)std::malloc((size_t)S*sizeof(*old));
    int *add=(int*)std::malloc((size_t)S*sizeof(*add));
    int *off=(int*)std::malloc((size_t)S*sizeof(*off));int packed_n=0;
    if(!old||!add||!off){std::free(old);std::free(add);std::free(off);return 0;}
    int page_stride=0;
    for(int s=0;s<S;s++){
        if(!keys[s]||lengths[s]<1||lengths[s]>T){std::free(old);std::free(add);std::free(off);return 0;}
        RaggedKVEntry *e=nullptr;
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){e=&w->ragged[i];break;}
        if(!e){
            if(w->ragged_count>=512){std::free(old);std::free(add);std::free(off);return 0;}
            e=&w->ragged[w->ragged_count++];std::memset(e,0,sizeof(*e));e->key=keys[s];
        }
        if(e->K!=K||e->R!=R||e->host_l!=latent[s]||e->host_r!=rope[s]||lengths[s]<e->length){
            ragged_kv_clear(e);
            e->K=K;e->R=R;e->host_l=latent[s];e->host_r=rope[s];
        }
        int need=(lengths[s]+COLI_KV_PAGE_TOKENS-1)/COLI_KV_PAGE_TOKENS;
        if(need>e->page_count){
            float **nl=(float**)std::calloc((size_t)need,sizeof(*nl));
            float **nr=(float**)std::calloc((size_t)need,sizeof(*nr));
            if(!nl||!nr){std::free(nl);std::free(nr);std::free(old);std::free(add);std::free(off);return 0;}
            for(int i=0;i<e->page_count;i++){nl[i]=e->latent_pages[i];nr[i]=e->rope_pages[i];}
            int made=e->page_count;
            for(;made<need;made++){
                if(!cuda_ok(cudaMalloc(&nl[made],(size_t)COLI_KV_PAGE_TOKENS*K*sizeof(float)),"ragged KV latent page")||
                   !cuda_ok(cudaMalloc(&nr[made],(size_t)COLI_KV_PAGE_TOKENS*R*sizeof(float)),"ragged KV rope page"))break;
            }
            if(made<need){
                if(nl[made])cudaFree(nl[made]);if(nr[made])cudaFree(nr[made]);
                for(int i=e->page_count;i<made;i++){cudaFree(nl[i]);cudaFree(nr[i]);}
                std::free(nl);std::free(nr);std::free(old);std::free(add);std::free(off);return 0;
            }
            std::free(e->latent_pages);std::free(e->rope_pages);
            e->latent_pages=nl;e->rope_pages=nr;e->page_count=need;
        }
        if(e->page_count>page_stride)page_stride=e->page_count;
        old[s]=e->length;add[s]=lengths[s]-e->length;
        off[s]=packed_n;packed_n+=add[s]*(K+R);
    }
    size_t table_n=(size_t)S*page_stride;
    float **dl=(float**)std::calloc(table_n,sizeof(*dl));
    float **dr=(float**)std::calloc(table_n,sizeof(*dr));
    if(!dl||!dr){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
    for(int s=0;s<S;s++)for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){
        for(int p=0;p<w->ragged[i].page_count;p++){
            dl[(size_t)s*page_stride+p]=w->ragged[i].latent_pages[p];
            dr[(size_t)s*page_stride+p]=w->ragged[i].rope_pages[p];
        }
        break;
    }
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float);
    size_t cb=(size_t)S*H*V*sizeof(float),ob=(size_t)S*proj->O*sizeof(float);
    size_t pb=(size_t)packed_n*sizeof(float);
    size_t desc=2*table_n*sizeof(float*)+(size_t)S*4*sizeof(int);
    int ok=reserve(&dc->aq,&dc->aq_cap,qb)&&reserve(&dc->ac,&dc->ac_cap,cb)&&
           reserve(&dc->y,&dc->y_cap,ob)&&reserve_bytes(&dc->group_desc,&dc->group_desc_cap,desc)&&
           (!pb||(reserve(&dc->al,&dc->al_cap,pb)&&reserve_pinned(&dc->host_kv,&dc->host_kv_cap,pb)));
    char *db=(char*)dc->group_desc;float **ddl=(float**)db,**ddr=ddl+table_n;
    int *dn=(int*)(ddr+table_n),*dold=dn+S,*dadd=dold+S,*doff=dadd+S;
    if(ok&&pb){
        for(int s=0;s<S;s++)if(add[s]){
            float *p=dc->host_kv+off[s];
            std::memcpy(p,latent[s]+(size_t)old[s]*K,(size_t)add[s]*K*sizeof(float));
            std::memcpy(p+(size_t)add[s]*K,rope[s]+(size_t)old[s]*R,(size_t)add[s]*R*sizeof(float));
        }
        ok=cuda_ok(cudaMemcpyAsync(dc->al,dc->host_kv,pb,cudaMemcpyHostToDevice,dc->stream),"ragged KV append upload");
    }
    if(ok)ok=cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"ragged q upload")&&
             cuda_ok(cudaMemcpyAsync(ddl,dl,table_n*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged latent page table")&&
             cuda_ok(cudaMemcpyAsync(ddr,dr,table_n*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged rope page table")&&
             cuda_ok(cudaMemcpyAsync(dn,lengths,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged lengths upload")&&
             cuda_ok(cudaMemcpyAsync(dold,old,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged old lengths")&&
             cuda_ok(cudaMemcpyAsync(dadd,add,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append lengths")&&
             cuda_ok(cudaMemcpyAsync(doff,off,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append offsets");
    if(ok&&pb)ragged_kv_append<<<S,256,0,dc->stream>>>(ddl,ddr,dc->al,dold,dadd,doff,K,R,page_stride);
    if(ok)for(int s=0;s<S;s++){
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){w->ragged[i].length=lengths[s];break;}
    }
    std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);if(!ok)return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_ragged_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,ddl,ddr,
        dn,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,page_stride,scale,w->gs,w->ng);
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    return cuda_ok(cudaGetLastError(),"ragged attention launch")&&
           cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"ragged output download")&&
           cuda_ok(cudaStreamSynchronize(dc->stream),"ragged attention synchronize");
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        /* Must mirror the upload's accounting exactly -- literally the same
         * expression upload uses to charge (scale_count * sizeof(float), gated
         * on fmt=6 never having a separate scale buffer), so the two can no
         * longer drift independently. Over-subtracting here trips the >= guard
         * below, which silently leaves the tensor's bytes on the device counter
         * forever. */
        size_t storage_bytes =
#ifdef COLI_ANS
            tensor->compressed ? tensor->archive_bytes :
#endif
            tensor->weight_bytes;
        size_t bytes = storage_bytes +
            ((tensor->fmt && tensor->fmt != 6) ? tensor->scale_count * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights&&tensor->weights_owned) cudaFree(tensor->weights);
    if (tensor->scales&&tensor->scales_owned) cudaFree(tensor->scales);
    for(int i=0;i<tensor->ragged_count;i++)ragged_kv_clear(&tensor->ragged[i]);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    /* Must mirror upload's and free's accounting exactly -- literally the same
     * expression they use (scale_count * sizeof(float), gated on fmt=6 never
     * having a separate scale buffer) -- so all three can no longer drift
     * independently. The prior `O * ng` shape over-reported for fmt=8 (real
     * footprint is (O+127)/128 * ng block scales, not O * ng) and for fmt=6
     * (which has no separate scale buffer at all). */
    size_t storage_bytes =
#ifdef COLI_ANS
        tensor->compressed ? tensor->archive_bytes :
#endif
        tensor->weight_bytes;
    return storage_bytes +
        ((tensor->fmt && tensor->fmt != 6) ? tensor->scale_count * sizeof(float) : 0);
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

__global__ static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    const float *xr=x+(size_t)blockIdx.x*xstride; float *yr=y+(size_t)blockIdx.x*ystride;
    __shared__ double sh[256];
    double a=0; for(int i=threadIdx.x;i<D;i+=blockDim.x){ double v=xr[i]; a+=v*v; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float r=rsqrtf((float)(sh[0]/D)+eps);
    for(int i=threadIdx.x;i<D;i+=blockDim.x) yr[i]=xr[i]*r*w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
__global__ static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    float *p=v+(size_t)blockIdx.x*stride+offset;
    int half=R/2, ps=pos?pos[blockIdx.x/heads]:pos_base+(int)(blockIdx.x/heads);
    __shared__ float in[256];
    for(int j=threadIdx.x;j<R;j+=blockDim.x) in[j]=p[j];
    __syncthreads();
    for(int j=threadIdx.x;j<half;j+=blockDim.x){
        float inv=__powf(theta,-2.0f*j/R);
        float ang=ps*inv, cs=__cosf(ang), sn=__sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

__global__ static void pipe_add_n(float *x,const float *t,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
__global__ static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    float *xr=x+(size_t)rows[blockIdx.x]*D;
    const float *pr=partial+(size_t)blockIdx.x*D;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) xr[i]+=pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=32||!select_ctx(ctx)) return NULL;
    if(!reserve(&ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
extern "C" void *coli_cuda_pipe_alloc(int device,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if(!cuda_ok(cudaMalloc(&p,bytes),"pipe alloc")) return NULL;
    return p;
}
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    cudaFree(p);
}
extern "C" int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"pipe upload");
}
extern "C" int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"pipe download");
}
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,D,D);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,xstride,ystride);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,pos_dev,0,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,NULL,pos_base,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope base");
}
/* ---- device router (#431 PR-A) -------------------------------------------
 * Router for one decode row, entirely on the layer's home device: logits GEMV
 * (E x D, tiny) + sigmoid, bias-augmented top-K selection, route-level TOPP
 * truncation, norm_topk and routed_scale — a float-faithful clone of moe()'s
 * plain routing path (colibri.c FASE A). Selection runs single-thread so the
 * argmax order, tie-breaking (strict >, lowest index wins) and weight math
 * match the CPU reference exactly; only the dot/expf rounding can differ,
 * which is the documented kernel-family divergence class (#100/#163).
 * Results are packed [idx[K] | w[K] | keff] in one scratch buffer and read
 * back with a single tiny D2H. */
__global__ void pipe_router_logits(const float *__restrict__ x,
                                   const float *__restrict__ W,
                                   const float *__restrict__ bias,
                                   int D, float *logit, float *choice){
    int e = blockIdx.x;
    const float *w = W + (size_t)e*D;
    float acc = 0.f;
    for(int i=threadIdx.x; i<D; i+=blockDim.x) acc += x[i]*w[i];
    __shared__ float sh[128];
    sh[threadIdx.x]=acc; __syncthreads();
    for(int s=blockDim.x>>1; s>0; s>>=1){
        if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s];
        __syncthreads();
    }
    if(!threadIdx.x){
        float lg = 1.f/(1.f+expf(-sh[0]));
        logit[e]=lg; choice[e]=lg+bias[e];
    }
}
__global__ void pipe_router_select(const float *__restrict__ logit,
                                   const float *__restrict__ choice, int E,
                                   int Ksel, float topp, int norm_topk,
                                   float routed_scale, char *out){
    if(threadIdx.x||blockIdx.x) return;
    int   *idx = (int*)out;
    float *w   = (float*)(out + Ksel*sizeof(int));
    int   *keff= (int*)(out + Ksel*(sizeof(int)+sizeof(float)));
    for(int kk=0;kk<Ksel;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
            if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
        idx[kk]=best; w[kk]=logit[best];
    }
    int Ke=Ksel;
    if(topp>0.f && topp<1.f){
        for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
            while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
        float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
        float cum=0.f; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
    }
    if(norm_topk){ float sm=0.f; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f;
                   for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
    for(int kk=0;kk<Ke;kk++) w[kk]*=routed_scale;
    *keff=Ke;
}
extern "C" int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host){
    DeviceContext *ctx=find_ctx(device);
    if(!x_dev||!rw_dev||!rb_dev||D<1||E<1||E>4096||Ksel<1||Ksel>64||!select_ctx(ctx)) return 0;
    size_t pack=(size_t)Ksel*(sizeof(int)+sizeof(float))+sizeof(int);
    float *logit=coli_cuda_pipe_scratch(device,22,(size_t)E*sizeof(float));
    float *chc  =coli_cuda_pipe_scratch(device,23,(size_t)E*sizeof(float));
    char  *out  =(char*)coli_cuda_pipe_scratch(device,24,pack);
    if(!logit||!chc||!out) return 0;
    pipe_router_logits<<<E,128>>>(x_dev,(const float*)rw_dev,(const float*)rb_dev,D,logit,chc);
    pipe_router_select<<<1,1>>>(logit,chc,E,Ksel,topp,norm_topk,routed_scale,out);
    if(!cuda_ok(cudaGetLastError(),"pipe router launch")) return 0;
    char buf[64*(sizeof(int)+sizeof(float))+sizeof(int)];
    if(!cuda_ok(cudaMemcpy(buf,out,pack,cudaMemcpyDeviceToHost),"pipe router readback")) return 0;
    memcpy(idx_host,buf,(size_t)Ksel*sizeof(int));
    memcpy(w_host,buf+Ksel*sizeof(int),(size_t)Ksel*sizeof(float));
    memcpy(keff_host,buf+Ksel*(sizeof(int)+sizeof(float)),sizeof(int));
    return 1;
}
/* ---- resident expert-group accumulation (#431 PR-C0) ----------------------
 * Decode-time (S=1) expert groups without the host round-trip: the input row
 * is P2P'd from the layer's home device, the group runs through the grouped-W4
 * kernels on its own stream, the down-projection outputs are weighted and
 * reduced ON DEVICE (fixed expert order), and the device's partial sum is
 * peer-pushed into a per-issue slot on the home device. take() makes the home
 * legacy stream wait on every issue event and reduces the slots in issue order
 * — deterministic, no atomics, no host bytes. The CPU tier overlaps with all
 * of it exactly as before. */
__global__ static void bcast_row(float *dst,const float *src,int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float v=src[i];
        for(int c=0;c<count;c++) dst[(size_t)c*D+i]=v;
    }
}
__global__ static void weighted_sum_rows(float *out,const float *y,const float *w,
                                         int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int c=0;c<count;c++) acc+=w[c]*y[(size_t)c*D+i];   /* fixed order */
        out[i]=acc;
    }
}
__global__ static void sum_slots(float *dst,const float *slots,int n,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int s=0;s<n;s++) acc+=slots[(size_t)s*D+i];        /* issue order */
        dst[i]=acc;
    }
}
extern "C" int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev){
    if(fault_injected()) return 0;
    if(!gates||!ups||!downs||!weights||count<1||count>64||!x_src_dev||!partial_slot_dev) return 0;
    ColiCudaTensor *first=gates[0]; if(!first) return 0;
    int device=first->device,D=first->I,I=first->O;
    GroupDesc host[64];
    int total=0,all_q4=1,any_g4=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,1,total,
                 g->gs,u->gs,d->gs};
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4);
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        total++;
    }
    if(!all_q4) return 0;                       /* resident path: W4/fmt4 only */
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    if(!ctx->ev_done_ok){
        if(!cuda_ok(cudaEventCreateWithFlags(&ctx->ev_done,cudaEventDisableTiming),
                    "resident group event")) return 0;
        ctx->ev_done_ok=1;
    }
    /* size for the 64-expert cap, not for `count`: reserve() reallocs on growth,
     * and a realloc here could free a buffer the PREVIOUS layer's still-queued
     * async work on this stream reads. Fixed caps make re-issue realloc-free. */
    size_t xb=(size_t)64*D*sizeof(float), ib=(size_t)64*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->ac,&ctx->ac_cap,(size_t)(D+64)*sizeof(float))||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)64*sizeof(GroupDesc)))
        return 0;
    float *w_dev=ctx->ac+D, *partial_local=ctx->ac;
    /* input row: normally P2P from the home device.  On a single GPU, however,
     * the home row is already on the same device.  When the proven SM61 DP4A
     * geometry is active, the executor only reads the row, so it can consume
     * x_src_dev directly and remove one device-to-device copy per resident
     * layer.  Keep the old ctx->x copy for every other format/path: the legacy
     * grouped kernels may broadcast into ctx->x and must retain that scratch
     * ownership. */
    int direct_input = 0;
    const char *direct_input_env = getenv("COLI_CUDA_DIRECT_INPUT");
    if (direct_input_env && atoi(direct_input_env) && device == home_device &&
        any_g4 && ctx->compute_major == 6 && ctx->compute_minor == 1 &&
        getenv("COLI_CUDA_DP4A") && atoi(getenv("COLI_CUDA_DP4A")) &&
        D % 64 == 0 && I % 64 == 0 && D % 4 == 0) {
        direct_input = 1;
        for (int c = 0; c < count; c++) {
            ColiCudaTensor *g = gates[c], *u = ups[c], *d = downs[c];
            if (!g || !u || !d || g->fmt != 4 || u->fmt != 4 || d->fmt != 4 ||
                g->gs != 64 || u->gs != 64 || d->gs != 64) {
                direct_input = 0;
                break;
            }
        }
    }
    const float *resident_input_dev = ctx->x;
    if (direct_input) {
        resident_input_dev = x_src_dev;
    } else if (!cuda_ok(cudaMemcpyPeerAsync(ctx->x,device,x_src_dev,home_device,
                                            (size_t)D*sizeof(float),ctx->stream),
                        "resident group x p2p")) {
        return 0;
    }
    int resident_timing = resident_timing_ensure(ctx);
    if (resident_timeline_enabled()) {
        ctx->resident_gpu_host_lower_ns = resident_host_clock_ns();
        ctx->resident_gpu_host_upper_ns = 0;
    }
    if (resident_timing && cudaEventRecord(ctx->resident_gpu_start, ctx->stream) != cudaSuccess)
        resident_timing = 0;
    int resident_dp4a = 0;
    int resident_graph_submitted = 0;
    int resident_graph_failed = 0;
    int one_rows[64];
    for (int c = 0; c < count; c++) one_rows[c] = 1;
    int graph_dp4a = resident_graph_enabled() && !resident_timing && any_g4 &&
                     getenv("COLI_CUDA_DP4A") && atoi(getenv("COLI_CUDA_DP4A")) &&
                     ctx->compute_major == 6 && ctx->compute_minor == 1;
    const int graph_debug = getenv("COLI_CUDA_GRAPH_DEBUG") &&
                            atoi(getenv("COLI_CUDA_GRAPH_DEBUG"));
    int graph_reject_fmt = 0;
    if (graph_dp4a) {
        /* The graph captures device pointers. Reserve the maximum resident
         * scratch once so a later larger routed count cannot invalidate a
         * previously captured graph by reallocating qx/qscale. */
        size_t qx_need = (size_t)D + (size_t)64 * (size_t)I;
        size_t qs_need = ((size_t)D + (size_t)64 * (size_t)I) / 64;
        if (!reserve_bytes((void **)&ctx->qx, &ctx->qx_cap, qx_need) ||
            !reserve(&ctx->qscale, &ctx->qscale_cap, qs_need * sizeof(float)))
            graph_dp4a = 0;
        for (int c = 0; graph_dp4a && c < count; c++) {
            ColiCudaTensor *g = gates[c], *u = ups[c], *d = downs[c];
            if (!g || !u || !d || g->fmt != 4 || u->fmt != 4 || d->fmt != 4 ||
                g->gs != 64 || u->gs != 64 || d->gs != 64) {
                graph_dp4a = 0;
                graph_reject_fmt = 1;
            }
        }
    }
    if (graph_debug && getenv("COLI_CUDA_GRAPH") && atoi(getenv("COLI_CUDA_GRAPH")) &&
        !graph_dp4a)
        std::fprintf(stderr, "[cuda-graph] resident graph unavailable: any_g4=%d timing=%d fmt_geometry=%d\n",
                     any_g4, resident_timing, graph_reject_fmt);
    if (graph_dp4a) {
        /* Metadata is dynamic per residency generation, but the graph must
         * not capture the host stack array used by dp4a_meta_prepare().
         * Refresh it before capture/replay; the helper's second call then
         * observes an unchanged table and submits no copy. */
        if (!dp4a_meta_prepare(ctx, gates, ups, downs, count, ctx->stream)) {
            graph_dp4a = 0;
            resident_graph_failed = 1;
        } else if (!cuda_ok(cudaMemcpyAsync(w_dev, weights,
                                            (size_t)count * sizeof(float),
                                            cudaMemcpyHostToDevice, ctx->stream),
                            "resident graph weights")) {
            graph_dp4a = 0;
            resident_graph_failed = 1;
        } else {
            int same = ctx->resident_graph_valid[count] &&
                       ctx->resident_graph_D == D && ctx->resident_graph_I == I &&
                       ctx->resident_graph_exec[count] != nullptr;
            if (ctx->resident_graph_D &&
                (ctx->resident_graph_D != D || ctx->resident_graph_I != I))
                resident_graph_destroy(ctx);
            else if (!same)
                resident_graph_slot_destroy(ctx, count);
            if (same) {
                if (cudaGraphLaunch(ctx->resident_graph_exec[count], ctx->stream) != cudaSuccess) {
                    resident_graph_failed = 1;
                    ctx->resident_graph_fallbacks++;
                    if (graph_debug)
                        std::fprintf(stderr, "[cuda-graph] replay error: %s\n",
                                     cudaGetErrorString(cudaGetLastError()));
                    resident_graph_slot_destroy(ctx, count);
                } else {
                    ctx->resident_graph_launches++;
                    resident_graph_submitted = 1;
                    resident_dp4a = 1;
                }
            } else {
                cudaGraph_t graph = nullptr;
                cudaGraphExec_t exec = nullptr;
                const char *graph_stage = "begin_capture";
                coli_cuda_dp4a_capture_mode(1);
                cudaError_t ce = cudaStreamBeginCapture(
                    ctx->stream, cudaStreamCaptureModeThreadLocal);
                if (ce == cudaSuccess) {
                    graph_stage = "dp4a_group_launch";
                    if (!dp4a_group_launch(ctx, gates, ups, downs, one_rows,
                                           count, D, I, resident_input_dev, 1))
                        ce = cudaErrorInvalidValue;
                }
                if (ce == cudaSuccess) {
                    graph_stage = "weighted_sum";
                    weighted_sum_rows<<<48,256,0,ctx->stream>>>(
                        partial_local, ctx->y, w_dev, count, D);
                    /* cudaGetLastError is not capture-safe.  Capture/graph
                     * instantiation reports an invalid launch; normal
                     * replay errors surface at the later stream sync. */
                    ce = cudaSuccess;
                }
                if (ce == cudaSuccess) graph_stage = "end_capture";
                cudaError_t end = cudaStreamEndCapture(ctx->stream, &graph);
                coli_cuda_dp4a_capture_mode(0);
                if (ce == cudaSuccess) ce = end;
                if (ce == cudaSuccess) {
                    graph_stage = "instantiate";
                    ce = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                }
                if (ce == cudaSuccess) {
                    ctx->resident_graph[count] = graph;
                    ctx->resident_graph_exec[count] = exec;
                    ctx->resident_graph_valid[count] = 1;
                    ctx->resident_graph_D = D;
                    ctx->resident_graph_I = I;
                    ctx->resident_graph_captures++;
                    if (graph_debug)
                        std::fprintf(stderr, "[cuda-graph] captured count=%d capture=%llu\n",
                                     count, (unsigned long long)ctx->resident_graph_captures);
                    if (cudaGraphLaunch(ctx->resident_graph_exec[count], ctx->stream) != cudaSuccess) {
                        resident_graph_failed = 1;
                        ctx->resident_graph_fallbacks++;
                        if (graph_debug)
                            std::fprintf(stderr, "[cuda-graph] first replay error: %s\n",
                                         cudaGetErrorString(cudaGetLastError()));
                        resident_graph_slot_destroy(ctx, count);
                    } else {
                        ctx->resident_graph_launches++;
                        resident_graph_submitted = 1;
                        resident_dp4a = 1;
                    }
                } else {
                    if (exec) cudaGraphExecDestroy(exec);
                    if (graph) cudaGraphDestroy(graph);
                    resident_graph_destroy(ctx);
                    resident_graph_failed = 1;
                    ctx->resident_graph_fallbacks++;
                    if (graph_debug)
                        std::fprintf(stderr, "[cuda-graph] capture error at %s: %s\n",
                                     graph_stage, cudaGetErrorString(ce));
                }
                (void)cudaGetLastError();
            }
        }
    }
    if (any_g4 && getenv("COLI_CUDA_DP4A") && atoi(getenv("COLI_CUDA_DP4A")) &&
        ctx->compute_major == 6 && ctx->compute_minor == 1) {
        if (!resident_dp4a && (!graph_dp4a || resident_graph_failed))
            resident_dp4a = dp4a_group_launch(ctx, gates, ups, downs, one_rows,
                                               count, D, I, resident_input_dev, 1);
        if (resident_dp4a) {
            if (graph_debug && resident_graph_submitted) {
                static int graph_dispatch_reported = 0;
                if (!graph_dispatch_reported) {
                    graph_dispatch_reported = 1;
                    std::fprintf(stderr, "[cuda-graph] resident graph dispatch: count=%d device=%d\n",
                                 count, device);
                }
            }
            if (!getenv("COLI_CUDA_DP4A_QUIET")) {
                if (resident_graph_submitted)
                    std::fprintf(stderr, "[dp4a] resident graph dispatch: %d experts on GPU%d\n",
                                 count, device);
                else
                    std::fprintf(stderr, "[dp4a] resident dispatch: %d experts on GPU%d\n",
                                 count, device);
            }
            if (!resident_graph_submitted) {
                if (!cuda_ok(cudaMemcpyAsync(w_dev, weights, (size_t)count*sizeof(float),
                                             cudaMemcpyHostToDevice, ctx->stream),
                              "resident DP4A weights")) return 0;
                weighted_sum_rows<<<48,256,0,ctx->stream>>>(partial_local,ctx->y,w_dev,count,D);
            }
        }
    }
    if (!resident_dp4a) {
        if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                    cudaMemcpyHostToDevice,ctx->stream),"resident group desc")||
           !cuda_ok(cudaMemcpyAsync(w_dev,weights,(size_t)count*sizeof(float),
                                    cudaMemcpyHostToDevice,ctx->stream),"resident group weights"))
            return 0;
        bcast_row<<<64,256,0,ctx->stream>>>(ctx->x,ctx->x,count,D);   /* row 0 -> rows 1..count-1 (in-place safe: row 0 rewritten with itself) */
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,1,(unsigned)count),og((unsigned)D,1,(unsigned)count);
        if(any_g4){
            grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
            grouped_down_g4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        }else{
            grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
            grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        }
        weighted_sum_rows<<<48,256,0,ctx->stream>>>(partial_local,ctx->y,w_dev,count,D);
    }
    if (!resident_graph_submitted) {
        if (resident_timing && cudaEventRecord(ctx->resident_gpu_end, ctx->stream) != cudaSuccess)
            resident_timing = 0;
    }
    if(!cuda_ok(cudaMemcpyPeerAsync(partial_slot_dev,home_device,partial_local,device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident partial p2p"))
        return 0;
    if (resident_timing) ctx->resident_gpu_timing_pending = 1;
    if(!cuda_ok(cudaEventRecord(ctx->ev_done,ctx->stream),"resident event record")) return 0;
    return cuda_ok(cudaGetLastError(),"resident group launch");
}
extern "C" int coli_cuda_expert_group_resident_take(int home_device,const int *devices,int n_issued,
                                           float *slots_dev,float *acc_dev,int D){
    if(fault_injected()) return 0;
    if(n_issued<1||!slots_dev||!acc_dev||D<1) return 0;
    DeviceContext *home=find_ctx(home_device); if(!select_ctx(home)) return 0;
    for(int i=0;i<n_issued;i++){
        DeviceContext *src=find_ctx(devices[i]);
        if(!src||!src->ev_done_ok) return 0;
        if(!cuda_ok(cudaStreamWaitEvent(0,src->ev_done,0),"resident take wait")) return 0;
    }
    int resident_timing = resident_timing_ensure(home);
    if (resident_timeline_enabled()) {
        home->resident_reduce_host_lower_ns = 0;
        home->resident_reduce_host_upper_ns = 0;
    }
    if (resident_timing && cudaEventRecord(home->resident_reduce_start, 0) != cudaSuccess)
        resident_timing = 0;
    /* For one active island the sole slot is already the final reduction.
     * The alias is installed by qwen36_tier only for this case. Keep the
     * event envelope intact for diagnostics, but do not launch a kernel whose
     * sole operation is `dst[i] = slots[i]`. */
    if (n_issued == 1 && slots_dev == acc_dev) {
        if (resident_timing && cudaEventRecord(home->resident_reduce_end, 0) != cudaSuccess)
            resident_timing = 0;
        if (resident_timeline_enabled()) {
            home->resident_reduce_host_lower_ns=resident_host_clock_ns();
            home->resident_reduce_host_upper_ns=0;
        }
        if (resident_timing) home->resident_reduce_timing_pending = 1;
        return cuda_ok(cudaGetLastError(),"resident take direct");
    }
    sum_slots<<<48,256>>>(acc_dev,slots_dev,n_issued,D);          /* legacy stream: ordered with pipe_* */
    if (resident_timing && cudaEventRecord(home->resident_reduce_end, 0) != cudaSuccess)
        resident_timing = 0;
    if (resident_timeline_enabled()) {
        home->resident_reduce_host_lower_ns=resident_host_clock_ns();
        home->resident_reduce_host_upper_ns=0;
    }
    if (resident_timing) home->resident_reduce_timing_pending = 1;
    return cuda_ok(cudaGetLastError(),"resident take reduce");
}
extern "C" int coli_cuda_expert_group_resident_sync(int home_device) {
    DeviceContext *home=find_ctx(home_device);
    if(!home||!select_ctx(home)) return 0;
    /* resident_take enqueues the cross-device waits and home reduction on the
     * legacy stream. This exposes the host wait formerly hidden in D2H. */
    uint64_t before=resident_timeline_enabled()?resident_host_clock_ns():0;
    int ok=cuda_ok(cudaStreamSynchronize(0),"resident take sync");
    uint64_t after=resident_timeline_enabled()?resident_host_clock_ns():0;
    if (resident_timeline_enabled()) {
        if (!home->resident_reduce_host_lower_ns ||
            before<home->resident_reduce_host_lower_ns)
            home->resident_reduce_host_lower_ns=before;
        home->resident_reduce_host_upper_ns=after;
    }
    cupti_trace_checkpoint();
    return ok;
}
extern "C" int coli_cuda_expert_group_resident_timing(
        int home_device,const int *devices,int n_issued,
        double *gpu_ms,double *reduce_ms) {
    double gpu=0.0, reduce=0.0;
    int have_gpu=0, have_reduce=0;
    if (devices && n_issued > 0) for (int i=0;i<n_issued;i++) {
        DeviceContext *src=find_ctx(devices[i]);
        if(!src||!src->resident_gpu_timing_pending||!select_ctx(src)) continue;
        float ms=0.f;
        if(cudaEventElapsedTime(&ms,src->resident_gpu_start,src->resident_gpu_end)==cudaSuccess){
            /* Devices run in parallel; report the critical-path maximum. */
            if(!have_gpu || (double)ms>gpu) gpu=(double)ms;
            have_gpu=1;
        }
        src->resident_gpu_timing_pending=0;
    }
    DeviceContext *home=find_ctx(home_device);
    if(home&&home->resident_reduce_timing_pending&&select_ctx(home)){
        float ms=0.f;
        if(cudaEventElapsedTime(&ms,home->resident_reduce_start,home->resident_reduce_end)==cudaSuccess){
            reduce=(double)ms; have_reduce=1;
        }
        home->resident_reduce_timing_pending=0;
    }
    if(gpu_ms) *gpu_ms=gpu;
    if(reduce_ms) *reduce_ms=reduce;
    return have_gpu || have_reduce;
}
extern "C" int coli_cuda_expert_group_resident_host_timing(
        int home_device,const int *devices,int n_issued,
        uint64_t *gpu_lower_ns,uint64_t *gpu_upper_ns,
        uint64_t *reduce_lower_ns,uint64_t *reduce_upper_ns) {
    uint64_t gpu_lower = 0, gpu_upper = 0;
    const uint64_t host_upper_stamp = resident_timeline_enabled() ? resident_host_clock_ns() : 0;
    if (devices && n_issued > 0) for (int i = 0; i < n_issued; i++) {
        DeviceContext *src = find_ctx(devices[i]);
        if (!src) continue;
        if (src->resident_gpu_host_lower_ns > gpu_lower)
            gpu_lower=src->resident_gpu_host_lower_ns;
        if (src->resident_gpu_host_upper_ns)
            gpu_upper=src->resident_gpu_host_upper_ns>gpu_upper?
                      src->resident_gpu_host_upper_ns:gpu_upper;
        else if (host_upper_stamp>gpu_upper)
            gpu_upper=host_upper_stamp;
    }
    uint64_t reduce_lower = 0, reduce_upper = 0;
    DeviceContext *home = find_ctx(home_device);
    if (home) {
        reduce_lower=home->resident_reduce_host_lower_ns;
        reduce_upper=home->resident_reduce_host_upper_ns;
    }
    if (gpu_lower_ns) *gpu_lower_ns = gpu_lower;
    if (gpu_upper_ns) *gpu_upper_ns = gpu_upper;
    if (reduce_lower_ns) *reduce_lower_ns = reduce_lower;
    if (reduce_upper_ns) *reduce_upper_ns = reduce_upper;
    return gpu_lower != 0 || gpu_upper != 0 || reduce_lower != 0 || reduce_upper != 0;
}
extern "C" int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                                     int spitch,int width,int height){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy2D(dst,(size_t)dpitch*4,src,(size_t)spitch*4,
        (size_t)width*4,height,cudaMemcpyDeviceToDevice),"pipe copy2d");
}
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
extern "C" int coli_cuda_attention_project_batch_dev(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"pipe attention download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(gate_dev,up_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    pipe_add_n<<<(unsigned)((n+255)/256),256>>>(x_dev,t_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rows_add<<<nrows,256>>>(x_dev,partial_dev,rows_dev,D);
    return cuda_ok(cudaGetLastError(),"pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S){
    if (fault_injected()) return 0;
    if(!t||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    dim3 grid((unsigned)t->O,(unsigned)S);
    quant_matmul<<<grid,256>>>(y_dev,x_dev,t->weights,t->scales,t->fmt,S,t->I,t->O,
        row_bytes(t->fmt,t->I),t->gs,t->ng);
    return cuda_ok(cudaGetLastError(),"pipe gemm");
}
/* Coarse dense-island submission: one host input is shared by a small set of
 * independent projections.  Keep the individual quant_matmul launches (and
 * therefore their reduction order) intact; only move the H2D/D2H and stream
 * bookkeeping to this backend boundary.  This is intentionally synchronous
 * for the first spike: qwen36's DeltaNet recurrence consumes qkv/z on the
 * host immediately after this call. */
extern "C" int coli_cuda_pipe_dense_batch(ColiCudaTensor *const *tensors,
                                           const int *out_offsets,int count,
                                           int input_dim,const float *x_host,
                                           float *out_host,int total_out,
                                           int device){
    if (fault_injected() || !tensors || !out_offsets || !x_host || !out_host ||
        count < 1 || count > 16 || input_dim < 1 || total_out < 1) return 0;
    DeviceContext *ctx=find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    for (int i=0;i<count;i++) {
        ColiCudaTensor *t=tensors[i];
        if (!t || t->device!=device || t->I!=input_dim || t->O<1 ||
            out_offsets[i]<0 || out_offsets[i]+t->O>total_out) return 0;
        if (i && out_offsets[i] < out_offsets[i-1]) return 0;
    }
    float *xd=coli_cuda_pipe_scratch(device,20,(size_t)input_dim*sizeof(float));
    float *yd=coli_cuda_pipe_scratch(device,21,(size_t)total_out*sizeof(float));
    if (!xd || !yd) return 0;
    if (!cuda_ok(cudaMemcpyAsync(xd,x_host,(size_t)input_dim*sizeof(float),
                                 cudaMemcpyHostToDevice,ctx->stream),
                 "dense batch input upload")) return 0;
    for (int i=0;i<count;i++) {
        ColiCudaTensor *t=tensors[i];
        dim3 grid((unsigned)t->O,1);
        if(t->fmt==1 && (t->I&31)==0 && getenv("COLI_DENSE_EXACT") && atoi(getenv("COLI_DENSE_EXACT")))
            dense_i8_matmul_exact<<<(unsigned)((t->O+127)/128),128,0,ctx->stream>>>(yd+out_offsets[i],xd,
                (const int8_t*)t->weights,t->scales,t->I,t->O);
        else if(t->fmt==1 && (t->I&31)==0)
            dense_i8_matmul_cpu_order<<<grid,32,0,ctx->stream>>>(yd+out_offsets[i],xd,
                (const int8_t*)t->weights,t->scales,t->I,t->O);
        else
            quant_matmul<<<grid,256,0,ctx->stream>>>(yd+out_offsets[i],xd,
                t->weights,t->scales,t->fmt,1,t->I,t->O,
                row_bytes(t->fmt,t->I),t->gs,t->ng);
        if (!cuda_ok(cudaGetLastError(),"dense batch projection launch")) return 0;
    }
    if (!cuda_ok(cudaMemcpyAsync(out_host,yd,(size_t)total_out*sizeof(float),
                                 cudaMemcpyDeviceToHost,ctx->stream),
                 "dense batch output download")) return 0;
    return cuda_ok(cudaStreamSynchronize(ctx->stream),"dense batch sync");
}

/* Coarse shared-MLP island submission.  This deliberately preserves the
 * validated projection boundaries and their reduction order.  Only the host
 * call boundary changes.  The reference qwen36 path performs SiLU with the
 * host libm, so this first exactness-preserving prototype deliberately keeps
 * that operation on the host inside this backend call.  A GPU-SiLU variant is
 * not acceptable here: small expf differences accumulate through GDN. */
extern "C" int coli_cuda_pipe_dense_mlp(
        ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down,
        const float *x_host, float *out_host,
        int input_dim, int intermediate_dim, int output_dim, int device){
    if (fault_injected() || !gate || !up || !down || !x_host || !out_host ||
        input_dim < 1 || intermediate_dim < 1 || output_dim < 1) return 0;
    DeviceContext *ctx=find_ctx(device);
    if (!select_ctx(ctx) || gate->device!=device || up->device!=device ||
        down->device!=device || gate->I!=input_dim || up->I!=input_dim ||
        gate->O!=intermediate_dim || up->O!=intermediate_dim ||
        down->I!=intermediate_dim || down->O!=output_dim) return 0;

    /* 27..29 are dedicated to this executor.  In particular, do not use
     * slots 25/26: the resident expert hub/accumulator owns those slots while
     * routed GPU work is in flight. */
    float *xd=coli_cuda_pipe_scratch(device,27,(size_t)input_dim*sizeof(float));
    float *gu=coli_cuda_pipe_scratch(device,28,(size_t)intermediate_dim*2*sizeof(float));
    float *yd=coli_cuda_pipe_scratch(device,29,(size_t)output_dim*sizeof(float));
    if (!xd || !gu || !yd) return 0;
    if (!cuda_ok(cudaMemcpyAsync(xd,x_host,(size_t)input_dim*sizeof(float),
                                 cudaMemcpyHostToDevice,ctx->stream),
                 "dense mlp input upload")) return 0;

    ColiCudaTensor *proj[2]={gate,up};
    float *dst[2]={gu,gu+intermediate_dim};
    for (int i=0;i<2;i++) {
        ColiCudaTensor *t=proj[i];
        dim3 grid((unsigned)t->O,1);
        if (t->fmt==1 && (t->I&31)==0 && getenv("COLI_DENSE_EXACT") && atoi(getenv("COLI_DENSE_EXACT")))
            dense_i8_matmul_exact<<<(unsigned)((t->O+127)/128),128,0,ctx->stream>>>(dst[i],xd,
                (const int8_t*)t->weights,t->scales,t->I,t->O);
        else if (t->fmt==1 && (t->I&31)==0)
            dense_i8_matmul_cpu_order<<<grid,32,0,ctx->stream>>>(dst[i],xd,
                (const int8_t*)t->weights,t->scales,t->I,t->O);
        else
            quant_matmul<<<grid,256,0,ctx->stream>>>(dst[i],xd,t->weights,t->scales,
                t->fmt,1,t->I,t->O,row_bytes(t->fmt,t->I),t->gs,t->ng);
        if (!cuda_ok(cudaGetLastError(),"dense mlp projection launch")) return 0;
    }
    /* Preserve the established CPU SiLU semantics.  This adds one internal
     * staging round trip, but keeps the prototype numerically comparable to
     * qt_dense_shared(), which is the invariant this architectural spike is
     * testing. */
    if (!reserve_pinned(&ctx->host_kv,&ctx->host_kv_cap,
                        (size_t)intermediate_dim*2*sizeof(float)) ||
        !cuda_ok(cudaMemcpyAsync(ctx->host_kv,gu,
                                 (size_t)intermediate_dim*2*sizeof(float),
                                 cudaMemcpyDeviceToHost,ctx->stream),
                 "dense mlp activation download") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream),"dense mlp activation sync")) return 0;
    for (int i=0;i<intermediate_dim;i++) {
        float v=ctx->host_kv[i];
        ctx->host_kv[i]=(v/(1.0f+expf(-v)))*ctx->host_kv[intermediate_dim+i];
    }
    if (!cuda_ok(cudaMemcpyAsync(gu,ctx->host_kv,
                                 (size_t)intermediate_dim*sizeof(float),
                                 cudaMemcpyHostToDevice,ctx->stream),
                 "dense mlp activation upload")) return 0;

    dim3 dgrid((unsigned)down->O,1);
    if (down->fmt==1 && (down->I&31)==0 && getenv("COLI_DENSE_EXACT") && atoi(getenv("COLI_DENSE_EXACT")))
        dense_i8_matmul_exact<<<(unsigned)((down->O+127)/128),128,0,ctx->stream>>>(yd,gu,
            (const int8_t*)down->weights,down->scales,down->I,down->O);
    else if (down->fmt==1 && (down->I&31)==0)
        dense_i8_matmul_cpu_order<<<dgrid,32,0,ctx->stream>>>(yd,gu,
            (const int8_t*)down->weights,down->scales,down->I,down->O);
    else
        quant_matmul<<<dgrid,256,0,ctx->stream>>>(yd,gu,down->weights,down->scales,
            down->fmt,1,down->I,down->O,row_bytes(down->fmt,down->I),down->gs,down->ng);
    if (!cuda_ok(cudaGetLastError(),"dense mlp down launch") ||
        !cuda_ok(cudaMemcpyAsync(out_host,yd,(size_t)output_dim*sizeof(float),
                                 cudaMemcpyDeviceToHost,ctx->stream),
                 "dense mlp output download") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream),"dense mlp sync")) return 0;
    return 1;
}

static float *dn_buf(DeviceContext *ctx,int slot,size_t bytes){
    if(slot<0 || slot>=12 || !reserve(&ctx->dn_buf[slot],&ctx->dn_cap[slot],bytes)) return NULL;
    return ctx->dn_buf[slot];
}

/* Disabled unless explicitly requested.  This captures the first complete
 * device DeltaNet call in a self-describing binary record so parity debugging
 * can compare components without changing the production executor. */
static void dn_full_debug_dump(DeviceContext *ctx,
                               const float *conv,const float *q,const float *k,
                               const float *outv,const float *z,const float *outr,
                               const float *out,const float *b,const float *a,
                               int conv_dim,int qn,int value_dim,int hidden,
                               int vheads){
    static int done=0;
    const char *path=getenv("COLI_DENSE_FULL_DBG");
    if(!path || done || !ctx) return;
    done=1;
    FILE *f=std::fopen(path,"wb");
    if(!f) return;
    uint32_t h[8]={0x444e4442u,1u,(uint32_t)conv_dim,(uint32_t)qn,
                   (uint32_t)value_dim,(uint32_t)hidden,(uint32_t)vheads,0u};
    std::fwrite(h,sizeof h,1,f);
    const struct { const float *p; size_t n; } aout[] = {
        {conv,(size_t)conv_dim},{q,(size_t)qn},{k,(size_t)qn},
        {outv,(size_t)value_dim},{z,(size_t)value_dim},{outr,(size_t)value_dim},
        {out,(size_t)hidden},{b,(size_t)vheads},{a,(size_t)vheads}
    };
    float *host=(float*)std::malloc((size_t)std::max(conv_dim,
                         std::max(qn,std::max(value_dim,hidden)))*sizeof(float));
    if(!host){ std::fclose(f); return; }
    for(const auto &x:aout){
        if(cudaMemcpy(host,x.p,x.n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
            break;
        std::fwrite(host,sizeof(float),x.n,f);
    }
    std::free(host);
    std::fclose(f);
}

extern "C" int coli_cuda_pipe_deltanet_layer(
        ColiCudaTensor *qkv, ColiCudaTensor *z, ColiCudaTensor *out,
        const float *conv_w_dev, const float *b_w_dev, const float *a_w_dev,
        const float *dtbias_dev, const float *alog_dev, const float *norm_dev,
        float *rec_dev, float *ring_dev,
        const float *x_host, float *out_host,
        int hidden, int vheads, int kheads, int kdim, int vdim,
        int convk, int conv_dim, float eps, int device){
    if(fault_injected() || !qkv || !z || !out || !conv_w_dev || !b_w_dev ||
       !a_w_dev || !dtbias_dev || !alog_dev || !norm_dev || !rec_dev ||
       !ring_dev || !x_host || !out_host || hidden<1 || vheads<1 ||
       kheads<1 || vheads%kheads || kdim<1 || vdim<1 || convk<2 ||
       conv_dim != 2*kheads*kdim + vheads*vdim || eps<=0.f) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(!select_ctx(ctx) || qkv->device!=device || z->device!=device ||
       out->device!=device || qkv->fmt!=1 || z->fmt!=1 || out->fmt!=1 ||
       qkv->I!=hidden || z->I!=hidden || out->I!=vheads*vdim ||
       qkv->O!=conv_dim || z->O!=vheads*vdim || out->O!=hidden) return 0;
    float *xd=dn_buf(ctx,0,(size_t)hidden*sizeof(float));
    float *qkv_d=dn_buf(ctx,1,(size_t)conv_dim*sizeof(float));
    float *z_d=dn_buf(ctx,2,(size_t)vheads*vdim*sizeof(float));
    float *b_d=dn_buf(ctx,3,(size_t)vheads*sizeof(float));
    float *a_d=dn_buf(ctx,4,(size_t)vheads*sizeof(float));
    float *co=dn_buf(ctx,5,(size_t)conv_dim*sizeof(float));
    float *q_d=dn_buf(ctx,6,(size_t)vheads*kdim*sizeof(float));
    float *k_d=dn_buf(ctx,7,(size_t)vheads*kdim*sizeof(float));
    float *ov=dn_buf(ctx,8,(size_t)vheads*vdim*sizeof(float));
    float *or_=dn_buf(ctx,9,(size_t)vheads*vdim*sizeof(float));
    float *kv=dn_buf(ctx,10,(size_t)vheads*vdim*sizeof(float));
    float *dl=dn_buf(ctx,11,(size_t)vheads*vdim*sizeof(float));
    if(!xd||!qkv_d||!z_d||!b_d||!a_d||!co||!q_d||!k_d||!ov||!or_||!kv||!dl) return 0;
    if(!cuda_ok(cudaMemcpyAsync(xd,x_host,(size_t)hidden*sizeof(float),
                                cudaMemcpyHostToDevice,ctx->stream),"deltanet input upload")) return 0;
    dim3 qgrid((unsigned)qkv->O), zgrid((unsigned)z->O), ogrid((unsigned)out->O);
    if (getenv("COLI_DENSE_EXACT") && atoi(getenv("COLI_DENSE_EXACT")))
        dense_i8_matmul_exact<<<(unsigned)((qkv->O+127)/128),128,0,ctx->stream>>>(qkv_d,xd,
            (const int8_t*)qkv->weights,qkv->scales,qkv->I,qkv->O);
    else
        dense_i8_matmul_cpu_order<<<qgrid,32,0,ctx->stream>>>(qkv_d,xd,
            (const int8_t*)qkv->weights,qkv->scales,qkv->I,qkv->O);
    if(!cuda_ok(cudaGetLastError(),"deltanet qkv launch")) return 0;
    if (getenv("COLI_DENSE_EXACT") && atoi(getenv("COLI_DENSE_EXACT")))
        dense_i8_matmul_exact<<<(unsigned)((z->O+127)/128),128,0,ctx->stream>>>(z_d,xd,
            (const int8_t*)z->weights,z->scales,z->I,z->O);
    else
        dense_i8_matmul_cpu_order<<<zgrid,32,0,ctx->stream>>>(z_d,xd,
            (const int8_t*)z->weights,z->scales,z->I,z->O);
    if(!cuda_ok(cudaGetLastError(),"deltanet z launch")) return 0;
    dn_f32_ba_kernel<<<(unsigned)((vheads+127)/128),128,0,ctx->stream>>>(
        b_d,a_d,b_w_dev,a_w_dev,xd,hidden,vheads);
    if(!cuda_ok(cudaGetLastError(),"deltanet b/a launch")) return 0;
    dn_conv_kernel<<<1,256,0,ctx->stream>>>(co,ring_dev,conv_w_dev,qkv_d,conv_dim,convk);
    if(!cuda_ok(cudaGetLastError(),"deltanet conv launch")) return 0;
    if (getenv("COLI_DENSE_FULL_PARALLEL") && atoi(getenv("COLI_DENSE_FULL_PARALLEL")))
        dn_state_kernel<<<(unsigned)vheads,256,0,ctx->stream>>>(ov,q_d,k_d,kv,dl,rec_dev,co,
            b_d,a_d,dtbias_dev,alog_dev,vheads,kheads,kdim,vdim,conv_dim);
    else
        dn_state_kernel_scalar<<<(unsigned)vheads,1,0,ctx->stream>>>(ov,q_d,k_d,kv,dl,rec_dev,co,
            b_d,a_d,dtbias_dev,alog_dev,vheads,kheads,kdim,vdim,conv_dim);
    if(!cuda_ok(cudaGetLastError(),"deltanet state launch")) return 0;
    if (getenv("COLI_DENSE_FULL_PARALLEL") && atoi(getenv("COLI_DENSE_FULL_PARALLEL")))
        dn_norm_kernel<<<(unsigned)vheads,256,0,ctx->stream>>>(or_,ov,z_d,norm_dev,vheads,vdim,eps);
    else
        dn_norm_kernel_scalar<<<(unsigned)vheads,1,0,ctx->stream>>>(or_,ov,z_d,norm_dev,vheads,vdim,eps);
    if(!cuda_ok(cudaGetLastError(),"deltanet norm launch")) return 0;
    float *dn_out=dn_buf(ctx,0,(size_t)hidden*sizeof(float));
    if (getenv("COLI_DENSE_EXACT") && atoi(getenv("COLI_DENSE_EXACT")))
        dense_i8_matmul_exact<<<(unsigned)((out->O+127)/128),128,0,ctx->stream>>>(dn_out,
            or_,(const int8_t*)out->weights,out->scales,out->I,out->O);
    else
        dense_i8_matmul_cpu_order<<<ogrid,32,0,ctx->stream>>>(dn_out,
            or_,(const int8_t*)out->weights,out->scales,out->I,out->O);
    if(!cuda_ok(cudaGetLastError(),"deltanet out launch")) return 0;
    float *out_d=dn_out;
    if(!cuda_ok(cudaMemcpyAsync(out_host,out_d,(size_t)hidden*sizeof(float),
                                cudaMemcpyDeviceToHost,ctx->stream),"deltanet output download")) return 0;
    if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"deltanet layer sync")) return 0;
    dn_full_debug_dump(ctx,co,q_d,k_d,ov,z_d,or_,out_d,b_d,a_d,
                       conv_dim,vheads*kdim,vheads*vdim,hidden,vheads);
    return 1;
}

extern "C" int coli_cuda_pipe_deltanet_state(
        const float *qkv_host, const float *z_host,
        const float *b_host, const float *a_host,
        const float *conv_w_dev, const float *dtbias_dev,
        const float *alog_dev, const float *norm_dev,
        float *rec_dev, float *ring_dev,
        float *norm_out_host,
        int vheads, int kheads, int kdim, int vdim,
        int convk, int conv_dim, float eps, int device){
    if (fault_injected() || !qkv_host || !z_host || !b_host || !a_host ||
        !conv_w_dev || !dtbias_dev || !alog_dev || !norm_dev || !rec_dev ||
        !ring_dev || !norm_out_host || vheads<1 || kheads<1 || vheads%kheads ||
        kdim<1 || vdim<1 || convk<2 || conv_dim != 2*kheads*kdim + vheads*vdim ||
        eps<=0.f) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(!select_ctx(ctx)) return 0;
    float *qkv_d=dn_buf(ctx,1,(size_t)conv_dim*sizeof(float));
    float *z_d=dn_buf(ctx,2,(size_t)vheads*vdim*sizeof(float));
    float *b_d=dn_buf(ctx,3,(size_t)vheads*sizeof(float));
    float *a_d=dn_buf(ctx,4,(size_t)vheads*sizeof(float));
    float *co=dn_buf(ctx,5,(size_t)conv_dim*sizeof(float));
    float *q_d=dn_buf(ctx,6,(size_t)vheads*kdim*sizeof(float));
    float *k_d=dn_buf(ctx,7,(size_t)vheads*kdim*sizeof(float));
    float *ov=dn_buf(ctx,8,(size_t)vheads*vdim*sizeof(float));
    float *or_=dn_buf(ctx,9,(size_t)vheads*vdim*sizeof(float));
    float *kv=dn_buf(ctx,10,(size_t)vheads*vdim*sizeof(float));
    float *dl=dn_buf(ctx,11,(size_t)vheads*vdim*sizeof(float));
    if(!qkv_d||!z_d||!b_d||!a_d||!co||!q_d||!k_d||!ov||!or_||!kv||!dl) return 0;
    cudaStream_t stream=ctx->stream;
    if(!cuda_ok(cudaMemcpyAsync(qkv_d,qkv_host,(size_t)conv_dim*sizeof(float),
                                cudaMemcpyHostToDevice,stream),"deltanet state qkv upload")||
       !cuda_ok(cudaMemcpyAsync(z_d,z_host,(size_t)vheads*vdim*sizeof(float),
                                cudaMemcpyHostToDevice,stream),"deltanet state z upload")||
       !cuda_ok(cudaMemcpyAsync(b_d,b_host,(size_t)vheads*sizeof(float),
                                cudaMemcpyHostToDevice,stream),"deltanet state b upload")||
       !cuda_ok(cudaMemcpyAsync(a_d,a_host,(size_t)vheads*sizeof(float),
                                cudaMemcpyHostToDevice,stream),"deltanet state a upload")) return 0;
    dn_conv_kernel<<<1,256,0,stream>>>(co,ring_dev,conv_w_dev,qkv_d,conv_dim,convk);
    if(!cuda_ok(cudaGetLastError(),"deltanet state conv launch")) return 0;
    /* Correctness-first default: one thread owns each head.  The opt-in
     * parallel variant gives independent value elements to different threads;
     * every value element still performs its kk loops in the same order. */
    int parallel = getenv("COLI_DENSE_STATE_PARALLEL") &&
                   atoi(getenv("COLI_DENSE_STATE_PARALLEL"));
    if (parallel)
        dn_state_kernel<<<(unsigned)vheads,256,0,stream>>>(ov,q_d,k_d,kv,dl,rec_dev,co,
            b_d,a_d,dtbias_dev,alog_dev,vheads,kheads,kdim,vdim,conv_dim);
    else
        dn_state_kernel_scalar<<<(unsigned)vheads,1,0,stream>>>(ov,q_d,k_d,kv,dl,rec_dev,co,
            b_d,a_d,dtbias_dev,alog_dev,vheads,kheads,kdim,vdim,conv_dim);
    if(!cuda_ok(cudaGetLastError(),"deltanet state recurrent launch")) return 0;
    if (parallel)
        dn_norm_kernel<<<(unsigned)vheads,256,0,stream>>>(or_,ov,z_d,norm_dev,
            vheads,vdim,eps);
    else
        dn_norm_kernel_scalar<<<(unsigned)vheads,1,0,stream>>>(or_,ov,z_d,norm_dev,
            vheads,vdim,eps);
    if(!cuda_ok(cudaGetLastError(),"deltanet state norm launch")) return 0;
    if(!cuda_ok(cudaMemcpyAsync(norm_out_host,or_,
                                (size_t)vheads*vdim*sizeof(float),
                                cudaMemcpyDeviceToHost,stream),
                "deltanet state output download")) return 0;
    return cuda_ok(cudaStreamSynchronize(stream),"deltanet state sync");
}
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                                        const float *src,size_t bytes){
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),"pipe intra copy"); }
    return cuda_ok(cudaMemcpyPeer(dst,dst_dev,src,src_dev,bytes),"pipe peer copy");
}
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch (dev out)"))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(out_dev,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch (dev out)"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync (dev out)");
}
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(ctx_dev,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe shard attention launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe shard attention sync");
}
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev q upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,1),256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,1,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"kvdev absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev absorb sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_sync(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaDeviceSynchronize(),"pipe sync");
}
