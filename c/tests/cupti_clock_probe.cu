/* Standalone CUPTI clock/queue probe.
 *
 * This intentionally does not link into Qwen. It answers the measurement
 * question first: are CUPTI activity timestamps stable against the host
 * monotonic clock, and do they expose queued/submitted GPU work without
 * inserting callbacks into the application stream?
 */
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_activity.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

struct KernelRecord {
    uint64_t start, end, queued, submitted;
    uint32_t stream, correlation;
    std::string name;
};

static std::mutex g_records_mx;
static std::vector<KernelRecord> g_records;
static size_t g_buffer_size = 1u << 20;

static uint64_t host_now_ns() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()).count();
}

static void die_cuda(cudaError_t rc, const char *what) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(rc));
        std::exit(2);
    }
}

static void die_cupti(CUptiResult rc, const char *what) {
    if (rc != CUPTI_SUCCESS) {
        const char *name = nullptr;
        cuptiGetResultString(rc, &name);
        std::fprintf(stderr, "CUPTI %s: %s\n", what, name ? name : "unknown");
        std::exit(3);
    }
}

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                       size_t *max_records) {
    void *p = std::malloc(g_buffer_size + 8);
    if (!p) { *buffer = nullptr; *size = 0; *max_records = 0; return; }
    uintptr_t aligned = ((uintptr_t)p + 7u) & ~(uintptr_t)7u;
    *buffer = (uint8_t *)aligned;
    *size = g_buffer_size;
    *max_records = 0;
}

static void CUPTIAPI buffer_completed(CUcontext, uint32_t stream_id,
                                       uint8_t *buffer, size_t,
                                       size_t valid_size) {
    if (buffer && valid_size) {
        uint8_t *cursor = buffer;
        CUpti_Activity *record = nullptr;
        while (cuptiActivityGetNextRecord(cursor, valid_size, &record) == CUPTI_SUCCESS) {
            if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                const CUpti_ActivityKernel9 *kernel =
                    reinterpret_cast<const CUpti_ActivityKernel9 *>(record);
                KernelRecord item{};
                item.start = kernel->start;
                item.end = kernel->end;
                item.queued = kernel->queued;
                item.submitted = kernel->submitted;
                item.stream = stream_id;
                item.correlation = kernel->correlationId;
                item.name = kernel->name ? kernel->name : "";
                std::lock_guard<std::mutex> lock(g_records_mx);
                g_records.push_back(std::move(item));
            }
        }
    }
    if (buffer) {
        /* CUPTI requires the original allocation. The aligned pointer is at
         * most 7 bytes into it, which is sufficient to recover it here. */
        uintptr_t aligned = (uintptr_t)buffer;
        std::free((void *)(aligned & ~(uintptr_t)7u));
    }
}

__global__ static void probe_kernel(float *value, int rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = value[i & 255];
    for (int r = 0; r < rounds; r++) x = x * 1.000001f + 0.000001f;
    value[i & 255] = x;
}

static uint64_t cupti_now() {
    uint64_t value = 0;
    die_cupti(cuptiGetTimestamp(&value), "get timestamp");
    return value;
}

int main(int argc, char **argv) {
    const char *out_path = argc > 1 ? argv[1] : "tests/cupti_clock_probe.csv";
    die_cupti(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed),
              "register activity callbacks");
    die_cupti(cuptiActivityEnableLatencyTimestamps(1), "enable latency timestamps");
    die_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
              "enable concurrent kernel activity");

    std::vector<int64_t> offsets;
    for (int i = 0; i < 32; i++) {
        uint64_t c0 = cupti_now();
        uint64_t h0 = host_now_ns();
        uint64_t h1 = host_now_ns();
        uint64_t c1 = cupti_now();
        offsets.push_back((int64_t)((h0 + h1) / 2) - (int64_t)((c0 + c1) / 2));
    }
    std::sort(offsets.begin(), offsets.end());
    int64_t offset = offsets[offsets.size() / 2];
    auto spread = offsets.back() - offsets.front();

    float *device = nullptr;
    die_cuda(cudaMalloc(&device, 256 * sizeof(float)), "malloc");
    cudaStream_t stream = nullptr;
    die_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    die_cuda(cudaMemsetAsync(device, 0, 256 * sizeof(float), stream), "memset");
    die_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

    const int launches = 128;
    uint64_t host_issue = host_now_ns();
    uint64_t cupti_issue = cupti_now();
    for (int i = 0; i < launches; i++)
        probe_kernel<<<32, 256, 0, stream>>>(device, 64);
    die_cuda(cudaGetLastError(), "kernel launch");
    uint64_t host_sync_before = host_now_ns();
    uint64_t cupti_sync_before = cupti_now();
    die_cuda(cudaStreamSynchronize(stream), "measurement synchronize");
    uint64_t host_sync_after = host_now_ns();
    uint64_t cupti_sync_after = cupti_now();
    die_cupti(cuptiActivityFlushAll(1), "flush activities");

    std::vector<KernelRecord> records;
    {
        std::lock_guard<std::mutex> lock(g_records_mx);
        records = g_records;
    }
    std::sort(records.begin(), records.end(),
              [](const KernelRecord &a, const KernelRecord &b) { return a.start < b.start; });

    std::ofstream file(out_path, std::ios::binary);
    if (!file) { std::fprintf(stderr, "cannot write %s\n", out_path); return 4; }
    file << "record,name,cupti_start_ns,cupti_end_ns,host_start_ns,host_end_ns,"
            "queued_ns,submitted_ns,stream,correlation\n";
    for (size_t i = 0; i < records.size(); i++) {
        const auto &r = records[i];
        file << i << ",\"" << r.name << "\"," << r.start << ',' << r.end << ','
             << (int64_t)r.start + offset << ',' << (int64_t)r.end + offset << ','
             << r.queued << ',' << r.submitted << ',' << r.stream << ',' << r.correlation << '\n';
    }
    file.close();

    std::printf("offset_ns=%lld calibration_spread_ns=%lld\n",
                (long long)offset, (long long)spread);
    std::printf("launches=%d activity_records=%zu\n", launches, records.size());
    std::printf("host_issue_ns=%llu cupti_issue_ns=%llu\n",
                (unsigned long long)host_issue, (unsigned long long)cupti_issue);
    std::printf("host_sync_ns=%llu..%llu span_ns=%llu\n",
                (unsigned long long)host_sync_before, (unsigned long long)host_sync_after,
                (unsigned long long)(host_sync_after - host_sync_before));
    std::printf("cupti_sync_ns=%llu..%llu span_ns=%llu\n",
                (unsigned long long)cupti_sync_before, (unsigned long long)cupti_sync_after,
                (unsigned long long)(cupti_sync_after - cupti_sync_before));
    if (!records.empty()) {
        uint64_t first = records.front().start, last = records.back().end;
        uint64_t queued = 0, submitted = 0;
        for (const auto &r : records) {
            queued = std::max(queued, r.queued);
            submitted = std::max(submitted, r.submitted);
        }
        std::printf("activity_gpu_span_ns=%llu..%llu span_ns=%llu\n",
                    (unsigned long long)first, (unsigned long long)last,
                    (unsigned long long)(last - first));
        std::printf("max_queued_ns=%llu max_submitted_ns=%llu\n",
                    (unsigned long long)queued, (unsigned long long)submitted);
    }
    die_cuda(cudaStreamDestroy(stream), "stream destroy");
    die_cuda(cudaFree(device), "free");
    return records.size() == (size_t)launches ? 0 : 5;
}
