/* tests/pci_probe.cu — PCIe link + transfer microbenchmark.
 *
 * Reports:
 *  - actual negotiated PCIe generation and width for the device
 *  - H2D and D2H latency for pinned vs pageable host buffers
 *  - effective bandwidth for the transfer shapes the engine uses (~64 KiB)
 *  - synchronous vs asynchronous memcpy latency
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#define CUDA_OK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(_e));std::exit(1);} } while(0)

static void report_pcie() {
    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, 0));
    fprintf(stderr, "=== PCIe link ===\n");
    fprintf(stderr, "  device 0: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    fprintf(stderr, "  pciBusID=%d  pciDeviceID=%d  pciDomainID=%d\n",
            p.pciBusID, p.pciDeviceID, p.pciDomainID);
    int gen = 0, width = 0;
    /* cudaDevAttr enum values are stable across CUDA versions, but the names
     * moved at CUDA 12. Use the integer values directly for portability. */
    cudaDeviceGetAttribute(&gen,   (cudaDeviceAttr)75,  0);  /* PciGenId   */
    cudaDeviceGetAttribute(&width, (cudaDeviceAttr)74,  0);  /* PciBusWidth */
    fprintf(stderr, "  cudaDevAttrPciGen   = %d  (effective ~Gen%d)\n", gen, gen);
    fprintf(stderr, "  cudaDevAttrPciWidth = %d  (effective x%d)\n", width, width);
    /* Peak BW: Gen1=2.5 GT/s, Gen2=5, Gen3=8, Gen4=16, Gen5=32 Gbps/lane,
       8b/10b encoded for Gen1/2 (×0.8), 128b/130b for Gen3+ (~×0.984).
       Worst case here is Gen3 x16 × 0.984 × 2 = ~25 GB/s peak. */
    double peak_gbps = 0.0;
    if (gen == 1) peak_gbps = 2.5 * width * 0.8 / 8.0;
    else if (gen == 2) peak_gbps = 5.0 * width * 0.8 / 8.0;
    else if (gen == 3) peak_gbps = 8.0 * width * 128.0 / 130.0 / 8.0;
    else if (gen == 4) peak_gbps = 16.0 * width * 0.984 / 8.0;
    else if (gen == 5) peak_gbps = 32.0 * width * 0.984 / 8.0;
    fprintf(stderr, "  theoretical peak H2D = %.2f GB/s\n", peak_gbps);
}

static double bench_xfer(int bytes, bool pinned, bool sync, int iters) {
    unsigned char *h_buf = nullptr;
    if (pinned) CUDA_OK(cudaMallocHost((void**)&h_buf, bytes));
    else h_buf = (unsigned char*)malloc(bytes);
    memset(h_buf, 0xA5, bytes);
    unsigned char *d_buf = nullptr;
    CUDA_OK(cudaMalloc(&d_buf, bytes));

    /* warmup */
    for (int i = 0; i < 5; i++) {
        if (sync) CUDA_OK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
        else CUDA_OK(cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice, 0));
    }
    CUDA_OK(cudaDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        if (sync) CUDA_OK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
        else CUDA_OK(cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice, 0));
    }
    CUDA_OK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    cudaFree(d_buf);
    if (pinned) cudaFreeHost(h_buf); else free(h_buf);
    return us;
}

static void report_xfers() {
    fprintf(stderr, "\n=== transfer microbenchmark (H2D, default stream) ===\n");
    /* the actual shapes the qwen36 expert call uses:
       expert_count=8, D=2048, sizeof(float)=4 -> 8*2048*4 = 64 KiB */
    int bytes = 64 * 1024;
    int iters = 500;
    fprintf(stderr, "  size=%d B (%.1f KiB), iters=%d\n", bytes, bytes / 1024.0, iters);
    double pageable_sync  = bench_xfer(bytes, false, true,  iters);
    double pageable_async = bench_xfer(bytes, false, false, iters);
    double pinned_sync    = bench_xfer(bytes, true,  true,  iters);
    double pinned_async   = bench_xfer(bytes, true,  false, iters);
    double bw = (double)bytes / 1e6;
    fprintf(stderr, "  pageable sync : %7.2f us/call  (%6.2f GB/s)\n", pageable_sync,  bw / pageable_sync);
    fprintf(stderr, "  pageable async: %7.2f us/call  (%6.2f GB/s)\n", pageable_async, bw / pageable_async);
    fprintf(stderr, "  pinned   sync : %7.2f us/call  (%6.2f GB/s)\n", pinned_sync,    bw / pinned_sync);
    fprintf(stderr, "  pinned   async: %7.2f us/call  (%6.2f GB/s)\n", pinned_async,   bw / pinned_async);

    /* pure async submission latency (no sync after) */
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        unsigned char *h_buf = nullptr;
        unsigned char *d_buf = nullptr;
        CUDA_OK(cudaMallocHost((void**)&h_buf, bytes));
        CUDA_OK(cudaMalloc(&d_buf, bytes));
        CUDA_OK(cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice, 0));
        CUDA_OK(cudaDeviceSynchronize());
        cudaFree(d_buf); cudaFreeHost(h_buf);
    }
    auto t1 = std::chrono::steady_clock::now();
    double all_inclusive_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    fprintf(stderr, "  alloc + async + sync: %7.2f us/call (full per-call cost including cudaMalloc)\n", all_inclusive_us);
}

int main() {
    report_pcie();
    report_xfers();
    return 0;
}