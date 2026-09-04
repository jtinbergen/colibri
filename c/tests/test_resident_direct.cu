/* Verify that the single-island direct accumulator is exactly equivalent to
 * the old one-slot sum_slots reduction.  This deliberately tests the public
 * resident issue/take boundary without Qwen routing or sampler state. */
#include "../backend_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static uint32_t next_u32(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

static int check_cuda(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 1;
    std::fprintf(stderr, "FAIL %s: %s\n", what, cudaGetErrorString(e));
    return 0;
}

int main() {
    constexpr int D = 2048, I = 512;
    constexpr size_t GU_BYTES = (size_t)I * D / 2;
    constexpr size_t D_BYTES = (size_t)D * I / 2;
    constexpr size_t GU_SCALES = (size_t)I * (D / 64);
    constexpr size_t D_SCALES = (size_t)D * (I / 64);

    uint32_t seed = 0x51A61D4Au;
    std::vector<uint8_t> gw(GU_BYTES), uw(GU_BYTES), dw(D_BYTES);
    std::vector<float> gs(GU_SCALES), us(GU_SCALES), ds(D_SCALES);
    for (uint8_t &v : gw) v = (uint8_t)(next_u32(seed) ^ 0x88u);
    for (uint8_t &v : uw) v = (uint8_t)(next_u32(seed) ^ 0x88u);
    for (uint8_t &v : dw) v = (uint8_t)(next_u32(seed) ^ 0x88u);
    for (float &v : gs) v = 0.005f + 0.04f * ((next_u32(seed) & 65535u) / 65535.0f);
    for (float &v : us) v = 0.005f + 0.04f * ((next_u32(seed) & 65535u) / 65535.0f);
    for (float &v : ds) v = 0.005f + 0.04f * ((next_u32(seed) & 65535u) / 65535.0f);

    std::vector<float> x(D);
    for (float &v : x)
        v = ((next_u32(seed) & 65535u) / 65535.0f - 0.5f) * 2.0f;

    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) return 1;
    _putenv_s("COLI_CUDA_DP4A", "1");
    _putenv_s("COLI_CUDA_DP4A_QUIET", "1");

    ColiCudaTensor *g = nullptr, *u = nullptr, *d = nullptr;
    float *x_dev = nullptr, *old_acc = nullptr, *direct_acc = nullptr;
    int ok =
        coli_cuda_tensor_upload_g(&g, gw.data(), gs.data(), 4, D, I, dev, 64) &&
        coli_cuda_tensor_upload_g(&u, uw.data(), us.data(), 4, D, I, dev, 64) &&
        coli_cuda_tensor_upload_g(&d, dw.data(), ds.data(), 4, I, D, dev, 64) &&
        check_cuda(cudaMalloc(&x_dev, (size_t)D * sizeof(float)), "x alloc") &&
        check_cuda(cudaMalloc(&old_acc, (size_t)D * sizeof(float)), "old accumulator alloc") &&
        check_cuda(cudaMalloc(&direct_acc, (size_t)D * sizeof(float)), "direct accumulator alloc") &&
        check_cuda(cudaMemcpy(x_dev, x.data(), (size_t)D * sizeof(float), cudaMemcpyHostToDevice),
                   "x upload");
    if (!ok || !g || !u || !d) {
        std::fprintf(stderr, "FAIL setup\n");
        if (g) coli_cuda_tensor_free(g);
        if (u) coli_cuda_tensor_free(u);
        if (d) coli_cuda_tensor_free(d);
        if (x_dev) cudaFree(x_dev);
        if (old_acc) cudaFree(old_acc);
        if (direct_acc) cudaFree(direct_acc);
        coli_cuda_shutdown();
        return 1;
    }

    ColiCudaTensor *gates[1] = {g}, *ups[1] = {u}, *downs[1] = {d};
    const float weight[1] = {0.731f};
    const int devices[1] = {dev};

    ok = coli_cuda_expert_group_resident_issue(gates, ups, downs, weight, 1,
                                               dev, x_dev, old_acc);
    ok = ok && coli_cuda_expert_group_resident_take(dev, devices, 1,
                                                    old_acc, old_acc, D);
    ok = ok && coli_cuda_expert_group_resident_sync(dev);
    ok = ok && coli_cuda_expert_group_resident_issue(gates, ups, downs, weight, 1,
                                                     dev, x_dev, direct_acc);
    ok = ok && coli_cuda_expert_group_resident_take(dev, devices, 1,
                                                    direct_acc, direct_acc, D);
    ok = ok && coli_cuda_expert_group_resident_sync(dev);

    std::vector<float> old_host(D), direct_host(D);
    if (ok) ok = check_cuda(cudaMemcpy(old_host.data(), old_acc,
                                       (size_t)D * sizeof(float), cudaMemcpyDeviceToHost),
                            "old result download");
    if (ok) ok = check_cuda(cudaMemcpy(direct_host.data(), direct_acc,
                                       (size_t)D * sizeof(float), cudaMemcpyDeviceToHost),
                            "direct result download");
    size_t differing = 0;
    float max_abs = 0.0f;
    if (ok) for (int i = 0; i < D; i++) {
        if (std::memcmp(&old_host[i], &direct_host[i], sizeof(float)) != 0) differing++;
        max_abs = std::max(max_abs, std::abs(old_host[i] - direct_host[i]));
    }
    std::printf("resident direct accumulator: differing=%zu max_abs=%.9g\n",
                differing, max_abs);

    coli_cuda_tensor_free(g);
    coli_cuda_tensor_free(u);
    coli_cuda_tensor_free(d);
    cudaFree(x_dev);
    cudaFree(old_acc);
    cudaFree(direct_acc);
    coli_cuda_shutdown();
    return ok && differing == 0 ? 0 : 1;
}
