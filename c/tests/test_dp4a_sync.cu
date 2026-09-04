/* Public synchronous API check for the SM61 DP4A batched dispatcher. */
#include "../backend_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t step(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

static void quantize(const float *x, int8_t *qx, float *qs, int K) {
    for (int g = 0; g < K / 64; g++) {
        const int base = g * 64;
        float amax = 0.0f;
        for (int j = 0; j < 64; j++) amax = std::max(amax, std::fabs(x[base + j]));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        qs[g] = s;
        for (int j = 0; j < 64; j++) {
            float v = std::max(-127.0f, std::min(127.0f, x[base + j] / s));
            qx[base + j] = (int8_t)std::rint(v);
        }
    }
}

static void qgemv(const uint8_t *w, const float *ws, const int8_t *x,
                  const float *xs, int K, int O, float *y) {
    const int rb = K / 2, ng = K / 64;
    for (int o = 0; o < O; o++) {
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            int dot = 0;
            for (int j = 0; j < 64; j++) {
                const int i = g * 64 + j;
                const uint8_t b = w[(size_t)o * rb + (i >> 1)];
                const int n = (((i & 1) ? (b >> 4) : (b & 15)) - 8);
                dot += n * (int)x[i];
            }
            acc += (double)dot * ws[(size_t)o * ng + g] * xs[g];
        }
        y[o] = (float)acc;
    }
}

int main() {
    constexpr int D = 2048, I = 512, C = 3;
    const size_t rbgu = D / 2, rbd = I / 2;
    const size_t bgu = (size_t)I * rbgu, bd = (size_t)D * rbd;
    const size_t sgu = (size_t)I * (D / 64), sd = (size_t)D * (I / 64);

    std::vector<std::vector<uint8_t>> hg(C, std::vector<uint8_t>(bgu));
    std::vector<std::vector<uint8_t>> hu(C, std::vector<uint8_t>(bgu));
    std::vector<std::vector<uint8_t>> hd(C, std::vector<uint8_t>(bd));
    std::vector<std::vector<float>> hgs(C, std::vector<float>(sgu));
    std::vector<std::vector<float>> hus(C, std::vector<float>(sgu));
    std::vector<std::vector<float>> hds(C, std::vector<float>(sd));
    uint32_t seed = 0x6D5034Au;
    for (int e = 0; e < C; e++) {
        for (auto &v : hg[e]) v = (uint8_t)step(seed);
        for (auto &v : hu[e]) v = (uint8_t)step(seed);
        for (auto &v : hd[e]) v = (uint8_t)step(seed);
        for (auto &v : hgs[e]) v = 0.005f + 0.04f * ((step(seed) & 65535) / 65535.0f);
        for (auto &v : hus[e]) v = 0.005f + 0.04f * ((step(seed) & 65535) / 65535.0f);
        for (auto &v : hds[e]) v = 0.005f + 0.04f * ((step(seed) & 65535) / 65535.0f);
    }
    std::vector<float> x((size_t)C * D);
    for (int i = 0; i < D; i++)
        x[i] = ((step(seed) & 65535) / 65535.0f - 0.5f) * 2.0f;
    for (int e = 1; e < C; e++)
        std::memcpy(x.data() + (size_t)e * D, x.data(), (size_t)D * sizeof(float));

    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { std::fprintf(stderr, "FAIL init\n"); return 1; }
    _putenv_s("COLI_CUDA_DP4A", "1");
    _putenv_s("COLI_CUDA_DP4A_QUIET", "1");
    ColiCudaTensor *g[C] = {}, *u[C] = {}, *d[C] = {};
    for (int e = 0; e < C; e++) {
        if (!coli_cuda_tensor_upload_g(&g[e], hg[e].data(), hgs[e].data(), 4, D, I, 0, 64) ||
            !coli_cuda_tensor_upload_g(&u[e], hu[e].data(), hus[e].data(), 4, D, I, 0, 64) ||
            !coli_cuda_tensor_upload_g(&d[e], hd[e].data(), hds[e].data(), 4, I, D, 0, 64)) {
            std::fprintf(stderr, "FAIL upload\n"); return 1;
        }
    }
    int rows[C] = {1, 1, 1};
    std::vector<float> y((size_t)C * D), refg(I), refu(I), refy(D);
    if (!coli_cuda_expert_group(g, u, d, rows, C, y.data(), x.data())) {
        std::fprintf(stderr, "FAIL sync dispatch\n"); return 1;
    }

    std::vector<int8_t> qx(D), qgate(I);
    std::vector<float> xs(D / 64), qgs(I / 64), qdown(D);
    quantize(x.data(), qx.data(), xs.data(), D);
    double maxerr = 0.0, sum2 = 0.0;
    int bad = 0;
    for (int e = 0; e < C; e++) {
        qgemv(hg[e].data(), hgs[e].data(), qx.data(), xs.data(), D, I, refg.data());
        qgemv(hu[e].data(), hus[e].data(), qx.data(), xs.data(), D, I, refu.data());
        for (int i = 0; i < I; i++) refg[i] = (refg[i] / (1.0f + std::exp(-refg[i]))) * refu[i];
        std::vector<float> qgscale(I / 64);
        quantize(refg.data(), qgate.data(), qgscale.data(), I);
        qgemv(hd[e].data(), hds[e].data(), qgate.data(), qgscale.data(), I, D, refy.data());
        for (int i = 0; i < D; i++) {
            const double err = y[(size_t)e * D + i] - refy[i];
            maxerr = std::max(maxerr, std::fabs(err)); sum2 += err * err;
            if (std::fabs(err) > 1e-3 * (std::fabs(refy[i]) + 0.01)) bad++;
        }
    }
    std::printf("sync DP4A batched: experts=%d max_abs=%.6g rms=%.6g bad=%d\n",
                C, maxerr, std::sqrt(sum2 / (C * D)), bad);
    for (int e = 0; e < C; e++) {
        coli_cuda_tensor_free(g[e]); coli_cuda_tensor_free(u[e]); coli_cuda_tensor_free(d[e]);
    }
    coli_cuda_shutdown();
    return bad ? 1 : 0;
}
