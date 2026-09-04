/* tests/bench_dp4a_down.cu — bench + correctness for the SM61 DP4A down
 * projection against the existing grouped_down_g4 kernel.
 *
 * Ornith 1.5 dims (D=2048, Ih=512, gs=64). Same upload XOR-0x88 transform the
 * engine applies. Compares THREE device paths on identical tensors plus the
 * CPU matmul_i4_grouped reference:
 *   1. grouped_down_g4           (existing FP32 FMA, 256 threads/block)
 *   2. dp4a_down_s1_s(.., strat=1) strategy A: 1 warp per output row
 *   3. dp4a_down_s1_s(.., strat=2) strategy B: 8 lanes per row, 4 rows/warp
 *
 * Plus activation quantization cost as a separate line.
 *
 * Build: nvcc -O3 -std=c++17 -arch=sm_61 -DCOLI_CUDA_BUILDING_DLL=0 tests/bench_dp4a_down.cu -o tests/bench_dp4a_down -lcudart
 */
#include "../backend_cuda.h"
#include "../backend_cuda_dp4a.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#include "../backend_cuda.cu"   /* grouped_down_g4 + offset_to_signed_s4 */

/* __CUDA_ARCH__ is only defined in the device pass; check it there. */
__device__ void arch_check(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 610
#error "build with -arch=sm_61 (Pascal); DP4A on sm_61 needs __dp4a()"
#endif
}

/* CPU reference (PRE-XOR s4 decode). */
static void cpu_gemv_g4_down(const uint8_t *q, const float *sc, int K, int O, int gs,
                            const float *x, float *y) {
    int rb = (K + 1) / 2;
    int ng = gs > 0 ? (K + gs - 1) / gs : 1;
    int egs = gs > 0 ? gs : K;
    for (int o = 0; o < O; o++) {
        const uint8_t *row = q + (size_t)o * rb;
        const float *scl = sc + (size_t)o * ng;
        double a = 0.0;
        for (int g = 0; g * egs < K; g++) {
            int base = g * egs, glen = egs;
            if (base + glen > K) glen = K - base;
            double p = 0.0;
            for (int i = base; i < base + glen; i++) {
                uint8_t v = row[i >> 1];
                int n = (i & 1) ? (v >> 4) : (v & 15);
                p += (double)x[i] * (n - 8);
            }
            a += p * scl[g];
        }
        y[o] = (float)a;
    }
}

static void fill_random_weights(uint8_t *w, size_t bytes, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < bytes; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        w[i] = (uint8_t)(s & 0xFF);
    }
}
static void fill_random_scales(float *s, size_t n, uint32_t seed) {
    uint32_t r = seed;
    for (size_t i = 0; i < n; i++) {
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;
        s[i] = 0.005f + 0.04f * ((r & 0xFFFF) / 65535.0f);
    }
}
static void fill_random_acts(float *x, size_t n, uint32_t seed) {
    uint32_t r = seed;
    for (size_t i = 0; i < n; i++) {
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;
        x[i] = ((r & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
    }
}

static void quantize_grouped(const float *x, int8_t *qx, float *xs, int K, int gs) {
    for (int g = 0; g < K / gs; g++) {
        const int base = g * gs;
        float amax = 0.0f;
        for (int j = 0; j < gs; j++) amax = std::max(amax, std::fabs(x[base + j]));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        xs[g] = s;
        for (int j = 0; j < gs; j++) {
            float v = std::max(-127.0f, std::min(127.0f, x[base + j] / s));
            qx[base + j] = (int8_t)std::rint(v);
        }
    }
}

static void cpu_qgemv_down(const uint8_t *q, const float *sc,
                           const int8_t *x, const float *xsc,
                           int K, int O, int gs, float *y) {
    const int rb = (K + 1) / 2, ng = K / gs;
    for (int o = 0; o < O; o++) {
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            int dot = 0;
            for (int j = 0; j < gs; j++) {
                const int i = g * gs + j;
                const uint8_t b = q[(size_t)o * rb + (i >> 1)];
                const int n = (i & 1) ? (b >> 4) : (b & 15);
                dot += (n - 8) * (int)x[i];
            }
            acc += (double)dot * sc[(size_t)o * ng + g] * xsc[g];
        }
        y[o] = (float)acc;
    }
}

static void launch_grouped_down_g4(const uint8_t *qd_dev, const float *sd_dev,
                                   const float *x_dev, float *y_dev,
                                   int D, int I, int gs) {
    GroupDesc host = {nullptr, nullptr, qd_dev,
                      nullptr, nullptr, sd_dev,
                      /*gf*/ 4, /*uf*/ 4, /*df*/ 4,
                      /*rows*/ 1, /*offset*/ 0, 0, 0, gs};
    GroupDesc *ddesc;
    cudaMalloc(&ddesc, sizeof(GroupDesc));
    cudaMemcpy(ddesc, &host, sizeof(GroupDesc), cudaMemcpyHostToDevice);
    dim3 grid((unsigned)D, 1, 1);
    grouped_down_g4<<<grid, 256>>>(y_dev, x_dev, ddesc, D, I);
    cudaDeviceSynchronize();
    cudaFree(ddesc);
}

static void compare_to_ref(const char *label, const float *got, const float *ref,
                           int n, double *out_max, double *out_rms, int *out_bad) {
    double mxa = 0.0, s2 = 0.0, sr = 0.0;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        double r = ref[i], g = got[i];
        double e = g - r;
        double ae = fabs(e);
        if (ae > mxa) mxa = ae;
        s2 += e * e; sr += r * r;
        if (ae > 1e-2 * (fabs(r) + 1e-2)) bad++;
    }
    *out_max = mxa;
    *out_rms = sqrt(s2 / (sr + 1e-20));
    *out_bad = bad;
    fprintf(stderr, "  %-22s  max_abs_err=%.4g  rms=%.4g  bad>1pct=%d/%d\n",
            label, *out_max, *out_rms, bad, n);
}

int main() {
    constexpr int D = 2048, Ih = 512, gs = 64;
    constexpr int ng = Ih / gs;
    constexpr int rbD = (Ih + 1) / 2;
    constexpr size_t wbytes = (size_t)D * rbD;
    constexpr size_t scbytes = (size_t)D * ng * sizeof(float);

    fprintf(stderr, "== bench_dp4a_down ==\n");
    fprintf(stderr, "shape: D=%d Ih=%d gs=%d (down, O=%d K=%d, ng=%d)\n",
            D, Ih, gs, D, Ih, ng);
    fprintf(stderr, "weights: %.1f KB packed, scales: %.1f KB\n",
            wbytes / 1024.0, scbytes / 1024.0);
    fprintf(stderr, "__CUDA_ARCH__=%d (must be 610 for sm_61)\n\n",
#ifdef __CUDA_ARCH__
            __CUDA_ARCH__
#else
            -1
#endif
    );

    std::vector<uint8_t> hW(wbytes);
    std::vector<float>   hSc(scbytes / sizeof(float));
    std::vector<float>   hX(Ih);
    std::vector<int8_t>   hXq(Ih);
    std::vector<float>    hXscale(ng);
    fill_random_weights(hW.data(), wbytes, 0xC0FFEEu);
    fill_random_scales(hSc.data(), scbytes / sizeof(float), 0xBADCAFEu);
    fill_random_acts(hX.data(), Ih, 0xFEED1234u);
    quantize_grouped(hX.data(), hXq.data(), hXscale.data(), Ih, gs);

    uint8_t *dW; float *dSc, *dX_fp32, *dY_fp32;
    int8_t *dX_int8; float *dXscale;
    float *hY_ref = (float*)malloc((size_t)D * sizeof(float));
    float *hY_w4 = (float*)malloc((size_t)D * sizeof(float));
    float *hY_A = (float*)malloc((size_t)D * sizeof(float));
    float *hY_B = (float*)malloc((size_t)D * sizeof(float));
    float *hY_qref = (float*)malloc((size_t)D * sizeof(float));

    cudaMalloc(&dW, wbytes);
    cudaMalloc(&dSc, scbytes);
    cudaMalloc(&dX_fp32, Ih * sizeof(float));
    cudaMalloc(&dY_fp32, D * sizeof(float));
    cudaMalloc(&dX_int8, Ih);
    cudaMalloc(&dXscale, ng * sizeof(float));

    cudaMemcpy(dW, hW.data(), wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dSc, hSc.data(), scbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dX_fp32, hX.data(), Ih * sizeof(float), cudaMemcpyHostToDevice);

    /* Apply upload XOR-0x88 so the device byte form matches the deployed form. */
    {
        size_t n = wbytes;
        dim3 grid((unsigned)((n + 255) / 256));
        offset_to_signed_s4<<<grid, 256>>>((uint8_t*)dW, n);
    }
    cudaDeviceSynchronize();

    cpu_gemv_g4_down(hW.data(), hSc.data(), Ih, D, gs, hX.data(), hY_ref);
    cpu_qgemv_down(hW.data(), hSc.data(), hXq.data(), hXscale.data(), Ih, D, gs, hY_qref);

    /* 1. Existing W4 kernel */
    launch_grouped_down_g4(dW, dSc, dX_fp32, dY_fp32, D, Ih, gs);
    cudaMemcpy(hY_w4, dY_fp32, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost);

    /* Quantize activations once, reuse for both A and B. */
    coli_cuda_dp4a_quantize_row_g(dX_fp32, dX_int8, dXscale, Ih, gs);
    cudaDeviceSynchronize();

    /* 2. DP4A strategy A */
    coli_cuda_dp4a_down_s1_s(dX_int8, dXscale, dW, dSc, dY_fp32, D, Ih, gs, /*strat*/ 1);
    cudaDeviceSynchronize();
    cudaMemcpy(hY_A, dY_fp32, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost);

    /* 3. DP4A strategy B */
    coli_cuda_dp4a_down_s1_s(dX_int8, dXscale, dW, dSc, dY_fp32, D, Ih, gs, /*strat*/ 2);
    cudaDeviceSynchronize();
    cudaMemcpy(hY_B, dY_fp32, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost);

    fprintf(stderr, "== correctness vs CPU matmul_i4_grouped ==\n");
    double max_w4, rms_w4; int bad_w4;
    compare_to_ref("W4  (FP32 FMA)", hY_w4, hY_ref, D, &max_w4, &rms_w4, &bad_w4);
    double max_A, rms_A; int bad_A;
    compare_to_ref("DP4A strategy A", hY_A, hY_ref, D, &max_A, &rms_A, &bad_A);
    double max_B, rms_B; int bad_B;
    compare_to_ref("DP4A strategy B", hY_B, hY_ref, D, &max_B, &rms_B, &bad_B);
    double max_Aq, rms_Aq; int bad_Aq;
    compare_to_ref("DP4A A vs quantized ref", hY_A, hY_qref, D, &max_Aq, &rms_Aq, &bad_Aq);
    double max_Bq, rms_Bq; int bad_Bq;
    compare_to_ref("DP4A B vs quantized ref", hY_B, hY_qref, D, &max_Bq, &rms_Bq, &bad_Bq);

    /* ---- timing ---- */
    double bytes_per_call = (double)wbytes;
    constexpr int ITER = 200;

    /* Warm up + time W4. */
    for (int i = 0; i < 5; i++) launch_grouped_down_g4(dW, dSc, dX_fp32, dY_fp32, D, Ih, gs);
    cudaDeviceSynchronize();
    auto wt0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITER; i++) launch_grouped_down_g4(dW, dSc, dX_fp32, dY_fp32, D, Ih, gs);
    cudaDeviceSynchronize();
    auto wt1 = std::chrono::steady_clock::now();
    double w4_ms = std::chrono::duration<double, std::milli>(wt1 - wt0).count() / ITER;

    /* Activation quant alone. */
    for (int i = 0; i < 5; i++) coli_cuda_dp4a_quantize_row_g(dX_fp32, dX_int8, dXscale, Ih, gs);
    cudaDeviceSynchronize();
    auto qt0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITER; i++)
        coli_cuda_dp4a_quantize_row_g(dX_fp32, dX_int8, dXscale, Ih, gs);
    cudaDeviceSynchronize();
    auto qt1 = std::chrono::steady_clock::now();
    double q_us = std::chrono::duration<double, std::micro>(qt1 - qt0).count() / ITER;

    /* Time DP4A strategy A in isolation (no quant in the loop). */
    for (int i = 0; i < 5; i++)
        coli_cuda_dp4a_down_s1_s(dX_int8, dXscale, dW, dSc, dY_fp32, D, Ih, gs, 1);
    cudaDeviceSynchronize();
    auto at0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITER; i++)
        coli_cuda_dp4a_down_s1_s(dX_int8, dXscale, dW, dSc, dY_fp32, D, Ih, gs, 1);
    cudaDeviceSynchronize();
    auto at1 = std::chrono::steady_clock::now();
    double a_us = std::chrono::duration<double, std::micro>(at1 - at0).count() / ITER;
    double a_ms = a_us / 1000.0;

    /* Time DP4A strategy B in isolation. */
    for (int i = 0; i < 5; i++)
        coli_cuda_dp4a_down_s1_s(dX_int8, dXscale, dW, dSc, dY_fp32, D, Ih, gs, 2);
    cudaDeviceSynchronize();
    auto bt0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITER; i++)
        coli_cuda_dp4a_down_s1_s(dX_int8, dXscale, dW, dSc, dY_fp32, D, Ih, gs, 2);
    cudaDeviceSynchronize();
    auto bt1 = std::chrono::steady_clock::now();
    double b_us = std::chrono::duration<double, std::micro>(bt1 - bt0).count() / ITER;
    double b_ms = b_us / 1000.0;

    double w4_us = w4_ms * 1000.0;
    double dp_total_us_A = q_us + a_us;
    double dp_total_us_B = q_us + b_us;

    fprintf(stderr, "\n== timing (S=1, %d iters after 5-iter warmup) ==\n", ITER);
    fprintf(stderr, "W4   existing kernel:           %7.1f us/call   %6.1f GB/s packed\n",
            w4_us, bytes_per_call / (w4_ms * 1e6));
    fprintf(stderr, "DP4A activation quantize alone: %7.1f us/call\n", q_us);
    fprintf(stderr, "DP4A strategy A (warp/row):     %7.1f us/call   %6.1f GB/s packed\n",
            a_us, bytes_per_call / (a_ms * 1e6));
    fprintf(stderr, "DP4A strategy B (8-lane/row):   %7.1f us/call   %6.1f GB/s packed\n",
            b_us, bytes_per_call / (b_ms * 1e6));
    fprintf(stderr, "DP4A A total (quant + GEMV):    %7.1f us/call   speedup vs W4: %.2fx\n",
            dp_total_us_A, w4_us / dp_total_us_A);
    fprintf(stderr, "DP4A B total (quant + GEMV):    %7.1f us/call   speedup vs W4: %.2fx\n",
            dp_total_us_B, w4_us / dp_total_us_B);

    cudaFree(dW); cudaFree(dSc); cudaFree(dX_fp32); cudaFree(dY_fp32);
    cudaFree(dX_int8); cudaFree(dXscale);
    free(hY_ref); free(hY_w4); free(hY_A); free(hY_B);
    free(hY_qref);
    return 0;
}
