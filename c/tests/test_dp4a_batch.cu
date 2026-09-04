/* Batched SM61 DP4A correctness gate: three S=1 experts through the exact
 * four-launch production sequence (quant-x, gate/up, quant-down, down). */
#include "../backend_cuda_dp4a.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

static uint32_t step(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

static void ref_gemv(const uint8_t *w, const float *sc, int K, int O,
                     const float *x, float *y) {
    const int rb = K / 2, ng = K / 64;
    for (int o = 0; o < O; o++) {
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            double dot = 0.0;
            for (int j = 0; j < 64; j++) {
                int i = g * 64 + j;
                uint8_t b = w[(size_t)o * rb + (i >> 1)];
                int n = (i & 1) ? (b >> 4) : (b & 15);
                n = (n ^ 8); /* h*w is deployed post-XOR form */
                dot += (double)(n - 8) * x[i];
            }
            acc += dot * sc[(size_t)o * ng + g];
        }
        y[o] = (float)acc;
    }
}

static void quantize_groups(const float *x, int8_t *qx, float *qs, int rows, int K) {
    constexpr int gs = 64;
    const int ng = K / gs;
    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < ng; g++) {
            const int base = r * K + g * gs;
            float amax = 0.0f;
            for (int j = 0; j < gs; j++) amax = std::max(amax, std::fabs(x[base + j]));
            const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
            qs[r * ng + g] = s;
            for (int j = 0; j < gs; j++) {
                float v = x[base + j] / s;
                v = std::max(-127.0f, std::min(127.0f, v));
                qx[base + j] = (int8_t)std::rint(v);
            }
        }
    }
}

static void ref_qgemv(const uint8_t *w, const float *wsc, const int8_t *qx,
                      const float *xsc, int K, int O, float *y) {
    const int rb = K / 2, ng = K / 64;
    for (int o = 0; o < O; o++) {
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            int dot = 0;
            for (int j = 0; j < 64; j++) {
                const int i = g * 64 + j;
                const uint8_t b = w[(size_t)o * rb + (i >> 1)];
                int n = (i & 1) ? (b >> 4) : (b & 15);
                n = (n ^ 8) - 8;
                dot += (n * (int)qx[i]);
            }
            acc += (double)dot * wsc[(size_t)o * ng + g] * xsc[g];
        }
        y[o] = (float)acc;
    }
}

int main() {
    constexpr int D = 2048, I = 512, C = 3;
    constexpr size_t rb_gu = D / 2, rb_d = I / 2;
    constexpr size_t ng_gu = D / 64, ng_d = I / 64;
    uint32_t seed = 0x51D4A7u;
    std::vector<std::vector<uint8_t>> hgw(C, std::vector<uint8_t>((size_t)I * rb_gu));
    std::vector<std::vector<uint8_t>> huw(C, std::vector<uint8_t>((size_t)I * rb_gu));
    std::vector<std::vector<uint8_t>> hdw(C, std::vector<uint8_t>((size_t)D * rb_d));
    std::vector<std::vector<float>> hgs(C, std::vector<float>((size_t)I * ng_gu));
    std::vector<std::vector<float>> hus(C, std::vector<float>((size_t)I * ng_gu));
    std::vector<std::vector<float>> hds(C, std::vector<float>((size_t)D * ng_d));
    for (int e = 0; e < C; e++) {
        for (auto &b : hgw[e]) b = (uint8_t)(step(seed) ^ 0x88u);
        for (auto &b : huw[e]) b = (uint8_t)(step(seed) ^ 0x88u);
        for (auto &b : hdw[e]) b = (uint8_t)(step(seed) ^ 0x88u);
        for (auto &s : hgs[e]) s = 0.005f + 0.04f * ((step(seed) & 0xffff) / 65535.0f);
        for (auto &s : hus[e]) s = 0.005f + 0.04f * ((step(seed) & 0xffff) / 65535.0f);
        for (auto &s : hds[e]) s = 0.005f + 0.04f * ((step(seed) & 0xffff) / 65535.0f);
    }
    std::vector<float> hx(D);
    for (float &v : hx) v = ((step(seed) & 0xffff) / 65535.0f - 0.5f) * 2.0f;

    uint8_t *dgw[C], *duw[C], *ddw[C];
    float *dgsc[C], *dusc[C], *ddsc[C];
    const uint8_t *pgw[C], *puw[C], *pdw[C];
    const float *pgsc[C], *pusc[C], *pdsc[C];
    for (int e = 0; e < C; e++) {
        cudaMalloc(&dgw[e], hgw[e].size()); cudaMalloc(&duw[e], huw[e].size());
        cudaMalloc(&ddw[e], hdw[e].size()); cudaMalloc(&dgsc[e], hgs[e].size() * sizeof(float));
        cudaMalloc(&dusc[e], hus[e].size() * sizeof(float)); cudaMalloc(&ddsc[e], hds[e].size() * sizeof(float));
        cudaMemcpy(dgw[e], hgw[e].data(), hgw[e].size(), cudaMemcpyHostToDevice);
        cudaMemcpy(duw[e], huw[e].data(), huw[e].size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ddw[e], hdw[e].data(), hdw[e].size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dgsc[e], hgs[e].data(), hgs[e].size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dusc[e], hus[e].data(), hus[e].size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ddsc[e], hds[e].data(), hds[e].size() * sizeof(float), cudaMemcpyHostToDevice);
        pgw[e] = dgw[e]; puw[e] = duw[e]; pdw[e] = ddw[e];
        pgsc[e] = dgsc[e]; pusc[e] = dusc[e]; pdsc[e] = ddsc[e];
    }
    const size_t ptrbytes = C * sizeof(void *);
    uint8_t **d_pgw, **d_puw, **d_pdw; float **d_pgsc, **d_pusc, **d_pdsc;
    cudaMalloc(&d_pgw, ptrbytes); cudaMalloc(&d_puw, ptrbytes); cudaMalloc(&d_pdw, ptrbytes);
    cudaMalloc(&d_pgsc, ptrbytes); cudaMalloc(&d_pusc, ptrbytes); cudaMalloc(&d_pdsc, ptrbytes);
    cudaMemcpy(d_pgw, pgw, ptrbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_puw, puw, ptrbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pdw, pdw, ptrbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pgsc, pgsc, ptrbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pusc, pusc, ptrbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pdsc, pdsc, ptrbytes, cudaMemcpyHostToDevice);

    float *dx, *dgate, *dy; int8_t *dqx; float *dxsc;
    int8_t *dgateq; float *dgatesc;
    cudaMalloc(&dx, D * sizeof(float)); cudaMalloc(&dgate, (size_t)C * I * sizeof(float));
    cudaMalloc(&dy, (size_t)C * D * sizeof(float)); cudaMalloc(&dqx, D);
    cudaMalloc(&dxsc, ng_gu * sizeof(float)); cudaMalloc(&dgateq, (size_t)C * I);
    cudaMalloc(&dgatesc, (size_t)C * ng_d * sizeof(float));
    cudaMemcpy(dx, hx.data(), D * sizeof(float), cudaMemcpyHostToDevice);
    int ok = coli_cuda_dp4a_quantize_row_g_s(dx, dqx, dxsc, D, 64, 0);
    ok = ok && coli_cuda_dp4a_gate_up_all_s(dqx, dxsc, d_pgw, d_puw, d_pgsc, d_pusc,
                                            dgate, C, I, D, 64, 0);
    ok = ok && coli_cuda_dp4a_quantize_rows_g_s(dgate, dgateq, dgatesc, C, I, 64, 0);
    ok = ok && coli_cuda_dp4a_down_all_s(dgateq, dgatesc, d_pdw, d_pdsc, dy, C, D, I, 64, 0);
    if (!ok || cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: batched DP4A launch\n"); return 1;
    }
    std::vector<float> got((size_t)C * D), gotgate((size_t)C * I),
        refg(I), refu(I), refy(D), refall((size_t)C * D);
    cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gotgate.data(), dgate, gotgate.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<int8_t> hqx(D), hgateq((size_t)C * I);
    std::vector<float> hxsc(ng_gu), hgate_sc((size_t)C * ng_d), qgate(I), qup(I), qdown(D);
    quantize_groups(hx.data(), hqx.data(), hxsc.data(), 1, D);
    double maxerr = 0.0, rel2 = 0.0, gate_maxerr = 0.0, gate_rel2 = 0.0;
    double qgate_maxerr = 0.0, qgate_rel2 = 0.0, qdown_maxerr = 0.0, qdown_rel2 = 0.0;
    int bad_fp32 = 0, bad_quant = 0;
    for (int e = 0; e < C; e++) {
        ref_gemv(hgw[e].data(), hgs[e].data(), D, I, hx.data(), refg.data());
        ref_gemv(huw[e].data(), hus[e].data(), D, I, hx.data(), refu.data());
        for (int i = 0; i < I; i++) {
            refg[i] = (refg[i] / (1.0f + std::exp(-refg[i]))) * refu[i];
            double ge = gotgate[(size_t)e * I + i] - refg[i];
            gate_maxerr = std::max(gate_maxerr, std::fabs(ge)); gate_rel2 += ge * ge;
        }
        std::vector<float> gate_row(gotgate.begin() + (size_t)e * I,
                                    gotgate.begin() + (size_t)(e + 1) * I);
        quantize_groups(gate_row.data(), hgateq.data() + (size_t)e * I,
                        hgate_sc.data() + (size_t)e * ng_d, 1, I);
        ref_qgemv(hgw[e].data(), hgs[e].data(), hqx.data(), hxsc.data(), D, I, qgate.data());
        ref_qgemv(huw[e].data(), hus[e].data(), hqx.data(), hxsc.data(), D, I, qup.data());
        for (int i = 0; i < I; i++) qgate[i] = (qgate[i] / (1.0f + std::exp(-qgate[i]))) * qup[i];
        for (int i = 0; i < I; i++) {
            double qe = gotgate[(size_t)e * I + i] - qgate[i];
            qgate_maxerr = std::max(qgate_maxerr, std::fabs(qe)); qgate_rel2 += qe * qe;
        }
        ref_gemv(hdw[e].data(), hds[e].data(), I, D, refg.data(), refy.data());
        ref_qgemv(hdw[e].data(), hds[e].data(), hgateq.data() + (size_t)e * I,
                  hgate_sc.data() + (size_t)e * ng_d, I, D, qdown.data());
        for (int i = 0; i < D; i++) {
            double err = got[(size_t)e * D + i] - refy[i];
            maxerr = std::max(maxerr, std::fabs(err)); rel2 += err * err;
            if (std::fabs(err) > 0.03 * (std::fabs(refy[i]) + 0.01)) bad_fp32++;
            double qe = got[(size_t)e * D + i] - qdown[i];
            qdown_maxerr = std::max(qdown_maxerr, std::fabs(qe)); qdown_rel2 += qe * qe;
            if (std::fabs(qe) > 1e-3 * (std::fabs(qdown[i]) + 0.01)) bad_quant++;
        }
    }
    std::printf("batched DP4A: experts=%d gate_max_abs=%.6g gate_rms=%.6g "
                "down_max_abs=%.6g down_rms=%.6g qgate_max_abs=%.6g qgate_rms=%.6g "
                "qdown_max_abs=%.6g qdown_rms=%.6g bad=%d\n",
                C, gate_maxerr, std::sqrt(gate_rel2 / (C * I)),
                maxerr, std::sqrt(rel2 / (C * D)), qgate_maxerr,
                std::sqrt(qgate_rel2 / (C * I)), qdown_maxerr,
                std::sqrt(qdown_rel2 / (C * D)), bad_quant);
    std::printf("  FP32-reference differences above are expected from INT8 quantization; "
                "bad_vs_FP32=%d bad_vs_quantized=%d\n", bad_fp32, bad_quant);
    /* Row-aware prefill gate/up: each route entry has an independent input
     * row. This catches the original shared-qx assumption directly. */
    std::vector<float> hrows((size_t)C * D), gotrows((size_t)C * I);
    std::vector<int8_t> hqrows((size_t)C * D);
    std::vector<float> hscrows((size_t)C * ng_gu);
    for (int r = 0; r < C; r++)
        for (int i = 0; i < D; i++) hrows[(size_t)r * D + i] = hx[i] * (0.5f + 0.25f * r);
    float *dx_rows; int8_t *dqx_rows; float *dxsc_rows;
    cudaMalloc(&dx_rows, (size_t)C * D * sizeof(float));
    cudaMalloc(&dqx_rows, (size_t)C * D); cudaMalloc(&dxsc_rows, (size_t)C * ng_gu * sizeof(float));
    cudaMemcpy(dx_rows, hrows.data(), (size_t)C * D * sizeof(float), cudaMemcpyHostToDevice);
    int row_ok = coli_cuda_dp4a_quantize_rows_g_s(dx_rows, dqx_rows, dxsc_rows, C, D, 64, 0);
    row_ok = row_ok && coli_cuda_dp4a_gate_up_rows_s(
        dqx_rows, dxsc_rows, d_pgw, d_puw, d_pgsc, d_pusc, dgate,
        C, I, D, 64, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(gotrows.data(), dgate, gotrows.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hqrows.data(), dqx_rows, hqrows.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hscrows.data(), dxsc_rows, hscrows.size() * sizeof(float), cudaMemcpyDeviceToHost);
    int row_bad = row_ok ? 0 : C * I;
    for (int r = 0; row_ok && r < C; r++) {
        std::vector<float> rg(I), ru(I), rq(I);
        ref_qgemv(hgw[r].data(), hgs[r].data(), hqrows.data() + (size_t)r * D,
                  hscrows.data() + (size_t)r * ng_gu, D, I, rg.data());
        ref_qgemv(huw[r].data(), hus[r].data(), hqrows.data() + (size_t)r * D,
                  hscrows.data() + (size_t)r * ng_gu, D, I, ru.data());
        for (int i = 0; i < I; i++) {
            rg[i] = (rg[i] / (1.0f + std::exp(-rg[i]))) * ru[i];
            if (std::fabs(gotrows[(size_t)r * I + i] - rg[i]) >
                1e-3 * (std::fabs(rg[i]) + 0.01)) row_bad++;
        }
    }
    std::printf("  row-aware gate/up: bad_vs_quantized=%d\n", row_bad);
    cudaFree(dx_rows); cudaFree(dqx_rows); cudaFree(dxsc_rows);
    for (int e = 0; e < C; e++) { cudaFree(dgw[e]); cudaFree(duw[e]); cudaFree(ddw[e]);
        cudaFree(dgsc[e]); cudaFree(dusc[e]); cudaFree(ddsc[e]); }
    cudaFree(d_pgw); cudaFree(d_puw); cudaFree(d_pdw); cudaFree(d_pgsc); cudaFree(d_pusc); cudaFree(d_pdsc);
    cudaFree(dx); cudaFree(dgate); cudaFree(dy); cudaFree(dqx); cudaFree(dxsc); cudaFree(dgateq); cudaFree(dgatesc);
    return (bad_quant || row_bad) ? 1 : 0;
}
