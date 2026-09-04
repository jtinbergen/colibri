/* backend_cuda_dp4a.cu — SM61 DP4A prototype for grouped-int4 expert GEMV.
 *
 * S=1 top-K expert path (decode hot path). Four CUDA launches per call:
 *
 *   L1: quantize_activations_dp4a_g_kernel   (one block per group, K elements)
 *   L2: dp4a_gate_up_all_kernel              (grid=(O_gu, count), all experts in parallel)
 *   L3: dp4a_down_all_kernel                 (grid=(O_d/4, count), all experts in parallel)
 *
 * L2 batches all K experts' gate+up projections into a single 2D-grid launch
 * (one block per (output_row, expert)). L3 batches all K experts' down
 * projections similarly. A separate batched quantization launch prepares the
 * silu*up rows for L3, so each down output row reuses the same INT8 inputs.
 * That collapses the previous "per-expert loop" pattern (which was 2*K + 2
 * launches) down to 4 total launches, matching the existing W4 path's
 * 2-launch structure within two quantization launches.
 *
 * Correctness notes (see comments at file top):
 *   - Nibble decode: `n & 8 ? n - 16 : n` (sign-extends the post-XOR nibble).
 *   - CUDA __dp4a wrapper: result = dot4(a, b) + c. Accumulator is THIRD arg.
 *   - Per-group scaling: INT32 resets at each group boundary, FP32 accumulates.
 *
 * Strategy A is unused here; the B-style "one lane per group, 4 rows per warp"
 * mapping is the only layout shipped because it maps exactly to the
 * qwen36/Ornith shapes (down: K=512 gs=64 ng=8; gate/up: K=2048 gs=64 ng=32).
 */
#include "backend_cuda_dp4a.h"

#include "backend_gpu_compat.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#ifndef COLI_DP4A_BLOCK_THREADS
#define COLI_DP4A_BLOCK_THREADS 128
#endif
#ifndef COLI_DP4A_QUANT_THREADS
#define COLI_DP4A_QUANT_THREADS 64
#endif

/* ---- helpers -------------------------------------------------------------- */

static int g_dp4a_capture_mode = 0;

extern "C" void coli_cuda_dp4a_capture_mode(int active) {
    g_dp4a_capture_mode = active != 0;
}

/* cudaGetLastError is not permitted while a stream is being captured.  The
 * graph prototype still uses the same launch wrappers, so defer this legacy
 * post-launch query until replay/normal execution; launch errors are reported
 * by cudaGraphInstantiate/Launch or by the later stream synchronization. */
static cudaError_t dp4a_post_launch_error(cudaStream_t stream) {
    (void)stream;
    return g_dp4a_capture_mode ? cudaSuccess : cudaGetLastError();
}

__device__ __forceinline__ int8_t s4_to_s8(int n) {
    int8_t v = (int8_t)((uint8_t)(n & 0x0F) << 4);
    v = (int8_t)((int)v >> 4);
    return v;
}

__device__ __forceinline__ int32_t pack4_s8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (int32_t)(uint32_t)(uint8_t)a
         | ((int32_t)(uint32_t)(uint8_t)b << 8)
         | ((int32_t)(uint32_t)(uint8_t)c << 16)
         | ((int32_t)(uint32_t)(uint8_t)d << 24);
}

/* ---- L1: per-group activation quantization -------------------------------- *
 * One block per group; reads gs_a contiguous activations from x, produces
 * 1 scale per group and gs_a INT8 values. */
__global__ static void quantize_activations_dp4a_g_kernel(const float *x,
                                                         int8_t *qx,
                                                         float *qscale,
                                                         int K, int gs_a) {
    int g = blockIdx.x;
    int base = g * gs_a;
    int tid = threadIdx.x;
    int nthr = blockDim.x;

    extern __shared__ float shf[];
    float v = 0.0f;
    for (int i = tid; i < gs_a; i += nthr) {
        float a = fabsf(x[base + i]);
        if (a > v) v = a;
    }
    shf[tid] = v;
    __syncthreads();
    for (int n = nthr >> 1; n; n >>= 1) {
        if (tid < n) {
            float a = shf[tid], b = shf[tid + n];
            shf[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    __shared__ float invs;
    if (!tid) {
        float amax = shf[0];
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        qscale[g] = s;
        invs = 1.0f / s;
    }
    __syncthreads();
    float inv = invs;
    for (int i = tid; i < gs_a; i += nthr) {
        float v = x[base + i] * inv;
        float c = v < -127.0f ? -127.0f : (v > 127.0f ? 127.0f : v);
        qx[base + i] = (int8_t)rintf(c);
    }
}

extern "C" int coli_cuda_dp4a_quantize_row_g_s(const float *x, int8_t *qx,
                                              float *qscale, int K, int gs_a,
                                              cudaStream_t stream) {
    if (!x || !qx || !qscale || K <= 0 || gs_a <= 0) return 0;
    if (K % gs_a != 0) return 0;
    int ng = K / gs_a;
    size_t sh_bytes = (size_t)COLI_DP4A_QUANT_THREADS * sizeof(float);
    quantize_activations_dp4a_g_kernel<<<ng, COLI_DP4A_QUANT_THREADS, sh_bytes, stream>>>(
        x, qx, qscale, K, gs_a);
    cudaError_t e = dp4a_post_launch_error(stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] quantize_g launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

extern "C" int coli_cuda_dp4a_quantize_row_g(const float *x, int8_t *qx,
                                            float *qscale, int K, int gs_a) {
    return coli_cuda_dp4a_quantize_row_g_s(x, qx, qscale, K, gs_a, 0);
}

/* Batch activation quantization. One block owns one (row, group), so the
 * work scales with the number of expert inputs rather than with the number
 * of output rows in the following GEMV. */
__global__ static void quantize_activations_dp4a_rows_g_kernel(const float *x,
                                                              int8_t *qx,
                                                              float *qscale,
                                                              int rows, int K,
                                                              int gs_a) {
    int ng = K / gs_a;
    int rg = blockIdx.x;
    int row = rg / ng;
    int g = rg - row * ng;
    if (row >= rows) return;
    int base = row * K + g * gs_a;
    int tid = threadIdx.x;
    extern __shared__ float shf[];
    float v = 0.0f;
    for (int i = tid; i < gs_a; i += blockDim.x) {
        float a = fabsf(x[base + i]);
        if (a > v) v = a;
    }
    shf[tid] = v;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (tid < n) {
            float a = shf[tid], b = shf[tid + n];
            shf[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    __shared__ float invs;
    if (!tid) {
        float amax = shf[0];
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        qscale[row * ng + g] = s;
        invs = 1.0f / s;
    }
    __syncthreads();
    float inv = invs;
    for (int i = tid; i < gs_a; i += blockDim.x) {
        float vq = x[base + i] * inv;
        float c = vq < -127.0f ? -127.0f : (vq > 127.0f ? 127.0f : vq);
        qx[base + i] = (int8_t)rintf(c);
    }
}

extern "C" int coli_cuda_dp4a_quantize_rows_g_s(const float *x, int8_t *qx,
                                                 float *qscale, int rows, int K,
                                                 int gs_a, cudaStream_t stream) {
    if (!x || !qx || !qscale || rows <= 0 || K <= 0 || gs_a <= 0) return 0;
    if (K % gs_a != 0) return 0;
    int ng = K / gs_a;
    size_t blocks = (size_t)rows * (size_t)ng;
    if (blocks > 2147483647u) return 0;
    size_t sh_bytes = (size_t)COLI_DP4A_QUANT_THREADS * sizeof(float);
    quantize_activations_dp4a_rows_g_kernel<<<(unsigned)blocks,
                                               COLI_DP4A_QUANT_THREADS,
                                               sh_bytes, stream>>>(
        x, qx, qscale, rows, K, gs_a);
    cudaError_t e = dp4a_post_launch_error(stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] quantize_rows_g launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

/* Legacy per-row quantize — kept for the bench's per-row-vs-per-group
 * comparison. */
__global__ static void quantize_activations_dp4a_kernel(const float *x,
                                                       int8_t *qx,
                                                       float *qscale, int K) {
    extern __shared__ float shf[];
    float v = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float a = fabsf(x[i]);
        if (a > v) v = a;
    }
    shf[threadIdx.x] = v;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) {
            float a = shf[threadIdx.x], b = shf[threadIdx.x + n];
            shf[threadIdx.x] = a > b ? a : b;
        }
        __syncthreads();
    }
    __shared__ float invs;
    if (!threadIdx.x) {
        float amax = shf[0];
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        *qscale = s;
        invs = 1.0f / s;
    }
    __syncthreads();
    float inv = invs;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = x[i] * inv;
        float c = v < -127.0f ? -127.0f : (v > 127.0f ? 127.0f : v);
        qx[i] = (int8_t)rintf(c);
    }
}

extern "C" int coli_cuda_dp4a_quantize_row(const float *x, int8_t *qx,
                                           float *qscale, int K) {
    if (!x || !qx || !qscale || K <= 0) return 0;
    size_t sh_bytes = (size_t)COLI_DP4A_BLOCK_THREADS * sizeof(float);
    quantize_activations_dp4a_kernel<<<1, COLI_DP4A_BLOCK_THREADS, sh_bytes>>>(
        x, qx, qscale, K);
    cudaError_t e = dp4a_post_launch_error(0);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] quantize launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

/* ---- L2: gate+up fused (silu*up epilogue), all experts in one launch ------ *
 * grid = (O_gu, count). blockDim = 32. Each block: one (output_row, expert).
 *
 * For qwen36/Ornith gate/up (O=Ih=512, K=D=2048, gs=64): ng=32 groups per row,
 * 32 threads cover exactly one group each. The math is the same as the
 * per-expert gate_up kernel, just batched into a 2D grid.
 *
 * Output: gate_out[e*O_gu + o] = silu( sum_g dp4a_g * gsc[e,o,g] * xsc_gate[g] )
 *                              * ( sum_g dp4a_u * usc[e,o,g] * xsc_gate[g] )
 */
template <bool UseMeta>
__global__ static void dp4a_gate_up_all_kernel(const int8_t * __restrict__ qx,
                                              const float * __restrict__ xscale,
                                              const uint8_t * const * __restrict__ gw,
                                              const uint8_t * const * __restrict__ uw,
                                              const float * const * __restrict__ gsc,
                                              const float * const * __restrict__ usc,
                                              const ColiCudaDp4aExpertMeta * __restrict__ meta,
                                              float * __restrict__ gate_out,
                                              int count, int O, int K,
                                              int ng, int gs, int row_inputs) {
    int o = blockIdx.x;
    int e = blockIdx.y;
    if (o >= O || e >= count) return;
    int tid = threadIdx.x;

    const int8_t  *xrow = qx + (row_inputs ? (size_t)e * (size_t)K : 0);
    const float   *xscales = xscale + (row_inputs ? (size_t)e * (size_t)ng : 0);
    const uint8_t *gw_e;
    const uint8_t *uw_e;
    const float *gsc_e;
    const float *usc_e;
    if (UseMeta) {
        gw_e = meta[e].gw; uw_e = meta[e].uw;
        gsc_e = meta[e].gsc; usc_e = meta[e].usc;
    } else {
        gw_e = gw[e]; uw_e = uw[e];
        gsc_e = gsc[e]; usc_e = usc[e];
    }
    const uint8_t *gwrow = gw_e + (size_t)o * (size_t)((K + 1) / 2);
    const uint8_t *uwrow = uw_e + (size_t)o * (size_t)((K + 1) / 2);
    const float   *gsc_row = gsc_e + (size_t)o * (size_t)ng;
    const float   *usc_row = usc_e + (size_t)o * (size_t)ng;

    int32_t partial_g = 0;
    int32_t partial_u = 0;
    __shared__ float row_g, row_u;
    if (tid == 0) { row_g = 0.0f; row_u = 0.0f; }
    __syncthreads();

    if (tid < ng) {
        int g = tid;
        const int8_t  *xbase = xrow + (size_t)g * (size_t)gs;
        const uint8_t *gwbase = gwrow + (size_t)g * (size_t)(gs / 2);
        const uint8_t *uwbase = uwrow + (size_t)g * (size_t)(gs / 2);
        float xsc = xscales[g];

        for (int i = 0; i < gs; i += 4) {
            int8_t a0 = xbase[i + 0];
            int8_t a1 = xbase[i + 1];
            int8_t a2 = xbase[i + 2];
            int8_t a3 = xbase[i + 3];
            int32_t av = pack4_s8(a0, a1, a2, a3);

            uint8_t gb0 = gwbase[(i >> 1) + 0];
            uint8_t gb1 = gwbase[(i >> 1) + 1];
            int8_t gw0 = s4_to_s8(gb0 & 0x0F);
            int8_t gw1 = s4_to_s8((gb0 >> 4) & 0x0F);
            int8_t gw2 = s4_to_s8(gb1 & 0x0F);
            int8_t gw3 = s4_to_s8((gb1 >> 4) & 0x0F);
            int32_t gv = pack4_s8(gw0, gw1, gw2, gw3);
            partial_g = __dp4a(gv, av, partial_g);

            uint8_t ub0 = uwbase[(i >> 1) + 0];
            uint8_t ub1 = uwbase[(i >> 1) + 1];
            int8_t uw0 = s4_to_s8(ub0 & 0x0F);
            int8_t uw1 = s4_to_s8((ub0 >> 4) & 0x0F);
            int8_t uw2 = s4_to_s8(ub1 & 0x0F);
            int8_t uw3 = s4_to_s8((ub1 >> 4) & 0x0F);
            int32_t uv = pack4_s8(uw0, uw1, uw2, uw3);
            partial_u = __dp4a(uv, av, partial_u);
        }
        float fp_g = (float)partial_g * gsc_row[g] * xsc;
        float fp_u = (float)partial_u * usc_row[g] * xsc;
        for (int off = 16; off > 0; off >>= 1) {
            fp_g += __shfl_down_sync(0xffffffffu, fp_g, off, 32);
            fp_u += __shfl_down_sync(0xffffffffu, fp_u, off, 32);
        }
        if (tid == 0) {
            row_g += fp_g;
            row_u += fp_u;
        }
    }
    __syncthreads();
    if (tid == 0) {
        gate_out[(size_t)e * (size_t)O + o] =
            (row_g / (1.0f + expf(-row_g))) * row_u;
    }
}

/* Single-expert variant used by the synchronous compatibility wrapper.  The
 * batched kernel's pointer tables are device-resident; passing a one-element
 * host array to it would make the GPU dereference a host virtual address. */
__global__ static void dp4a_gate_up_s1_kernel(const int8_t * __restrict__ qx,
                                             const float * __restrict__ xscale,
                                             const uint8_t * __restrict__ gw,
                                             const uint8_t * __restrict__ uw,
                                             const float * __restrict__ gsc,
                                             const float * __restrict__ usc,
                                             float * __restrict__ gate_out,
                                             int O, int K, int ng, int gs) {
    int o = blockIdx.x;
    if (o >= O) return;
    int tid = threadIdx.x;
    const int8_t *xrow = qx;
    const uint8_t *gwrow = gw + (size_t)o * (size_t)((K + 1) / 2);
    const uint8_t *uwrow = uw + (size_t)o * (size_t)((K + 1) / 2);
    const float *gsc_row = gsc + (size_t)o * (size_t)ng;
    const float *usc_row = usc + (size_t)o * (size_t)ng;
    int32_t partial_g = 0, partial_u = 0;
    __shared__ float row_g, row_u;
    if (tid == 0) { row_g = 0.0f; row_u = 0.0f; }
    __syncthreads();
    if (tid < ng) {
        int g = tid;
        const int8_t *xbase = xrow + (size_t)g * (size_t)gs;
        const uint8_t *gwbase = gwrow + (size_t)g * (size_t)(gs / 2);
        const uint8_t *uwbase = uwrow + (size_t)g * (size_t)(gs / 2);
        float xsc = xscale[g];
        for (int i = 0; i < gs; i += 4) {
            int32_t av = pack4_s8(xbase[i], xbase[i + 1], xbase[i + 2], xbase[i + 3]);
            uint8_t gb0 = gwbase[(i >> 1) + 0], gb1 = gwbase[(i >> 1) + 1];
            int32_t gv = pack4_s8(s4_to_s8(gb0 & 15), s4_to_s8(gb0 >> 4),
                                  s4_to_s8(gb1 & 15), s4_to_s8(gb1 >> 4));
            uint8_t ub0 = uwbase[(i >> 1) + 0], ub1 = uwbase[(i >> 1) + 1];
            int32_t uv = pack4_s8(s4_to_s8(ub0 & 15), s4_to_s8(ub0 >> 4),
                                  s4_to_s8(ub1 & 15), s4_to_s8(ub1 >> 4));
            partial_g = __dp4a(gv, av, partial_g);
            partial_u = __dp4a(uv, av, partial_u);
        }
        float fp_g = (float)partial_g * gsc_row[g] * xsc;
        float fp_u = (float)partial_u * usc_row[g] * xsc;
        for (int off = 16; off > 0; off >>= 1) {
            fp_g += __shfl_down_sync(0xffffffffu, fp_g, off, 32);
            fp_u += __shfl_down_sync(0xffffffffu, fp_u, off, 32);
        }
        if (tid == 0) { row_g += fp_g; row_u += fp_u; }
    }
    __syncthreads();
    if (tid == 0)
        gate_out[o] = (row_g / (1.0f + expf(-row_g))) * row_u;
}

/* ---- L3: down, all experts in one launch, using batched INT8 input -------- *
 * grid = (O/4, count). blockDim = 32 (4 rows per warp × 8 lanes per row).
 * Each lane owns ONE 64-element group of ONE row of the silu*up input; it
 * computes the absmax locally (no cross-thread sync needed — the 64
 * elements all belong to this lane), quantizes them to INT8, then runs the
 * 16 dp4a calls for that group. Cross-lane reduction via __shfl_xor across
 * 8 lanes per row.
 *
 * For qwen36/Ornith down (O=D=2048, K=Ih=512, gs=64): ng=8 groups, lanes
 * 0..7 cover row R=0, lanes 8..15 cover R=1, etc. The activation source
 * is `gate_in[e*K + i]` (FP32, silu*up output of expert e, from L2). */
template <bool UseMeta>
__global__ static void dp4a_down_all_kernel(const int8_t * __restrict__ gate_qx,
                                           const float * __restrict__ gate_qscale,
                                           const uint8_t * const * __restrict__ w,
                                           const float * const * __restrict__ wscale,
                                           const ColiCudaDp4aExpertMeta * __restrict__ meta,
                                           float * __restrict__ y_out,
                                           int count, int O, int K,
                                           int ng, int gs) {
    int tid = threadIdx.x;
    int row_in_warp = tid >> 3;
    int group = tid & 7;
    int o4 = blockIdx.x;          /* O / 4 */
    int e = blockIdx.y;
    if (e >= count) return;
    int o = o4 * 4 + row_in_warp;
    if (o >= O) return;
    /* (debug print removed: was conflicting with the previous early-return) */

    const int8_t *xrow = gate_qx + (size_t)e * (size_t)K;
    const uint8_t *w_e;
    const float *wscale_e;
    if (UseMeta) {
        w_e = meta[e].dw; wscale_e = meta[e].dsc;
    } else {
        w_e = w[e]; wscale_e = wscale[e];
    }
    const uint8_t *wrow = w_e + (size_t)o * (size_t)((K + 1) / 2);
    const float *wsc = wscale_e + (size_t)o * (size_t)ng + group;

    int base = group * gs;
    float xsc = gate_qscale[(size_t)e * (size_t)ng + group];

    const uint8_t *wbase = wrow + (size_t)group * (size_t)(gs / 2);
    int32_t partial = 0;
    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        int i = p * 4;
        int32_t av = pack4_s8(xrow[base + i + 0], xrow[base + i + 1],
                              xrow[base + i + 2], xrow[base + i + 3]);
        uint8_t b0 = wbase[(i >> 1) + 0];
        uint8_t b1 = wbase[(i >> 1) + 1];
        int8_t w0 = s4_to_s8(b0 & 0x0F);
        int8_t w1 = s4_to_s8((b0 >> 4) & 0x0F);
        int8_t w2 = s4_to_s8(b1 & 0x0F);
        int8_t w3 = s4_to_s8((b1 >> 4) & 0x0F);
        int32_t wv = pack4_s8(w0, w1, w2, w3);
        partial = __dp4a(wv, av, partial);
    }
    float fp = (float)partial * wsc[0] * xsc;
    fp += __shfl_xor_sync(0xffffffffu, fp, 4, 32);
    fp += __shfl_xor_sync(0xffffffffu, fp, 2, 32);
    fp += __shfl_xor_sync(0xffffffffu, fp, 1, 32);
    if ((tid & 7) == 0) {
        y_out[(size_t)e * (size_t)O + o] = fp;
    }
}

/* ---- Host wrappers for the 4-launch S=1 batched path --------------------- *
 *
 * The wrappers take arrays of expert pointers and ONE launch covers all
 * experts in count. Per-expert weight strides are pulled from the
 * ColiCudaTensor (weight_bytes already encodes (I+1)/2 for the packed
 * int4 layout). */

extern "C" int coli_cuda_dp4a_gate_up_all_s(const int8_t *qx, const float *xscale,
                                              const uint8_t * const * gw,
                                              const uint8_t * const * uw,
                                              const float * const * gsc,
                                              const float * const * usc,
                                              float *gate_out,
                                              int count, int O, int K, int gs,
                                              cudaStream_t stream) {
    if (!qx || !xscale || !gw || !uw || !gsc || !usc || !gate_out) return 0;
    if (count <= 0 || O <= 0 || K <= 0) return 0;
    if (gs <= 0) return 0;                     /* guard against div-by-zero in (K + gs - 1)/gs */
    int ng = (K + gs - 1) / gs;
    if (K % 4 != 0 || ng > 32) return 0;
    if (K % gs != 0) return 0;
    /* Strategy-B gate_up_all kernel assumes one thread per group (ng threads
     * active per row, the rest idle); it also assumes ng<=32 (single warp). */
    if (ng > 32) return 0;
    dim3 grid((unsigned)O, (unsigned)count);
    dp4a_gate_up_all_kernel<false><<<grid, 32, 0, stream>>>(
        qx, xscale, gw, uw, gsc, usc, nullptr, gate_out, count, O, K, ng, gs, 0);
    cudaError_t e = dp4a_post_launch_error(stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] gate_up_all launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

extern "C" int coli_cuda_dp4a_gate_up_rows_s(const int8_t *qx, const float *xscale,
                                              const uint8_t * const *gw,
                                              const uint8_t * const *uw,
                                              const float * const *gsc,
                                              const float * const *usc,
                                              float *gate_out, int count, int O,
                                              int K, int gs, cudaStream_t stream) {
    if (!qx || !xscale || !gw || !uw || !gsc || !usc || !gate_out) return 0;
    if (count <= 0 || count > 64 || O <= 0 || K <= 0 || gs <= 0 || K % 4 != 0 || K % gs != 0) return 0;
    int ng = K / gs;
    if (ng > 32) return 0;
    dim3 grid((unsigned)O, (unsigned)count);
    dp4a_gate_up_all_kernel<false><<<grid, 32, 0, stream>>>(
        qx, xscale, gw, uw, gsc, usc, nullptr, gate_out, count, O, K, ng, gs, 1);
    cudaError_t e = dp4a_post_launch_error(stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] gate_up_rows launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

extern "C" int coli_cuda_dp4a_gate_up_all(const int8_t *qx, const float *xscale,
                                          const uint8_t * const * gw,
                                          const uint8_t * const * uw,
                                          const float * const * gsc,
                                          const float * const * usc,
                                          float *gate_out,
                                          int count, int O, int K, int gs) {
    return coli_cuda_dp4a_gate_up_all_s(qx, xscale, gw, uw, gsc, usc, gate_out,
                                       count, O, K, gs, 0);
}

extern "C" int coli_cuda_dp4a_gate_up_meta_s(
        const int8_t *qx, const float *xscale,
        const ColiCudaDp4aExpertMeta *meta, float *gate_out,
        int count, int O, int K, int gs, cudaStream_t stream) {
    if (!qx || !xscale || !meta || !gate_out) return 0;
    if (count <= 0 || count > 64 || O <= 0 || K <= 0 || gs <= 0) return 0;
    int ng = (K + gs - 1) / gs;
    if (K % 4 != 0 || K % gs != 0 || ng > 32) return 0;
    dim3 grid((unsigned)O, (unsigned)count);
    dp4a_gate_up_all_kernel<true><<<grid, 32, 0, stream>>>(
        qx, xscale, nullptr, nullptr, nullptr, nullptr, meta, gate_out,
        count, O, K, ng, gs, 0);
    cudaError_t e = dp4a_post_launch_error(stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] gate_up_meta launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

extern "C" int coli_cuda_dp4a_down_all_s(const int8_t *gate_qx,
                                         const float *gate_qscale,
                                         const uint8_t * const * w,
                                         const float * const * wscale,
                                         float *y_out,
                                         int count, int O, int K, int gs,
                                         cudaStream_t stream) {
    if (!gate_qx || !gate_qscale || !w || !wscale || !y_out) return 0;
    if (count <= 0 || O <= 0 || K <= 0) return 0;
    if (gs <= 0) return 0;                     /* gs==0 would divide by zero in (K + gs - 1) / gs */
    int ng = (K + gs - 1) / gs;
    if (K % gs != 0) return 0;
    /* Strategy B (8 lanes per row × 4 rows per warp) is hard-coded:
     *   gs must be 64 (so each lane owns exactly one 64-element group of one row)
     *   ng must be 8 (so one warp covers 8 lanes × 1 group = 8 groups = one row)
     *   O must be divisible by 4 (so 4 rows fit in one warp)
     * Any other (K, gs) combination must fall back. The header comment
     * promised ng != 8 → A fallback but the legacy strategy-2 dispatch
     * did NOT honor that. Treat all unknown geometry as a refusal. */
    if (gs != 64 || ng != 8) return 0;
    if (O % 4 != 0) return 0;
    int O4 = O / 4;
    dim3 grid((unsigned)O4, (unsigned)count);
    dp4a_down_all_kernel<false><<<grid, 32, 0, stream>>>(
        gate_qx, gate_qscale, w, wscale, nullptr, y_out, count, O, K, ng, gs);
    cudaError_t e = dp4a_post_launch_error(stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] down_all launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

extern "C" int coli_cuda_dp4a_down_all(const int8_t *gate_qx,
                                       const float *gate_qscale,
                                       const uint8_t * const * w,
                                       const float * const * wscale,
                                       float *y_out,
                                       int count, int O, int K, int gs) {
    return coli_cuda_dp4a_down_all_s(gate_qx, gate_qscale, w, wscale,
                                     y_out, count, O, K, gs, 0);
}

extern "C" int coli_cuda_dp4a_down_meta_s(
        const int8_t *gate_qx, const float *gate_qscale,
        const ColiCudaDp4aExpertMeta *meta, float *y_out,
        int count, int O, int K, int gs, cudaStream_t stream) {
    if (!gate_qx || !gate_qscale || !meta || !y_out) return 0;
    if (count <= 0 || count > 64 || O <= 0 || K <= 0 || gs <= 0) return 0;
    int ng = (K + gs - 1) / gs;
    if (K % gs != 0 || gs != 64 || ng != 8 || O % 4 != 0) return 0;
    int O4 = O / 4;
    dim3 grid((unsigned)O4, (unsigned)count);
    dp4a_down_all_kernel<true><<<grid, 32, 0, stream>>>(
        gate_qx, gate_qscale, nullptr, nullptr, meta, y_out,
        count, O, K, ng, gs);
    cudaError_t e = dp4a_post_launch_error(stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] down_meta launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

/* ---- Per-expert wrappers (kept for the bench harness, not used by the
 * engine integration after the 4-launch redesign). */
extern "C" int coli_cuda_dp4a_down_s1_s(const int8_t *x_dev,
                                        const float *xscale_dev,
                                        const uint8_t *w_dev,
                                        const float *wscale_dev,
                                        float *y_dev,
                                        int O, int K, int gs, int strategy);

extern "C" int coli_cuda_dp4a_down_s1(const int8_t *x_dev,
                                      const float *xscale_dev,
                                      const uint8_t *w_dev,
                                      const float *wscale_dev,
                                      float *y_dev,
                                      int O, int K, int gs) {
    return coli_cuda_dp4a_down_s1_s(x_dev, xscale_dev, w_dev, wscale_dev,
                                   y_dev, O, K, gs, /*strategy*/ 0);
}

__global__ static void dp4a_down_s1_warp(const int8_t * __restrict__ x,
                                         const float * __restrict__ xscale,
                                         const uint8_t * __restrict__ w,
                                         const float * __restrict__ wscale,
                                         float * __restrict__ y,
                                         int O, int K, int ng, int gs) {
    int o = blockIdx.x;
    if (o >= O) return;
    int tid = threadIdx.x;
    const int8_t  *xrow = x;
    const uint8_t *wrow = w + (size_t)o * (size_t)((K + 1) / 2);
    const float   *wsc  = wscale + (size_t)o * (size_t)ng;
    int32_t partial = 0;
    __shared__ float row_acc;
    if (tid == 0) row_acc = 0.0f;
    __syncthreads();
    for (int g = 0; g < ng; ++g) {
        const int8_t  *xbase = xrow + (size_t)g * (size_t)gs;
        const uint8_t *wbase = wrow + (size_t)g * (size_t)(gs / 2);
        float xsc = xscale[g];
        partial = 0;
        for (int i4 = tid * 4; i4 < gs; i4 += 32 * 4) {
            int8_t a0 = xbase[i4 + 0];
            int8_t a1 = xbase[i4 + 1];
            int8_t a2 = xbase[i4 + 2];
            int8_t a3 = xbase[i4 + 3];
            int32_t av = pack4_s8(a0, a1, a2, a3);
            uint8_t b0 = wbase[(i4 >> 1) + 0];
            uint8_t b1 = wbase[(i4 >> 1) + 1];
            int8_t w0 = s4_to_s8(b0 & 0x0F);
            int8_t w1 = s4_to_s8((b0 >> 4) & 0x0F);
            int8_t w2 = s4_to_s8(b1 & 0x0F);
            int8_t w3 = s4_to_s8((b1 >> 4) & 0x0F);
            int32_t wv = pack4_s8(w0, w1, w2, w3);
            partial = __dp4a(wv, av, partial);
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, off, 32);
        if (tid == 0) {
            float fp = (float)partial * wsc[g] * xsc;
            row_acc += fp;
        }
        __syncthreads();
    }
    if (tid == 0) y[o] = row_acc;
}

__global__ static void dp4a_down_s1_octant(const int8_t * __restrict__ x,
                                          const float * __restrict__ xscale,
                                          const uint8_t * __restrict__ w,
                                          const float * __restrict__ wscale,
                                          float * __restrict__ y,
                                          int O, int K, int ng, int gs) {
    int tid = threadIdx.x;
    int row_in_warp = tid >> 3;
    int group = tid & 7;
    int o = blockIdx.x * 4 + row_in_warp;
    if (o >= O) return;
    const int8_t  *xrow = x;
    const uint8_t *wrow = w + (size_t)o * (size_t)((K + 1) / 2);
    const float   *wsc  = wscale + (size_t)o * (size_t)ng + group;
    float xsc = xscale[group];
    const int8_t  *xbase = xrow + (size_t)group * (size_t)gs;
    const uint8_t *wbase = wrow + (size_t)group * (size_t)(gs / 2);
    int32_t partial = 0;
    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        int i = p * 4;
        int8_t a0 = xbase[i + 0];
        int8_t a1 = xbase[i + 1];
        int8_t a2 = xbase[i + 2];
        int8_t a3 = xbase[i + 3];
        int32_t av = pack4_s8(a0, a1, a2, a3);
        uint8_t b0 = wbase[(i >> 1) + 0];
        uint8_t b1 = wbase[(i >> 1) + 1];
        int8_t w0 = s4_to_s8(b0 & 0x0F);
        int8_t w1 = s4_to_s8((b0 >> 4) & 0x0F);
        int8_t w2 = s4_to_s8(b1 & 0x0F);
        int8_t w3 = s4_to_s8((b1 >> 4) & 0x0F);
        int32_t wv = pack4_s8(w0, w1, w2, w3);
        partial = __dp4a(wv, av, partial);
    }
    float fp = (float)partial * wsc[0] * xsc;
    fp += __shfl_xor_sync(0xffffffffu, fp, 4, 32);
    fp += __shfl_xor_sync(0xffffffffu, fp, 2, 32);
    fp += __shfl_xor_sync(0xffffffffu, fp, 1, 32);
    if ((tid & 7) == 0) y[o] = fp;
}

extern "C" int coli_cuda_dp4a_down_s1_s(const int8_t *x_dev,
                                        const float *xscale_dev,
                                        const uint8_t *w_dev,
                                        const float *wscale_dev,
                                        float *y_dev,
                                        int O, int K, int gs, int strategy) {
    if (!x_dev || !xscale_dev || !w_dev || !wscale_dev || !y_dev) return 0;
    if (O <= 0 || K <= 0) return 0;
    int ng = gs > 0 ? (K + gs - 1) / gs : 1;
    if (K % 4 != 0) return 0;
    if (ng > 1 && (K % gs != 0)) return 0;
    int use_b = (strategy == 0 ? (ng == 8) : (strategy == 2));
    if (use_b) {
        int O4 = O & ~3;
        if (O4 > 0) {
            dp4a_down_s1_octant<<<O4 / 4, 32>>>(
                x_dev, xscale_dev, w_dev, wscale_dev, y_dev, O, K, ng, gs);
        }
        for (int o = O4; o < O; o += 32) {
            int rows = O - o; if (rows > 32) rows = 32;
            dp4a_down_s1_warp<<<rows, 32, 0>>>(
                x_dev, xscale_dev, w_dev + (size_t)o * (size_t)((K + 1) / 2),
                wscale_dev + (size_t)o * (size_t)ng,
                y_dev + o, rows, K, ng, gs);
        }
    } else {
        dp4a_down_s1_warp<<<O, 32>>>(
            x_dev, xscale_dev, w_dev, wscale_dev, y_dev, O, K, ng, gs);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] down_s1 launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

/* Legacy per-expert gate_up wrapper (count=1 over the batched API). Used by
 * the sync path of expert_group_impl and by the standalone bench. */
extern "C" int coli_cuda_dp4a_gate_up_s1(const int8_t *x_dev,
                                         const float *xscale_dev,
                                         const uint8_t *gw_dev,
                                         const uint8_t *uw_dev,
                                         const float *gsc_dev,
                                         const float *usc_dev,
                                         float *gate_dev,
                                         float *up_dev,
                                         int O, int K, int gs) {
    if (!x_dev || !xscale_dev || !gw_dev || !uw_dev || !gsc_dev || !usc_dev ||
        !gate_dev) return 0;
    if (O <= 0 || K <= 0 || gs <= 0) return 0;
    int ng = (K + gs - 1) / gs;
    if (K % 4 != 0 || ng > 32) return 0;
    if (K % gs != 0) return 0;
    dp4a_gate_up_s1_kernel<<<(unsigned)O, 32>>>(
        x_dev, xscale_dev, gw_dev, uw_dev, gsc_dev, usc_dev, gate_dev,
        O, K, ng, gs);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[dp4a] gate_up_s1 launch: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

/* ---- gate query ----------------------------------------------------------- */
extern "C" int coli_cuda_dp4a_supported(int device) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    (void)device;
    return 0;
#else
    if (device < 0) return 0;
    const char *e = std::getenv("COLI_CUDA_DP4A");
    if (!e || *e != '1') return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    int major = prop.major, minor = prop.minor;
    return (major == 6 && minor == 1) ? 1 : 0;
#endif
}
