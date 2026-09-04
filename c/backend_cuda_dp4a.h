/* backend_cuda_dp4a.h — Pascal SM61 DP4A prototype for the grouped-int4 (fmt=4)
 * expert projection that the W4 GEMV path serves today.
 *
 * The CUDA backend's existing expert path uses 256-thread blocks where each
 * thread unpacks 1 byte = 2 nibbles and FMAs in FP32 with inline per-group
 * scales. Pascal has no tensor cores, but `__dp4a()` packs four INT8×INT8 MACs
 * into one SASS instruction (DP4A, sm_61+) — same register-file traffic as one
 * FFMA, but four times the dot-products per instruction issued.
 *
 * This file is the FIRST prototype for one projection only (down: O=D, K=Ih),
 * gated on COLI_CUDA_DP4A and on the device actually being SM61. The goal is
 * to measure whether nibble-unpack + activation-quantization + DP4A on this
 * single kernel beats the FP32 FMA path, NOT to replace the whole engine. The
 * existing W4 path remains the default and the fallback.
 *
 * IMPORTANT: this kernel keeps weights packed in VRAM. There is no INT8 mirror
 * anywhere — the only device-side INT8 buffer is the K-element activation
 * scratch `qact` filled by quantize_activations_dp4a before the GEMV. */
#ifndef COLIBRI_BACKEND_CUDA_DP4A_H
#define COLIBRI_BACKEND_CUDA_DP4A_H

#include <stddef.h>
#include <stdint.h>

#include "backend_cuda.h"

/* One compact device-side metadata record for a routed expert.  The tensor
 * addresses are stable while a resident tensor remains in its cache slot;
 * the selected record set is the genuinely dynamic per-call value.  Keeping
 * the six addresses together avoids six separate runtime memcpy submissions
 * without changing the DP4A arithmetic or the routing order. */
typedef struct ColiCudaDp4aExpertMeta {
    const uint8_t *gw;
    const uint8_t *uw;
    const uint8_t *dw;
    const float *gsc;
    const float *usc;
    const float *dsc;
} ColiCudaDp4aExpertMeta;

#ifdef __cplusplus
extern "C" {
#endif

/* Quantize an FP32 activation row of length K to INT8 with a symmetric
 * PER-GROUP scale aligned with the weight group size (gs=64 for qwen36/Ornith).
 *
 * Output: qx[K] bytes (INT8), qscale[K/gs] floats (one per group).
 *
 * Why per-group: per-row scales clip values to the global amax, throwing away
 * precision in groups whose local distribution is tighter. Per-group (gs_a=64)
 * keeps the per-group relative error bounded by 1/127 regardless of how heavy
 * the global tail is. It also lets the GEMV kernel fold the activation scale
 * into the same per-group partial multiply, so the accumulation structure is
 * identical to the FP32 W4 path.
 *
 * Cost: one parallel absmax per group instead of one per row. The activation
 * is shared across all output rows and experts, so this cost is paid ONCE per
 * layer per token, not once per output row.
 *
 * Activation is shared across experts in a routed expert call (the same row
 * of `x` feeds every expert the router chose), so quantizing once into a
 * reusable per-row A8 buffer is strictly better than fusing the quant into
 * each output-row GEMV thread. The bench and engine integration both rely on
 * this property. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_quantize_row_g(const float *x, int8_t *qx,
                                                     float *qscale, int K, int gs_a);
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_quantize_row_g_s(const float *x, int8_t *qx,
                                                     float *qscale, int K, int gs_a,
                                                     cudaStream_t stream);

/* Batch form used for the down projection: rows are contiguous, qx has
 * rows*K bytes and qscale has rows*(K/gs_a) scales. Each input row is
 * quantized once, independent of the number of output rows in the GEMV. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_quantize_rows_g_s(const float *x, int8_t *qx,
                                                       float *qscale, int rows,
                                                       int K, int gs_a,
                                                       cudaStream_t stream);

/* Backwards-compatible per-row quantize (one scale for the whole K row).
 * Use only when gs_a == K or for comparison. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_quantize_row(const float *x, int8_t *qx,
                                                  float *qscale, int K);

/* Down projection, single S=1 row, group-scaled activation (qscale is per-group,
 * length K/gs). One block per output row, 64 threads (2 warps); strategy A.
 * For K=512 gs=64, the dispatch site prefers strategy B (8 lanes/row, 4 rows
 * per warp) — see coli_cuda_dp4a_down_s1_s.
 *
 * Math:
 *   for each output row o:
 *     y = 0
 *     for each group g:
 *       partial = 0
 *       for k in [g*gs, (g+1)*gs), 4 at a time:
 *         partial = __dp4a(w4, a4, partial)   (CUDA wrapper: result = dot4(a,b) + c)
 *       y += float(partial) * weight_scale[o, g] * activation_scale[g]
 *     output[o] = y
 *
 * The activation scale is folded at the group boundary, the same place the
 * weight scale folds, so the FP32 accumulator sees both together. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_down_s1(const int8_t *x_dev,
                                              const float *xscale_dev,
                                              const uint8_t *w_dev,
                                              const float *wscale_dev,
                                              float *y_dev,
                                              int O, int K, int gs);

/* Same as down_s1 with explicit strategy override:
 *   strategy=0  auto (B when ng==8, A otherwise)
 *   strategy=1  force strategy A
 *   strategy=2  force strategy B (falls back to A for ng != 8) */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_down_s1_s(const int8_t *x_dev,
                                              const float *xscale_dev,
                                              const uint8_t *w_dev,
                                              const float *wscale_dev,
                                              float *y_dev,
                                              int O, int K, int gs, int strategy);

/* Fused gate+up projection, all experts in one launch (decode hot path).
 * Reads x_dev (INT8, length K, shared across all experts), xscale_dev
 * (per-group, length K/gs), and per-expert weight/scale arrays.
 *
 * grid = (O_gu, count). Each block: one (output_row, expert).
 * Writes silu(gate) * up into gate_out (length count*O_gu); up_out is NOT
 * written (silu_mul epilogue fused).
 *
 * The "all experts in one launch" form replaces the previous per-expert
 * loop pattern (which cost 2 launches per expert). Combined with a shared
 * input quantize and a single down launch using pre-quantized down input, the
 * full S=1 expert pipeline runs in 4 CUDA launches total, matching the W4
 * path's 2-launch structure within two quantizers. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_gate_up_all(const int8_t *qx,
                                                  const float *xscale,
                                                  const uint8_t * const *gw,
                                                  const uint8_t * const *uw,
                                                  const float * const *gsc,
                                                  const float * const *usc,
                                                  float *gate_out,
                                                  int count, int O, int K, int gs);
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_gate_up_all_s(const int8_t *qx,
                                                  const float *xscale,
                                                  const uint8_t * const *gw,
                                                  const uint8_t * const *uw,
                                                  const float * const *gsc,
                                                  const float * const *usc,
                                                  float *gate_out,
                                                  int count, int O, int K, int gs,
                                                  cudaStream_t stream);
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_gate_up_rows_s(const int8_t *qx,
                                                  const float *xscale,
                                                  const uint8_t * const *gw,
                                                  const uint8_t * const *uw,
                                                  const float * const *gsc,
                                                  const float * const *usc,
                                                  float *gate_out,
                                                  int count, int O, int K, int gs,
                                                  cudaStream_t stream);

/* Down projection, all experts in one launch. Reads gate_qx (INT8, length
 * count*K) and gate_qscale (per-expert per-group scales, length count*K/gs),
 * produced by coli_cuda_dp4a_quantize_rows_g_s, plus per-expert weight/scale
 * arrays. Writes y_out (length count*O).
 *
 * Requires ng == 8 (K=512, gs=64 — the qwen36/Ornith down shape) and O % 4
 * == 0 (B-strategy 4-rows-per-warp). The engine integration should fall
 * back to the W4 path when these don't hold. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_down_all(const int8_t *gate_qx,
                                                const float *gate_qscale,
                                                const uint8_t * const *w,
                                                const float * const *wscale,
                                                float *y_out,
                                                int count, int O, int K, int gs);
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_down_all_s(const int8_t *gate_qx,
                                                const float *gate_qscale,
                                                const uint8_t * const *w,
                                                const float * const *wscale,
                                                float *y_out,
                                                int count, int O, int K, int gs,
                                                cudaStream_t stream);

/* Metadata-tuple variants used by the resident path.  The device metadata
 * allocation is persistent; at most one compact H2D update is submitted when
 * the selected expert tuple changes. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_gate_up_meta_s(const int8_t *qx,
                                                  const float *xscale,
                                                  const ColiCudaDp4aExpertMeta *meta,
                                                  float *gate_out,
                                                  int count, int O, int K, int gs,
                                                  cudaStream_t stream);
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_down_meta_s(const int8_t *gate_qx,
                                                const float *gate_qscale,
                                                const ColiCudaDp4aExpertMeta *meta,
                                                float *y_out,
                                                int count, int O, int K, int gs,
                                                cudaStream_t stream);

/* Per-expert wrappers (legacy, kept for the bench). The engine integration
 * uses the *all variants for batched launch amortization. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_gate_up_s1(const int8_t *x_dev,
                                                  const float *xscale_dev,
                                                  const uint8_t *gw_dev,
                                                  const uint8_t *uw_dev,
                                                  const float *gsc_dev,
                                                  const float *usc_dev,
                                                  float *gate_dev,
                                                  float *up_dev,
                                                  int O, int K, int gs);

/* Gate query (host-side): 1 iff device is SM61 AND COLI_CUDA_DP4A is set. */
COLI_CUDA_DLLEXPORT int coli_cuda_dp4a_supported(int device);

/* Internal graph-capture guard.  CUDA rejects post-launch error queries while
 * a stream is being captured; the backend toggles this only around capture. */
COLI_CUDA_DLLEXPORT void coli_cuda_dp4a_capture_mode(int active);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_BACKEND_CUDA_DP4A_H */
