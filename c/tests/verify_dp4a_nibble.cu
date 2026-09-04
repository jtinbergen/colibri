/* tests/verify_dp4a_nibble.cu — exhaustive 16-code test for the device-resident
 * nibble representation.
 *
 * The upload path runs `offset_to_signed_s4` which XORs each packed byte by
 * 0x88 (per-nibble XOR 0x8). XOR 0x8 is the EXACT transform between offset-
 * binary and two's-complement-of-the-low-bit-of-s4, so the SAME byte value
 * decoded as `n & 8 ? n - 16 : n` (existing kernel's decode) yields a
 * different number than `n - 8` (what I had in the first DP4A prototype).
 *
 * This test exhaustively enumerates nibble values 0..15 and the signed-byte
 * decode the device produces, and compares them against the CPU quant
 * decoder. The output is the single source of truth for which decode the
 * DP4A kernel must use.
 *
 * Build: nvcc -O3 -std=c++17 -arch=sm_61 -DCOLI_CUDA_BUILDING_DLL=0 tests/verify_dp4a_nibble.cu -o tests/verify_dp4a_nibble -lcudart
 */
#include "../backend_cuda.h"
#include "../backend_cuda_dp4a.h"
#include "../backend_gpu_compat.h"
#include "../backend_cuda.cu"     /* brings in offset_to_signed_s4 + weight_at */

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* __CUDA_ARCH__ is only defined in the device pass; check it there. */
__device__ void arch_check(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 610
#error "build with -arch=sm_61 (Pascal); DP4A on sm_61 needs __dp4a()"
#endif
}

/* A simple "decode both nibbles of every byte, write to an output buffer"
 * kernel. Each block handles N bytes; thread t handles byte t.
 * Writes s8_lo[N] and s8_hi[N] (signed INT8 representations of the
 * low/high nibbles respectively), using the SAME decode weight_at uses
 * (`n & 8 ? n - 16 : n`). This is the reference decode. */
__global__ static void decode_nibbles_kernel(const uint8_t *w, int8_t *s8_lo,
                                             int8_t *s8_hi, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    uint8_t b = w[t];
    int n0 = b & 0x0F, n1 = (b >> 4) & 0x0F;
    s8_lo[t] = (int8_t)(n0 & 8 ? n0 - 16 : n0);
    s8_hi[t] = (int8_t)(n1 & 8 ? n1 - 16 : n1);
}

/* Same decode but with `n - 8` — what the first DP4A prototype was using.
 * Kept here so we can show the diff explicitly. */
__global__ static void decode_nibbles_naive_kernel(const uint8_t *w,
                                                  int8_t *s8_lo,
                                                  int8_t *s8_hi, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    uint8_t b = w[t];
    int n0 = b & 0x0F, n1 = (b >> 4) & 0x0F;
    s8_lo[t] = (int8_t)(n0 - 8);
    s8_hi[t] = (int8_t)(n1 - 8);
}

int main() {
    fprintf(stderr, "== verify_dp4a_nibble ==\n");
    fprintf(stderr, "__CUDA_ARCH__=%d\n\n",
#ifdef __CUDA_ARCH__
            __CUDA_ARCH__
#else
            -1
#endif
    );

    /* Build a host buffer that, after the XOR-0x88 upload transform, contains
     * one byte for each possible low/high nibble pair (256 entries). We do
     * NOT pre-XOR the host bytes — we want the kernel to receive them as if
     * the upload transform had run, so we XOR here. */
    constexpr int N = 256;
    uint8_t hPre[N];
    for (int b = 0; b < N; b++) hPre[b] = (uint8_t)b;

    /* Apply the upload transform: XOR 0x88 (= 0x8 per nibble). */
    uint8_t hPost[N];
    for (int b = 0; b < N; b++) hPost[b] = (uint8_t)(hPre[b] ^ 0x88);

    /* Copy POST-XOR form to device (this is what the kernel sees at runtime). */
    uint8_t *dW;
    int8_t *dLo, *dHi, *dLoN, *dHiN;
    cudaMalloc(&dW, N);
    cudaMalloc(&dLo, N);
    cudaMalloc(&dHi, N);
    cudaMalloc(&dLoN, N);
    cudaMalloc(&dHiN, N);
    cudaMemcpy(dW, hPost, N, cudaMemcpyHostToDevice);

    dim3 grid((unsigned)((N + 255) / 256));
    decode_nibbles_kernel<<<grid, 256>>>(dW, dLo, dHi, N);
    decode_nibbles_naive_kernel<<<grid, 256>>>(dW, dLoN, dHiN, N);
    cudaDeviceSynchronize();

    int8_t hLo[N], hHi[N], hLoN[N], hHiN[N];
    cudaMemcpy(hLo, dLo, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hHi, dHi, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hLoN, dLoN, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hHiN, dHiN, N, cudaMemcpyDeviceToHost);

    /* The CORRECT expected value is the signed 4-bit value the weight
     * represents, which is what BOTH the CPU matmul_i4_grouped AND the
     * existing device kernels produce. The CPU path uses `n - 8` on the
     * pre-XOR nibble; the device path uses `n & 8 ? n - 16 : n` on the
     * post-XOR nibble. These two decodes give the SAME signed value
     * because XOR-0x88 is self-inverse (it flips the same bit both decodes
     * use to distinguish positive from negative). So:
     *   expect_signed = pre_nib - 8   (CPU reference)
     * The test asserts the device decode EQUALS this expected value.
     *
     * The "naive" `n - 8` decode applied to the post-XOR nibble gives the
     * NEGATION of the expected value (sign-flipped) — that's the WRONG
     * decode for the device byte form. */
    fprintf(stderr, "byte  pre_nib  post_nib  expect  dev_correct  dev_naive\n");
    int wrong = 0;
    for (int b = 0; b < 16; b++) {
        int v_pre  = b & 0x0F;
        int v_post = (hPost[b] & 0x0F);
        int expect = v_pre - 8;                  /* CPU reference (signed value) */
        int dev_c  = (int)hLo[b];                /* device: n & 8 ? n - 16 : n */
        int dev_n  = (int)hLoN[b];               /* device: n - 8 (naive, sign-flipped) */
        fprintf(stderr, " 0x%02X   %2d       %2d       %4d   %4d        %4d %s\n",
                b, v_pre, v_post, expect, dev_c, dev_n,
                (dev_c == expect ? "OK" : "WRONG"));
        if (dev_c != expect) wrong++;
    }

    fprintf(stderr, "\n== verdict ==\n");
    if (wrong == 0) {
        fprintf(stderr, "PASS: device decode `n & 8 ? n - 16 : n` matches CPU `n - 8` on the pre-XOR form.\n");
        fprintf(stderr, "      DP4A kernel MUST use this decode (i.e. sign-extend, NOT `n - 8`).\n");
    } else {
        fprintf(stderr, "FAIL: %d nibbles decoded incorrectly.\n", wrong);
        fprintf(stderr, "      The upload XOR convention has changed; re-derive.\n");
    }

    cudaFree(dW); cudaFree(dLo); cudaFree(dHi); cudaFree(dLoN); cudaFree(dHiN);
    return wrong == 0 ? 0 : 1;
}
