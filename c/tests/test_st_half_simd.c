/* st.h's bf16_to_f32_batch / f16_to_f32_batch (SIMD, used by st_read_f32 and
 * st_read_slice_f32) must be bit-for-bit identical to the scalar
 * bf16_to_f32 / f16_to_f32 they replace in those two call sites -- pure bit
 * manipulation (bf16) or an exact renormalization (f16, see st.h's comment
 * above f16_to_f32_batch for why the subnormal case is exact), no rounding
 * tree to reassociate, so this is a memcmp-exact gate like
 * tests/test_olmoe_dot_i8_16.c, not a tolerance one.
 *
 * Exhaustive, not sampled: both formats are 16 bits wide, so all 65536
 * patterns (every sign/exponent/mantissa combination, including every NaN
 * payload and both infinities) are cheap to enumerate completely instead of
 * random + edge cases. This is a strictly stronger claim than sampling and
 * costs a fraction of a second.
 *
 * Build: make tests/test_st_half_simd             (native tier: AVX2 or scalar)
 *        make tests/test_st_half_simd_sse41        (forces the SSE4.1 body)
 */
#include "../st.h"
#include <stdio.h>
#include <string.h>

static int failures;

static void check_bf16(void) {
    uint16_t in[65536]; float scalar_out[65536], simd_out[65536];
    for (int v = 0; v < 65536; v++) in[v] = (uint16_t)v;
    for (int v = 0; v < 65536; v++) scalar_out[v] = bf16_to_f32(in[v]);
    bf16_to_f32_batch(in, simd_out, 65536);
    if (memcmp(scalar_out, simd_out, sizeof(scalar_out))) {
        int shown = 0, different = 0;
        for (int v = 0; v < 65536; v++) {
            uint32_t sb, mb; memcpy(&sb, &scalar_out[v], 4); memcpy(&mb, &simd_out[v], 4);
            if (sb == mb) continue;
            different++;
            if (shown++ < 10) fprintf(stderr, "bf16 MISMATCH v=0x%04x scalar=0x%08x simd=0x%08x\n", v, sb, mb);
        }
        fprintf(stderr, "bf16_to_f32_batch: %d/65536 mismatches\n", different);
        failures++;
    } else {
        printf("bf16_to_f32_batch: ALL PASS, 65536/65536 values bit-exact\n");
    }
}

static void check_f16(void) {
    uint16_t in[65536]; float scalar_out[65536], simd_out[65536];
    for (int v = 0; v < 65536; v++) in[v] = (uint16_t)v;
    for (int v = 0; v < 65536; v++) scalar_out[v] = f16_to_f32(in[v]);
    f16_to_f32_batch(in, simd_out, 65536);
    if (memcmp(scalar_out, simd_out, sizeof(scalar_out))) {
        /* Compare bit patterns directly (not float ==): several f16 inputs
         * are NaN, and NaN != NaN under float comparison even when the two
         * bit patterns are identical. */
        int shown = 0, different = 0;
        for (int v = 0; v < 65536; v++) {
            uint32_t sb, mb; memcpy(&sb, &scalar_out[v], 4); memcpy(&mb, &simd_out[v], 4);
            if (sb == mb) continue;
            different++;
            if (shown++ < 10) fprintf(stderr, "f16 MISMATCH v=0x%04x scalar=0x%08x simd=0x%08x\n", v, sb, mb);
        }
        if (different) { fprintf(stderr, "f16_to_f32_batch: %d/65536 mismatches\n", different); failures++; }
        else printf("f16_to_f32_batch: ALL PASS, 65536/65536 values bit-exact\n");
    } else {
        printf("f16_to_f32_batch: ALL PASS, 65536/65536 values bit-exact\n");
    }
}

/* Also exercise the odd-length scalar tail explicitly: batch sizes not a
 * multiple of the SIMD width (8 for AVX2, 4 for SSE4.1) are exactly what
 * st_read_f32/st_read_slice_f32 hand in for most real tensors. */
static void check_tail_sizes(void) {
    int sizes[] = {0,1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,63,64,65,127,4096,4097};
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int n = sizes[s];
        uint16_t *in = malloc((size_t)n * sizeof(uint16_t));
        float *sref = malloc((size_t)n * sizeof(float)), *sgot = malloc((size_t)n * sizeof(float));
        float *fref = malloc((size_t)n * sizeof(float)), *fgot = malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) in[i] = (uint16_t)((i * 2654435761u) & 0xFFFF);
        for (int i = 0; i < n; i++) { sref[i] = bf16_to_f32(in[i]); fref[i] = f16_to_f32(in[i]); }
        bf16_to_f32_batch(in, sgot, n);
        f16_to_f32_batch(in, fgot, n);
        if (n && (memcmp(sref, sgot, (size_t)n*sizeof(float)) || memcmp(fref, fgot, (size_t)n*sizeof(float)))) {
            fprintf(stderr, "tail-size n=%d: mismatch\n", n);
            failures++;
        }
        free(in); free(sref); free(sgot); free(fref); free(fgot);
    }
    if (!failures) printf("tail sizes: ALL PASS (%zu sizes incl. n=0)\n", sizeof(sizes)/sizeof(sizes[0]));
}

int main(void) {
    check_bf16();
    check_f16();
    check_tail_sizes();
    if (failures) { fprintf(stderr, "test_st_half_simd: %d failure(s)\n", failures); return 1; }
    puts("test_st_half_simd: ok");
    return 0;
}
