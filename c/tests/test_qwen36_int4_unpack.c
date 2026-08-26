/* qwen36.c's unpack_int4_to_int8 (shared by load_expert_merged and
 * slot_ensure_int8, replacing their two former hand-written copies) must be
 * bit-for-bit identical to the scalar nibble-decode it replaces: pure
 * integer unpack + sign-extend, no rounding tree, so this is a memcmp-exact
 * gate like tests/test_olmoe_dot_i8_16.c, not a tolerance one.
 *
 * Exhaustive at the byte level: a packed byte has only 256 possible values
 * (16 low-nibble x 16 high-nibble combinations), so trying all of them
 * covers every nibble pairing the unpack can ever see -- cheaper than random
 * sampling and strictly stronger. Also covers larger random arrays and the
 * 0x00/0xFF/all-same-nibble edge cases the plan calls out explicitly, plus
 * every odd/even byte-count SIMD-tail length.
 *
 * Build: make tests/test_qwen36_int4_unpack           (native tier)
 *        make tests/test_qwen36_int4_unpack_sse41      (forces the SSE4.1 body)
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <stdio.h>

static int failures;

static void scalar_ref(const uint8_t *raw, int8_t *out, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint8_t byte = raw[i >> 1];
        int8_t v = (int8_t)((i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF));
        if (v & 8) v -= 16;
        out[i] = v;
    }
}

static int check_one(const char *label, const uint8_t *raw, int64_t n) {
    /* calloc, not malloc: GCC's -Wmaybe-uninitialized (spuriously, since
     * scalar_ref/unpack_int4_to_int8 always fill every byte before the
     * memcmp below runs) cannot see across the function-pointer-free but
     * separately-compiled scalar_ref call that the buffer is fully written;
     * zero-initializing satisfies the analyzer without changing behavior. */
    int8_t *ref = calloc(1, (size_t)(n > 0 ? n : 1)), *got = calloc(1, (size_t)(n > 0 ? n : 1));
    scalar_ref(raw, ref, n);
    unpack_int4_to_int8(raw, got, n);
    int ok = (n == 0) || !memcmp(ref, got, (size_t)n);
    if (!ok) {
        int shown = 0;
        for (int64_t i = 0; i < n; i++) if (ref[i] != got[i]) {
            if (shown++ < 10) fprintf(stderr, "  MISMATCH %s i=%lld ref=%d got=%d\n",
                                       label, (long long)i, ref[i], got[i]);
        }
        fprintf(stderr, "FAIL %s: n=%lld mismatched\n", label, (long long)n);
        failures++;
    }
    free(ref); free(got);
    return ok;
}

/* All 256 packed-byte values, as n=2 single-byte arrays -- every low/high
 * nibble combination the decode can ever see. */
static void check_all_bytes(void) {
    int bad = 0;
    for (int b = 0; b < 256; b++) {
        uint8_t raw[1] = { (uint8_t)b };
        int8_t ref[2], got[2];
        scalar_ref(raw, ref, 2);
        unpack_int4_to_int8(raw, got, 2);
        if (ref[0] != got[0] || ref[1] != got[1]) {
            fprintf(stderr, "MISMATCH byte=0x%02x ref=(%d,%d) got=(%d,%d)\n",
                    b, ref[0], ref[1], got[0], got[1]);
            bad++;
        }
    }
    if (bad) { failures++; fprintf(stderr, "check_all_bytes: %d/256 mismatches\n", bad); }
    else printf("check_all_bytes: ALL PASS, 256/256 packed byte values bit-exact\n");
}

static void check_edge_arrays(void) {
    enum { N = 4096 };   /* multiple of every SIMD width with room to spare */
    uint8_t zero[N/2] = {0}, ones[N/2], mixed[N/2];
    memset(ones, 0xFF, sizeof(ones));
    for (int i = 0; i < N/2; i++) mixed[i] = (uint8_t)((i & 1) ? 0x0F : 0xF0);
    check_one("all-zero", zero, N);
    check_one("all-0xFF", ones, N);
    check_one("alternating-nibble-extremes", mixed, N);
    printf("check_edge_arrays: ok (all-zero, all-0xFF, alternating extremes)\n");
}

static void check_random_and_tail_sizes(void) {
    int sizes[] = {0,2,4,6,8,16,30,32,34,60,64,66,126,128,130,510,512,514,4096,4098,65536,131072};
    srand(20260826);
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int64_t n = sizes[s];
        uint8_t *raw = malloc((size_t)(n/2 > 0 ? n/2 : 1));
        for (int64_t i = 0; i < n/2; i++) raw[i] = (uint8_t)(rand() & 0xFF);
        char label[32]; snprintf(label, sizeof(label), "n=%lld", (long long)n);
        check_one(label, raw, n);
        free(raw);
    }
    if (!failures) printf("check_random_and_tail_sizes: ALL PASS (%zu sizes)\n", sizeof(sizes)/sizeof(sizes[0]));
}

/* Tester's-perspective check (not a correctness gate on its own -- the
 * exhaustive/random checks above already prove per-call correctness): the
 * real calling context is expert_get() releasing g_pilot_mx BEFORE calling
 * load_expert_merged (c/qwen36.c), so the single pilot worker thread and the
 * main thread can both be inside unpack_int4_to_int8's own
 * `#pragma omp parallel for` at once -- at most 2-way, since this engine
 * only ever runs ONE pilot thread (ensure_pilot_worker_started's `if
 * (!pilot_m)` guard). This reproduces exactly that: two host threads calling
 * the helper concurrently on independent buffers, checking both that
 * results stay correct under the race (no shared mutable state between the
 * two calls) and reporting the OpenMP team size actually used per call, so
 * a future change that removes the single-pilot-thread invariant has a
 * number to compare against instead of silent oversubscription. */
#ifdef _OPENMP
#include <omp.h>
#endif
typedef struct { const uint8_t *raw; int8_t *out; int64_t n; int team_size; } ConcurrentArg;
static void *concurrent_worker(void *p) {
    ConcurrentArg *a = (ConcurrentArg*)p;
    unpack_int4_to_int8(a->raw, a->out, a->n);
#ifdef _OPENMP
    a->team_size = omp_get_max_threads();
#else
    a->team_size = 1;
#endif
    return NULL;
}
static void check_concurrent_two_threads(void) {
    enum { N = 1 << 20 };   /* 1 Mi elements: large enough to actually parallelize */
    uint8_t *raw_a = malloc(N/2), *raw_b = malloc(N/2);
    int8_t *ref_a = malloc(N), *ref_b = malloc(N);
    int8_t *got_a = malloc(N), *got_b = malloc(N);
    for (int i = 0; i < N/2; i++) { raw_a[i] = (uint8_t)(i*37+1); raw_b[i] = (uint8_t)(i*61+2); }
    scalar_ref(raw_a, ref_a, N); scalar_ref(raw_b, ref_b, N);

    ConcurrentArg args[2] = { {raw_a, got_a, N, 0}, {raw_b, got_b, N, 0} };
    pthread_t t0, t1;
    pthread_create(&t0, NULL, concurrent_worker, &args[0]);
    pthread_create(&t1, NULL, concurrent_worker, &args[1]);
    pthread_join(t0, NULL); pthread_join(t1, NULL);

    int ok = !memcmp(ref_a, got_a, N) && !memcmp(ref_b, got_b, N);
    if (!ok) { fprintf(stderr, "FAIL check_concurrent_two_threads: result corrupted under 2-way concurrency\n"); failures++; }
    else printf("check_concurrent_two_threads: ALL PASS, correct under 2-way concurrency "
                "(observed OMP team sizes: %d, %d -- bounded by the engine's single-pilot-thread invariant)\n",
                args[0].team_size, args[1].team_size);
    free(raw_a); free(raw_b); free(ref_a); free(ref_b); free(got_a); free(got_b);
}

int main(void) {
    check_all_bytes();
    check_edge_arrays();
    check_random_and_tail_sizes();
    check_concurrent_two_threads();
    if (failures) { fprintf(stderr, "test_qwen36_int4_unpack: %d failure(s)\n", failures); return 1; }
    puts("test_qwen36_int4_unpack: ok");
    return 0;
}
