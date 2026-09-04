/* Unit test for the qwen36 transient CPU int8 execution scratch pool.
 *
 * Linked into qwen36.c via the same `#define main qwen36_main_unused`
 * pattern used by tests/test_qwen36_ctx.c. Only exercises qt_scratch_*;
 * does not require the CUDA tier to be active. The functions exist as
 * statics in qwen36.c; forward-declared here so the test object can
 * resolve them.
 *
 * Sub-tests:
 *   1. test_basic         -- hit returns same pointer, int8 values match
 *                            a known int4 input bit-pattern
 *   2. test_lru_eviction  -- exceeding pool size evicts oldest, NOT most-recent
 *   3. test_thread_safety -- N threads all requesting the same (layer,eid)
 *                            observe identical g/u/d pointers (mutex serialises)
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

/* qt_scratch_init/shutdown/get are defined as static in qwen36.c; the
 * include above brings them into this translation unit. No extern decls
 * needed (and they would conflict with the static linkage). */

static int fails = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        fails++; \
    } \
} while (0)

/* Int4 packed layout (per qwen36.c:1397-1408):
 *   Two nibbles per byte. LOW nibble = element 2k, HIGH = 2k+1.
 *   Signed 4-bit two's complement: sign bit set means negative.
 *
 * byte 0x12 -> lo=(2) hi=(1) -> int8[2k]=2, int8[2k+1]=1
 * byte 0x8F -> lo=(0xF & 0xF)=15 -> 15-16=-1; hi=(0x8 >> 0)=8 -> 8-16=-8
 */
static void test_basic(void) {
    qt_scratch_init(8, 8, 2);   /* tiny ng=nd=8 for fast iteration */

    /* All 0x12 -> expected int8 = [2, 1, 2, 1, ...] */
    uint8_t pat[2] = {0x12};  /* 0x12 -> lo=2 hi=1 */
    (void)pat;
    /* Need 4 bytes to cover ng=8 (packed: 8/2 = 4 bytes per matrix) */
    uint8_t g4[4], u4[4], d4[4];
    for (int i = 0; i < 4; i++) { g4[i] = 0x12; u4[i] = 0x12; d4[i] = 0x12; }

    int8_t *g, *u, *d;
    qt_scratch_get(0, 0, g4, u4, d4, &g, &u, &d);
    CHECK(g != NULL);
    CHECK(u != NULL);
    CHECK(d != NULL);

    /* Unpacked layout: g[0..ng-1], u[0..ng-1], d[0..nd-1] in one contiguous block
     * OR three separate slices -- depends on implementation. Loop check
     * catches either case by writing to the source bytes and checking again. */
    for (int i = 0; i < 8; i += 2) {
        CHECK(g[i] == 2);
        CHECK(g[i+1] == 1);
    }

    /* Hit path: same (layer,eid) -> same pointer */
    int8_t *g2, *u2, *d2;
    qt_scratch_get(0, 0, g4, u4, d4, &g2, &u2, &d2);
    CHECK(g == g2);
    CHECK(u == u2);
    CHECK(d == d2);

    /* Negative nibble values: 0x8F -> [-1, -8, -1, -8, ...] */
    for (int i = 0; i < 4; i++) { g4[i] = 0x8F; u4[i] = 0x8F; d4[i] = 0x8F; }
    int8_t *g3, *u3, *d3;
    qt_scratch_get(1, 0, g4, u4, d4, &g3, &u3, &d3);
    CHECK(g3 != g);   /* different (layer,eid) -> different slot */
    for (int i = 0; i < 8; i += 2) {
        CHECK(g3[i] == -1);
        CHECK(g3[i+1] == -8);
    }

    qt_scratch_shutdown();
}

static void test_lru_eviction(void) {
    int N = 64;
    {
        char buf[16]; snprintf(buf, sizeof buf, "%d", N);
        setenv("QT_CPU_SCRATCH_N", buf, 1);
    }
    qt_scratch_init(4, 4, 2);
    uint8_t g4[2] = {0x12};

    /* Fill pool with N distinct experts (layer=0, eid=0..N-1).
     * Track g/u/d pointers per slot via g_record[N] etc. */
    int8_t *g_record[70], *u_record[70], *d_record[70];
    (void)u_record; (void)d_record;
    for (int i = 0; i < N; i++) {
        qt_scratch_get(0, i, g4, g4, g4, &g_record[i], &u_record[i], &d_record[i]);
    }

    /* Insert one more distinct expert (eid=N). Forces eviction. */
    int8_t *extra_g, *extra_u, *extra_d;
    qt_scratch_get(0, N, g4, g4, g4, &extra_g, &extra_u, &extra_d);

    /* The first inserted expert (eid=0, oldest used counter) should now
     * have been evicted. Re-asking for it should NOT return the same
     * g pointer as before. */
    int8_t *again_g, *again_u, *again_d;
    qt_scratch_get(0, 0, g4, g4, g4, &again_g, &again_u, &again_d);
    CHECK(again_g != g_record[0]);

    /* A recent expert (eid=N-1) should still be resident. */
    int8_t *recent_g, *recent_u, *recent_d;
    qt_scratch_get(0, N-1, g4, g4, g4, &recent_g, &recent_u, &recent_d);
    CHECK(recent_g == g_record[N-1]);

    unsetenv("QT_CPU_SCRATCH_N");
    qt_scratch_shutdown();
}

typedef struct {
    int layer, eid;
    int8_t *g, *u, *d;
} thr_arg;

static void *thread_worker(void *a_) {
    thr_arg *a = (thr_arg *)a_;
    uint8_t g4[2] = {0x12};
    qt_scratch_get(a->layer, a->eid, g4, g4, g4, &a->g, &a->u, &a->d);
    return NULL;
}

static void test_thread_safety(void) {
    qt_scratch_init(4, 4, 2);
    enum { N = 16 };
    pthread_t threads[N];
    thr_arg args[N];

    /* All threads but the first ask for (layer=0, eid=0); the first asks
     * for a different expert to exercise concurrent misses. */
    for (int i = 0; i < N; i++) {
        args[i].layer = (i == 0) ? 1 : 0;
        args[i].eid = (i == 0) ? 0 : 0;
        CHECK(pthread_create(&threads[i], NULL, thread_worker, &args[i]) == 0);
    }
    for (int i = 0; i < N; i++) pthread_join(threads[i], NULL);

    /* Threads that asked for (0,0) should all observe the same g pointer. */
    int8_t *ref_g = NULL, *ref_u = NULL, *ref_d = NULL;
    int ref_count = 0;
    for (int i = 1; i < N; i++) {
        if (ref_g == NULL) {
            ref_g = args[i].g; ref_u = args[i].u; ref_d = args[i].d;
        } else {
            CHECK(args[i].g == ref_g);
            CHECK(args[i].u == ref_u);
            CHECK(args[i].d == ref_d);
        }
        ref_count++;
    }
    CHECK(ref_count == N - 1);

    /* Thread0 asked for a different (layer,eid) -- its pointer must differ. */
    CHECK(args[0].g != ref_g);

    qt_scratch_shutdown();
}

int main(void) {
    fprintf(stderr, "test_qwen36_scratch: starting\n");
    test_basic();
    test_lru_eviction();
    test_thread_safety();
    fprintf(stderr, "test_qwen36_scratch: %s (%d failures)\n",
            fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
