/* qwen36.c's load_tq (dense weight int8-during-load, streamed row-chunks via
 * st_read_slice_f32) must be memcmp-exact against an independent per-row
 * abs-max/127 symmetric quantizer -- the same math the old post-model_init
 * qdw_register pass used before this change (per-row scale is independent
 * of every other row, so streaming in chunks cannot change the result; see
 * quantize_row_sym8's own comment in qwen36.c). This is a from-scratch
 * reference (not load_tq's own quantize_row_sym8 helper) so the test is not
 * circular.
 *
 * Also checks:
 *  - the COLI_KEEP_F32 branch (materialize f32 then quantize) against the
 *    same reference;
 *  - dense_quant=0 (the Segment/embedding API's path -- see
 *    model_init_range's dense_quant comment in qwen36.c): must return exact
 *    f32, untouched, regardless of COLI_DENSE_I8;
 *  - COLI_DENSE_I8=0 (dense_quant=1 but globally disabled): must also fall
 *    back to exact f32.
 *
 * dense_i8_on() caches COLI_DENSE_I8 in a function-local static on first
 * call, so the COLI_DENSE_I8=0 case runs in a forked child (before any
 * dense_i8_on() call in that lineage) to get its own independent cache
 * without disturbing the parent's.
 *
 * Shapes span the LOAD_TQ_CHUNK_ROWS=128 streaming-chunk boundary (O one
 * below/at/above a multiple of 128) and range from a single row to several
 * chunks, non-square, per the "shape-diverse dense weight" review point --
 * not one tensor standing in for all the dense matrices this engine loads.
 *
 * Build: make tests/test_qwen36_load_tq
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <stdio.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

static int failures;

/* Independent reference: qdw_register's original per-row math, written
 * fresh here rather than calling quantize_row_sym8 (which load_tq itself
 * uses) -- so a bug shared by load_tq and its own helper would not be
 * invisible to this test. */
static void ref_quantize(const float *W, int I, int O, int8_t *q, float *sc) {
    for (int o = 0; o < O; o++) {
        const float *r = W + (int64_t)o*I; float am = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        float s = am > 1e-12f ? am/127.f : 1.f; sc[o] = s; float inv = 1.f/s;
        int8_t *d = q + (int64_t)o*I;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(r[i]*inv); if (v>127) v=127; if (v<-127) v=-127; d[i] = (int8_t)v; }
    }
}

static float fill_value(int64_t idx, int salt) {
    /* deterministic pseudo-random, including some exact zeros (all-zero-row
     * scale branch) and some large magnitudes (saturation clamp branch) */
    int v = (int)((idx * 2654435761u + (unsigned)salt * 97) % 4001);
    if (v % 37 == 0) return 0.f;                    /* occasional exact zero */
    float f = ((float)v - 2000.f) / 7.3f;
    if (v % 101 == 0) f *= 5000.f;                   /* occasional saturating magnitude */
    return f;
}

/* Writes one F32 tensor named `name` of I*O elements into a safetensors
 * file at dir/model.safetensors, alongside any tensors already appended
 * (single-shard, multi-tensor header). Returns via *bytes_out the raw f32
 * content so the caller can build its own reference without re-reading. */
typedef struct { char name[64]; int I, O; } TensorSpec;

static void write_snapshot(const char *dir, const TensorSpec *specs, int n, float **contents) {
    char path[512]; snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    char hdr[4096]; int hp = 0;
    hdr[hp++] = '{';
    int64_t offset = 0;
    for (int t = 0; t < n; t++) {
        int64_t n_elems = (int64_t)specs[t].I * specs[t].O;
        hp += snprintf(hdr+hp, sizeof(hdr)-hp,
            "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]}",
            t ? "," : "", specs[t].name, specs[t].O, specs[t].I,
            (long long)offset, (long long)(offset + n_elems*4));
        offset += n_elems*4;
    }
    hdr[hp++] = '}';
    uint64_t hlen = (uint64_t)hp;
    FILE *f = fopen(path, "wb");
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, (size_t)hp, f);
    for (int t = 0; t < n; t++) fwrite(contents[t], 4, (size_t)specs[t].I*specs[t].O, f);
    fclose(f);
}

static int check_shape(Model *m, const char *name, int I, int O, const float *content) {
    int8_t *ref_q = malloc((size_t)O*I); float *ref_sc = malloc((size_t)O*sizeof(float));
    ref_quantize(content, I, O, ref_q, ref_sc);

    QW streamed = load_tq(m, name, I, O);
    int ok = 1;
    if (!streamed.q || streamed.f32) { fprintf(stderr, "FAIL %s streamed: expected q!=NULL, f32==NULL\n", name); ok = 0; }
    else if (memcmp(streamed.q, ref_q, (size_t)O*I) || memcmp(streamed.sc, ref_sc, (size_t)O*sizeof(float))) {
        fprintf(stderr, "FAIL %s streamed: quantized bytes differ from reference (I=%d O=%d)\n", name, I, O);
        ok = 0;
    }
    free((void*)streamed.q); free(streamed.sc);

    setenv("COLI_KEEP_F32", "1", 1);
    QW kept = load_tq(m, name, I, O);
    unsetenv("COLI_KEEP_F32");
    if (!kept.q || !kept.f32) { fprintf(stderr, "FAIL %s KEEP_F32: expected both q and f32 set\n", name); ok = 0; }
    else if (memcmp(kept.f32, content, (size_t)O*I*sizeof(float))) {
        fprintf(stderr, "FAIL %s KEEP_F32: f32 copy differs from source\n", name); ok = 0;
    } else if (memcmp(kept.q, ref_q, (size_t)O*I) || memcmp(kept.sc, ref_sc, (size_t)O*sizeof(float))) {
        fprintf(stderr, "FAIL %s KEEP_F32: quantized bytes differ from reference\n", name); ok = 0;
    }
    free((void*)kept.f32); free(kept.q); free(kept.sc);

    free(ref_q); free(ref_sc);
    if (ok) printf("check_shape %s I=%d O=%d: streamed + KEEP_F32 both exact\n", name, I, O);
    else failures++;
    return ok;
}

static void run_quantized_cases(const char *dir, const TensorSpec *specs, int n, float **contents) {
    Model m; memset(&m, 0, sizeof(m));
    st_init(&m.S, dir);
    m.dense_quant = 1;
    for (int t = 0; t < n; t++) check_shape(&m, specs[t].name, specs[t].I, specs[t].O, contents[t]);

    /* dense_quant=0 (Segment/embedding API path): always exact f32,
     * regardless of COLI_DENSE_I8/COLI_KEEP_F32 (dense_i8_on() must not even
     * be consulted -- see load_tq's short-circuit). */
    Model m0; memset(&m0, 0, sizeof(m0));
    m0.S = m.S; m0.dense_quant = 0;
    setenv("COLI_KEEP_F32", "1", 1);
    QW seg = load_tq(&m0, specs[0].name, specs[0].I, specs[0].O);
    unsetenv("COLI_KEEP_F32");
    if (seg.q || seg.sc || !seg.f32 ||
        memcmp(seg.f32, contents[0], (size_t)specs[0].I*specs[0].O*sizeof(float))) {
        fprintf(stderr, "FAIL dense_quant=0: expected exact untouched f32\n");
        failures++;
    } else {
        printf("dense_quant=0 (Segment API path): exact f32, q/sc left NULL\n");
    }
    free((void*)seg.f32);
}

/* COLI_DENSE_I8=0: dense_i8_on() caches per-process, so this runs in a
 * fresh forked child (before any dense_i8_on() call happens in it). */
static int run_disabled_case(const char *dir, const TensorSpec *specs, float **contents) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 0; }
    if (pid == 0) {
        setenv("COLI_DENSE_I8", "0", 1);
        Model m; memset(&m, 0, sizeof(m));
        st_init(&m.S, dir);
        m.dense_quant = 1;
        QW w = load_tq(&m, specs[0].name, specs[0].I, specs[0].O);
        int ok = w.f32 && !w.q && !w.sc &&
                 !memcmp(w.f32, contents[0], (size_t)specs[0].I*specs[0].O*sizeof(float));
        if (!ok) fprintf(stderr, "FAIL COLI_DENSE_I8=0: expected exact untouched f32\n");
        else printf("COLI_DENSE_I8=0: exact f32 fallback even with dense_quant=1\n");
        /* exit(), not _exit(): stdout is still empty at this point (fork()
         * happens before main() prints anything), so there is no risk of the
         * parent later re-flushing a duplicated buffer -- and unlike _exit(),
         * exit() actually flushes this child's own stdio so the line above
         * is not silently lost. */
        exit(ok ? 0 : 1);
    }
    int status = 0; waitpid(pid, &status, 0);
    int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (!ok) failures++;
    return ok;
}

int main(void) {
    unsetenv("COLI_DENSE_I8"); unsetenv("COLI_KEEP_F32");
    char dir[] = "test_qwen36_load_tq_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }

    /* Shapes: single row, tiny square, non-square, and around the
     * LOAD_TQ_CHUNK_ROWS=128 streaming boundary (127/128/129/130), plus a
     * multi-chunk case -- spans the variety this engine actually loads
     * (embed/lm_head-like wide, attention-proj-like narrow, several chunks). */
    TensorSpec specs[] = {
        { "t_1x1",     1,   1 },
        { "t_5x17",    5,  17 },
        { "t_17x5",   17,   5 },
        { "t_64x127", 64, 127 },
        { "t_64x128", 64, 128 },
        { "t_64x129", 64, 129 },
        { "t_64x130", 64, 130 },
        { "t_300x500",300,500 },
    };
    int n = (int)(sizeof(specs)/sizeof(specs[0]));
    float *contents[8];
    for (int t = 0; t < n; t++) {
        int64_t cnt = (int64_t)specs[t].I * specs[t].O;
        contents[t] = malloc((size_t)cnt * sizeof(float));
        for (int64_t i = 0; i < cnt; i++) contents[t][i] = fill_value(i, t);
    }
    write_snapshot(dir, specs, n, contents);

    /* Fork BEFORE any OpenMP parallel region has run in this process (every
     * load_tq/quantize_dense_qw call below uses one): forking a process that
     * has already spun up an OpenMP thread pool can deadlock the child, since
     * only the calling thread survives fork() while the runtime's internal
     * locks/state assume the whole pool is still there. */
    run_disabled_case(dir, specs, contents);
    run_quantized_cases(dir, specs, n, contents);

    char snap_path[600]; snprintf(snap_path, sizeof(snap_path), "%s/model.safetensors", dir);
    remove(snap_path); rmdir(dir);

    if (failures) { fprintf(stderr, "test_qwen36_load_tq: %d failure(s)\n", failures); return 1; }
    puts("test_qwen36_load_tq: ok");
    return 0;
}
