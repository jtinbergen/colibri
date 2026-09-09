#include "../expert_store_registry.h"
#include "../m3_shadow_store.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int lookups; int destroys; } fake_state_t;

static int fake_lookup(ColiExpertStore *store, ColiExpertKey key,
                       ColiExpertView *view) {
    fake_state_t *state = (fake_state_t *)store->state;
    state->lookups++;
    memset(view, 0, sizeof(*view));
    view->key = key;
    view->lease = state;
    return 0;
}

static void fake_release(ColiExpertStore *store, ColiExpertView *view) {
    (void)store;
    memset(view, 0, sizeof(*view));
}

static int fake_prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                         size_t count) {
    (void)store; (void)keys; return (int)count;
}

static void fake_stats(const ColiExpertStore *store,
                       ColiExpertStoreStats *stats) {
    const fake_state_t *state = (const fake_state_t *)store->state;
    memset(stats, 0, sizeof(*stats));
    stats->requests = (uint64_t)state->lookups;
}

static void fake_destroy(ColiExpertStore *store) {
    fake_state_t *state = (fake_state_t *)store->state;
    state->destroys++;
    free(state);
    free(store);
}

static const ColiExpertStoreOps fake_ops = {
    fake_lookup, fake_release, fake_prefetch, fake_stats, fake_destroy
};

static int fake_open(ColiV4Engine *engine,
                     const ColiDeepSeekV4Config *config,
                     const ColiDeepSeekV4ExpertStoreOptions *options,
                     ColiExpertStore **output, char *error, size_t error_size) {
    (void)engine; (void)config; (void)options; (void)error; (void)error_size;
    ColiExpertStore *store = (ColiExpertStore *)calloc(1, sizeof(*store));
    fake_state_t *state = (fake_state_t *)calloc(1, sizeof(*state));
    assert(store && state);
    store->ops = &fake_ops;
    store->state = state;
    *output = store;
    return 0;
}

int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine, const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    return fake_open(engine, config, options, output, error, error_size);
}

int main(void) {
    assert(coli_expert_store_backend_register("fake", fake_open) == 0);
#ifdef _WIN32
    assert(_putenv("COLI_EXPERT_STORE=shadow-fake") == 0);
#else
    assert(putenv("COLI_EXPERT_STORE=shadow-fake") == 0);
#endif
    ColiExpertStore *store = NULL;
    char error[128] = {0};
    assert(coli_expert_store_backend_open_selected(NULL, NULL, NULL, &store,
                                                   error, sizeof(error)) == 0);
    assert(store && m3_shd_store_is_wrapper(store));
    ColiExpertView view;
    assert(coli_expert_lookup(store, (ColiExpertKey){1, 2}, &view) == 0);
    coli_expert_release(store, &view);
    store->ops->destroy(store);
    puts("m3 shadow registry boundary: PASS");
    return 0;
}
