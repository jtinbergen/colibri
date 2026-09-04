#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../qwen36_topology.h"

static int fail(const char *s) { fprintf(stderr, "topology test failed: %s\n", s); return 1; }

static void putenv_test(const char *name, const char *value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main(void)
{
    const int devices[] = { 2, 0, 7 };
    QtTopology t;
    putenv_test("COLI_NUMA", "1");
    putenv_test("COLI_MODEL", "D:/models/main");
    putenv_test("COLI_MODEL_MIRROR", "E:/models/mirror;F:/models/mirror2");
    putenv_test("COLI_MODEL_DIRS", "G:/shard0,G:/shard1");
    qt_topology_init(&t, devices, 3);
    if (t.gpu_count != 3) return fail("GPU count");
    if (t.numa_enabled) return fail("NUMA policy falsely reported as applied");
    if (t.gpu[0].ordinal != 2 || t.gpu[1].ordinal != 0) return fail("CUDA ordinal preserved");
    if (qt_topology_home_gpu(&t, 0) != 0 || qt_topology_home_gpu(&t, 4) != 1)
        return fail("deterministic home mapping");
    qt_topology_set_peer(&t, 0, 0, 1);
    qt_topology_set_peer(&t, 1, 1, 1);
    qt_topology_set_peer(&t, 2, 2, 1);
    qt_topology_set_peer(&t, 1, 0, 1);
    qt_topology_set_peer(&t, 1, 2, 1);
    qt_topology_set_peer(&t, 0, 1, 1);
    qt_topology_set_peer(&t, 2, 1, 1);
    const int active[] = {0, 1, 2};
    if (qt_topology_pick_hub(&t, active, 3) != 1)
        return fail("peer-connected hub selection");
    qt_topology_set_gpu_pci(&t, 1, 0, 65, 0, 0);
    if (strcmp(t.gpu[1].pci_address, "0000:41:00.0") != 0 ||
        t.gpu[1].pci_bus != 65)
        return fail("CUDA PCI identity");
    qt_topology_set_peer(&t, 1, 0, 0);
    if ((t.gpu[1].peer_mask & 1u) != 0)
        return fail("peer edge clear");
    if (t.storage_count != 5) return fail("model/mirror/split storage endpoints");
    if (t.storage[0].replica != 0 || t.storage[1].replica != 1 || t.storage[2].replica != 2)
        return fail("replica identity");
    qt_topology_record_storage_read(&t, 1, 4096, 100);
    if (t.storage[1].read_ops != 1 || t.storage[1].read_bytes != 4096 || t.storage[1].busy_ns != 100)
        return fail("storage counters");
    qt_topology_record_storage_read_path(&t, "E:/models/mirror/model-00001.safetensors", 8192, 200);
    if (t.storage[1].read_ops != 2 || t.storage[1].read_bytes != 12288 || t.storage[1].busy_ns != 300)
        return fail("path storage attribution");
    puts("qwen36 topology tests: ok");
    return 0;
}
