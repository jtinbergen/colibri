/* qwen36_topology.c -- best-effort locality discovery for the Qwen tier. */
#include "qwen36_topology.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

static int env_on(const char *name)
{
    const char *v = getenv(name);
    return v && *v && atoi(v) != 0;
}

#ifdef __linux__
static int count_numa_nodes(void)
{
    int n = 0;
    for (;;) {
        char path[64];
        struct stat st;
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", n);
        if (stat(path, &st) != 0) break;
        n++;
    }
    return n > 0 ? n : 1;
}

static int read_int_file(const char *path, int fallback)
{
    FILE *f = fopen(path, "r");
    int value = fallback;
    if (f) {
        if (fscanf(f, "%d", &value) != 1) value = fallback;
        fclose(f);
    }
    return value;
}

static int pci_root_id(const char *resolved)
{
    if (!resolved) return -1;
    const char *p = resolved;
    while ((p = strstr(p, "/pci")) != NULL) {
        unsigned domain = 0, bus = 0;
        if (sscanf(p, "/pci%4x:%2x", &domain, &bus) == 2)
            return (int)(domain * 256u + bus);
        p += 4;
    }
    return -1;
}

static void discover_gpu_pci_sysfs(QtGpuTopology *g)
{
    if (!g || g->pci_address[0] == '\0') return;
    char link[PATH_MAX], resolved[PATH_MAX];
    snprintf(link, sizeof(link), "/sys/bus/pci/devices/%s", g->pci_address);
    if (realpath(link, resolved)) {
        int root = pci_root_id(resolved);
        if (root >= 0) g->pcie_root = root;
        char np[PATH_MAX];
        snprintf(np, sizeof(np), "%s/numa_node", link);
        g->numa_node = read_int_file(np, g->numa_node);
    }
}

static void discover_gpu_sysfs(QtGpuTopology *g)
{
    /* CUDA ordinal -> DRM card numbering is not guaranteed.  The stable
     * mapping is added later through the CUDA backend; for now, probe the
     * common card ordinal only and leave unknown values as -1. */
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/numa_node", g->ordinal);
    g->numa_node = read_int_file(path, -1);
    g->pcie_root = -1;
}

static void discover_storage_sysfs(QtStorageTopology *s)
{
    struct stat sb;
    if (stat(s->path, &sb) != 0) return;

    char devlink[128], resolved[PATH_MAX];
    snprintf(devlink, sizeof(devlink), "/sys/dev/block/%u:%u",
             major(sb.st_dev), minor(sb.st_dev));
    if (!realpath(devlink, resolved)) return;

    /* A mounted NVMe namespace resolves through .../nvme/nvmeN/nvmeNcM.
     * Keep the controller number as a local identity; PCI root discovery is
     * left unknown until the backend/OS mapper can distinguish root bridges. */
    for (char *p = resolved; (p = strstr(p, "/nvme")) != NULL; p += 5) {
        char *q = p + 5;
        if (*q < '0' || *q > '9') continue;
        int id = 0;
        while (*q >= '0' && *q <= '9') id = id * 10 + (*q++ - '0');
        if (*q == '/' || *q == '\0') { s->controller = id; break; }
    }
    s->pcie_root = pci_root_id(resolved);

    /* NUMA placement is exposed on the controller or one of its ancestors.
     * Walk the resolved sysfs path without changing the stored endpoint path. */
    char cur[PATH_MAX];
    snprintf(cur, sizeof(cur), "%s", resolved);
    for (;;) {
        char np[PATH_MAX];
        snprintf(np, sizeof(np), "%s/numa_node", cur);
        int node = read_int_file(np, -1);
        if (node >= 0) { s->numa_node = node; break; }
        char *slash = strrchr(cur, '/');
        if (!slash || slash == cur) break;
        *slash = '\0';
    }
}
#endif

static void add_storage(QtTopology *t, int replica, const char *path)
{
    if (!path || !*path || t->storage_count >= QT_TOPO_MAX_STORAGE) return;
    QtStorageTopology *s = &t->storage[t->storage_count++];
    memset(s, 0, sizeof(*s));
    s->replica = replica;
    s->numa_node = -1;
    s->controller = -1;
    s->pcie_root = -1;
    size_t n = strlen(path);
    if (n >= sizeof(s->path)) n = sizeof(s->path) - 1;
    memcpy(s->path, path, n);
    s->path[n] = '\0';
}

static void discover_storage(QtTopology *t)
{
    const char *primary = getenv("COLI_MODEL");
    if (!primary || !*primary) primary = getenv("SNAP");
    add_storage(t, 0, primary);

    /* Match GLM's accepted mirror spelling: semicolon or comma separated
     * directories.  These are endpoint identities only; the shard loader
     * remains the authority for which replica served an individual read. */
    const char *mirrors = getenv("COLI_MODEL_MIRROR");
    if (!mirrors || !*mirrors) mirrors = getenv("SNAP_MIRROR");
    if (mirrors && *mirrors) {
        char buf[4096];
        snprintf(buf, sizeof(buf), "%s", mirrors);
        int replica = 1;
        for (char *p = buf; *p && replica < QT_TOPO_MAX_STORAGE; ) {
            char *end = strpbrk(p, ";,");
            if (end) *end = '\0';
            add_storage(t, replica++, p);
            if (!end) break;
            p = end + 1;
        }
    }

    /* COLI_MODEL_DIRS denotes split shard roots rather than replicas.  Keep
     * them visible as storage endpoints with unique local identities. */
    const char *dirs = getenv("COLI_MODEL_DIRS");
    if (dirs && *dirs) {
        char buf[4096];
        snprintf(buf, sizeof(buf), "%s", dirs);
        for (char *p = buf; *p && t->storage_count < QT_TOPO_MAX_STORAGE; ) {
            char *end = strpbrk(p, ";,");
            if (end) *end = '\0';
            add_storage(t, t->storage_count, p);
            if (!end) break;
            p = end + 1;
        }
    }
#ifdef __linux__
    for (int i = 0; i < t->storage_count; i++) discover_storage_sysfs(&t->storage[i]);
#endif
}

void qt_topology_init(QtTopology *t, const int *devices, int count)
{
    if (!t) return;
    memset(t, 0, sizeof(*t));
#ifdef __linux__
    t->numa_nodes = count_numa_nodes();
#else
    /* Windows desktop and other non-Linux builds use the safe single-node
     * representation until a native NUMA provider is added. */
    t->numa_nodes = 1;
#endif
    t->numa_requested = env_on("COLI_NUMA") && t->numa_nodes > 1;
    /* The Qwen loader does not yet own arena allocation, so it cannot truthfully
     * claim that interleave was applied.  Keep this separate from the request;
     * the later GLM-compatible arena binding will flip numa_enabled. */
    t->numa_enabled = 0;

    if (count < 0) count = 0;
    if (count > QT_TOPO_MAX_GPU) count = QT_TOPO_MAX_GPU;
    t->gpu_count = count;
    for (int i = 0; i < count; i++) {
        QtGpuTopology *g = &t->gpu[i];
        memset(g, 0, sizeof(*g));
        g->ordinal = devices ? devices[i] : i;
        g->island = i;
        g->numa_node = 0;
        g->pci_domain = g->pci_bus = g->pci_device = g->pci_function = -1;
        g->pcie_root = -1;
        g->pci_address[0] = '\0';
#ifdef __linux__
        discover_gpu_sysfs(g);
#endif
    }
    discover_storage(t);
}

int qt_topology_home_gpu(const QtTopology *t, int eid)
{
    if (!t || t->gpu_count <= 0) return -1;
    int i = eid % t->gpu_count;
    return i < 0 ? i + t->gpu_count : i;
}

void qt_topology_set_peer(QtTopology *t, int src_index, int dst_index, int capable)
{
    if (!t || src_index < 0 || src_index >= t->gpu_count ||
        dst_index < 0 || dst_index >= t->gpu_count) return;
    if (capable) t->gpu[src_index].peer_mask |= (uint32_t)1u << dst_index;
    else t->gpu[src_index].peer_mask &= ~((uint32_t)1u << dst_index);
}

void qt_topology_set_gpu_pci(QtTopology *t, int index, int domain, int bus,
                             int device, int function)
{
    if (!t || index < 0 || index >= t->gpu_count) return;
    QtGpuTopology *g = &t->gpu[index];
    g->pci_domain = domain;
    g->pci_bus = bus;
    g->pci_device = device;
    g->pci_function = function;
    if (domain >= 0 && bus >= 0 && device >= 0 && function >= 0) {
        snprintf(g->pci_address, sizeof(g->pci_address), "%04x:%02x:%02x.%d",
                 (unsigned)domain, (unsigned)bus, (unsigned)device, function);
#ifdef __linux__
        discover_gpu_pci_sysfs(g);
#endif
    }
}

int qt_topology_pick_hub(const QtTopology *t, const int *active_indices, int count)
{
    if (!t || t->gpu_count <= 0) return -1;
    int limited = active_indices && count > 0;
    int nmax = limited ? count : t->gpu_count;
    int best = limited ? active_indices[0] : 0;
    unsigned best_score = 0;
    for (int n = 0; n < nmax; n++) {
        int i = limited ? active_indices[n] : n;
        if (i < 0 || i >= t->gpu_count) continue;
        unsigned score = 0;
        for (int m = 0; m < nmax; m++) {
            int j = limited ? active_indices[m] : m;
            if (j >= 0 && j < t->gpu_count &&
                (t->gpu[i].peer_mask & ((uint32_t)1u << j))) score++;
        }
        if (score > best_score) { best = i; best_score = score; }
    }
    return best;
}

void qt_topology_record_storage_read(QtTopology *t, int replica,
                                     uint64_t bytes, uint64_t busy_ns)
{
    if (!t) return;
    for (int i = 0; i < t->storage_count; i++) {
        QtStorageTopology *s = &t->storage[i];
        if (s->replica != replica) continue;
        s->read_ops++;
        s->read_bytes += bytes;
        s->busy_ns += busy_ns;
        return;
    }
}

void qt_topology_record_storage_read_path(QtTopology *t, const char *path,
                                          uint64_t bytes, uint64_t busy_ns)
{
    if (!t || !path || !*path) return;
    int best = -1;
    size_t best_len = 0;
    for (int i = 0; i < t->storage_count; i++) {
        const char *root = t->storage[i].path;
        size_t n = strlen(root);
        size_t plen = strlen(path);
        if (!n || n < best_len || n > plen || strncmp(path, root, n) != 0) continue;
        /* A root must match a directory boundary; /model2 must not match
         * /model20. Windows accepts either separator in configured paths. */
        char c = path[n];
        if (c && c != '/' && c != '\\') continue;
        best = i; best_len = n;
    }
    if (best >= 0) {
        t->storage[best].read_ops++;
        t->storage[best].read_bytes += bytes;
        t->storage[best].busy_ns += busy_ns;
    }
}

void qt_topology_report(const QtTopology *t, const char *prefix)
{
    if (!t) return;
    if (!prefix) prefix = "[qtier]";
    fprintf(stderr, "%s topology: NUMA nodes %d, interleave %s%s, GPUs %d, storage endpoints %d\n",
            prefix, t->numa_nodes, t->numa_enabled ? "on" : "off",
            t->numa_requested && !t->numa_enabled ? " (requested, pending arena binding)" : "",
            t->gpu_count, t->storage_count);
    for (int i = 0; i < t->gpu_count; i++) {
        const QtGpuTopology *g = &t->gpu[i];
        fprintf(stderr, "%s   island %d: GPU%d NUMA%d root%d peer-mask=0x%08x%s%s%s\n", prefix,
                g->island, g->ordinal, g->numa_node, g->pcie_root, (unsigned)g->peer_mask,
                g->integrated ? " integrated" : "",
                g->pci_address[0] ? " PCI=" : "", g->pci_address);
    }
    for (int i = 0; i < t->storage_count; i++) {
        const QtStorageTopology *s = &t->storage[i];
        fprintf(stderr, "%s   storage%d: NUMA%d controller%d root%d path=%s reads=%llu bytes=%llu\n",
                prefix, s->replica, s->numa_node, s->controller, s->pcie_root,
                s->path, (unsigned long long)s->read_ops,
                (unsigned long long)s->read_bytes);
    }
}
