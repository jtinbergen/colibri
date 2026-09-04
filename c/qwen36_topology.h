/* qwen36_topology.h -- small, portable hardware locality description.
 *
 * This is deliberately a description layer, not a scheduler.  The GLM engine
 * already owns the measured NUMA policy (interleave resident arenas and keep
 * transient staging separate); the Qwen CUDA tier uses this object to expose
 * the same locality vocabulary before it starts making policy decisions.
 */
#ifndef QWEN36_TOPOLOGY_H
#define QWEN36_TOPOLOGY_H

#include <stdint.h>

#define QT_TOPO_MAX_GPU     16
#define QT_TOPO_MAX_STORAGE 16
#define QT_TOPO_PATH        512

typedef struct {
    int ordinal;                    /* CUDA ordinal, not list position */
    int island;                     /* stable process-local island id */
    int numa_node;                  /* -1 when the OS cannot report it */
    int integrated;
    int pci_domain, pci_bus, pci_device, pci_function;
    int pcie_root;                 /* stable domain:bus root-bridge identity */
    uint32_t peer_mask;              /* destination GPU indices reachable directly */
    char pci_address[64];           /* domain:bus:device.function, if known */
} QtGpuTopology;

typedef struct {
    int replica;                    /* 0=primary, 1..=configured mirror */
    int numa_node;                  /* locality of the mounted filesystem */
    int controller;                 /* stable local id, -1 until discovered */
    int pcie_root;                  /* stable local id, -1 until discovered */
    char path[QT_TOPO_PATH];
    uint64_t read_ops;
    uint64_t read_bytes;
    uint64_t busy_ns;
} QtStorageTopology;

typedef struct {
    int numa_nodes;
    int numa_requested;             /* COLI_NUMA=1 on a multi-node host */
    int numa_enabled;               /* policy is actually applied to Qwen RAM */
    int gpu_count;
    QtGpuTopology gpu[QT_TOPO_MAX_GPU];
    int storage_count;
    QtStorageTopology storage[QT_TOPO_MAX_STORAGE];
} QtTopology;

/* Discover the process-visible GPU list plus model/mirror storage roots.
 * Discovery is best effort and never makes CUDA initialization fail. */
void qt_topology_init(QtTopology *topology, const int *devices, int count);

/* Deterministic home mapping used by the current no-replication tier. */
int qt_topology_home_gpu(const QtTopology *topology, int eid);
void qt_topology_set_peer(QtTopology *topology, int src_index, int dst_index, int capable);
void qt_topology_set_gpu_pci(QtTopology *topology, int index, int domain, int bus,
                             int device, int function);
int qt_topology_pick_hub(const QtTopology *topology, const int *active_indices, int count);

/* Record a loader event.  The source replica is intentionally supplied by the
 * loader; the CUDA tier must not guess it from an already-materialized pointer. */
void qt_topology_record_storage_read(QtTopology *topology, int replica,
                                     uint64_t bytes, uint64_t busy_ns);
void qt_topology_record_storage_read_path(QtTopology *topology, const char *path,
                                          uint64_t bytes, uint64_t busy_ns);

/* Human-readable one-shot report. */
void qt_topology_report(const QtTopology *topology, const char *prefix);

#endif
