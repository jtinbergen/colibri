#ifndef COLIBRI_AFFINITY_H
#define COLIBRI_AFFINITY_H
/* Sizing and placing the OpenMP team on a heterogeneous CPU.
 *
 * coli_count_physical_cores() in omp_tune.h sizes the team to the number of
 * physical cores, which is right on a homogeneous part (issue #718: +2.3x on a
 * Zen3 5950X) and wrong on a hybrid one. Every expert matmul in qwen36 runs
 * under `schedule(static)`, so the rows are split evenly and the barrier at the
 * end of the region waits for the SLOWEST thread. A core that is 20% slower
 * does not contribute 80% of a share -- it makes the whole team wait.
 *
 * Measured on a Core Ultra 7 155H (6 P-cores with SMT, 8 E-cores, 2 low-power
 * E-cores), 400 decode tokens of Qwen3.6-35B-A3B int4 gs64:
 *
 *      6 threads, one per physical P-core     5.85 tok/s     82.0 ms/token
 *      4 threads, one per physical P-core     5.27 tok/s     85.3 ms/token
 *      8 threads, 6 P-cores + 2 E-cores       4.84 tok/s     98.2 ms/token
 *     12 threads, 6 P-cores with SMT          5.07 tok/s    105.4 ms/token
 *     16 threads, every physical core         4.06 tok/s    131.5 ms/token
 *
 * So both filters below carry their own weight: one thread per physical core,
 * and only cores of the fastest class. Sizing alone is not enough -- leaving
 * placement to the scheduler produced a 4.72-4.81 tok/s spread across runs,
 * because six threads land on a different mix of cores every time.
 *
 * The policy is a pure function over a topology table so it can be tested
 * without the machine it was tuned on; see tests/test_affinity.c. Only the
 * reader below is platform-specific, and it narrows the inherited affinity
 * mask rather than replacing it, so taskset and container limits still win.
 */

typedef struct {
    int  cpu;      /* logical cpu id */
    int  core;     /* topology/core_id -- shared by SMT siblings */
    int  package;  /* topology/physical_package_id -- core ids repeat per socket */
    long khz;      /* cpufreq/cpuinfo_max_freq, 0 when unreadable */
} ColiCpu;

/* A core counts as "fastest class" at or above this fraction of the highest
 * max frequency present. The gap it has to straddle, measured across parts:
 * Intel E-cores sit at 74-80% of their P-cores (155H: 3.8/4.8 = 79%), while
 * the spread WITHIN a P-core group is a favoured-core bin or two, 93% and up
 * (155H: 4.5/4.8 = 94%). 85% sits in that gap with room on both sides.
 * ARM's middle tier (A720 next to an X4) lands near the boundary; that split
 * has not been measured here, and the conservative side of the boundary is to
 * include it, which is what 85% does.
 */
#define COLI_FAST_CORE_PERCENT 85

/* Selects one logical cpu per physical core, keeping only cores of the fastest
 * class. Input is expected in ascending cpu order; output preserves it, since
 * the caller feeds it straight to a cpu mask. Returns the number written.
 *
 * When no frequency is readable at all (some VMs expose no cpufreq driver)
 * every physical core qualifies, degrading to plain physical-core sizing
 * rather than to an empty team.
 */
static int coli_select_fast_cores(const ColiCpu *cpus, int n, int *out, int max_out)
{
    long top = 0;
    for (int i = 0; i < n; i++)
        if (cpus[i].khz > top) top = cpus[i].khz;
    long floor_khz = top / 100 * COLI_FAST_CORE_PERCENT;

    int nout = 0;
    for (int i = 0; i < n && nout < max_out; i++) {
        if (cpus[i].khz < floor_khz) continue;
        int sibling_taken = 0;
        for (int j = 0; j < i; j++) {
            if (cpus[j].core == cpus[i].core &&
                cpus[j].package == cpus[i].package &&
                cpus[j].khz >= floor_khz) { sibling_taken = 1; break; }
        }
        if (!sibling_taken) out[nout++] = cpus[i].cpu;
    }
    return nout;
}

#if defined(__linux__)

#if !defined(_GNU_SOURCE)
#error "affinity.h needs _GNU_SOURCE defined before the first libc header (sched_setaffinity)"
#endif

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

static long coli_read_sysfs_long(const char *path, long fallback)
{
    FILE *f = fopen(path, "r");
    if (!f) return fallback;
    long v;
    if (fscanf(f, "%ld", &v) != 1) v = fallback;
    fclose(f);
    return v;
}

/* Narrows this process to one cpu per physical core of the fastest class.
 * Returns how many cpus the team may use, or 0 if nothing was changed --
 * on a non-hybrid part without SMT that is the same set, so 0 is not an error.
 * Must run before the first parallel region: libgomp reads the mask once.
 * Set COLI_NO_AFFINITY=1 to leave placement to the scheduler.
 */
static int coli_pin_fast_cores(void)
{
    const char *off = getenv("COLI_NO_AFFINITY");
    if (off && *off && *off != '0') return 0;

    cpu_set_t inherited;
    CPU_ZERO(&inherited);
    if (sched_getaffinity(0, sizeof inherited, &inherited) != 0) return 0;

    static ColiCpu cpus[CPU_SETSIZE];
    int n = 0;
    for (int c = 0; c < CPU_SETSIZE && n < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, &inherited)) continue;
        char path[128];
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        long core = coli_read_sysfs_long(path, -1);
        if (core < 0) continue;   /* offline, or a kernel without topology */
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        long package = coli_read_sysfs_long(path, 0);
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        long khz = coli_read_sysfs_long(path, 0);
        cpus[n].cpu = c;
        cpus[n].core = (int)core;
        cpus[n].package = (int)package;
        cpus[n].khz = khz;
        n++;
    }
    if (n == 0) return 0;

    static int chosen[CPU_SETSIZE];
    int nchosen = coli_select_fast_cores(cpus, n, chosen, CPU_SETSIZE);
    if (nchosen <= 0 || nchosen == n) return 0;

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < nchosen; i++) CPU_SET(chosen[i], &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) return 0;
    return nchosen;
}

#else  /* not Linux: no portable way to read core classes, so change nothing */

static int coli_pin_fast_cores(void) { return 0; }

#endif

#endif /* COLIBRI_AFFINITY_H */
