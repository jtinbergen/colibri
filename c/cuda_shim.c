/* cuda_shim.c — shim for the single coli_cuda_* wrapper that backend_loader.c
 * does not expose. qwen36_tier.c calls coli_cuda_available_device_count()
 * before coli_cuda_init() to know the raw CUDA device count; backend_loader.c
 * only provides the post-init coli_cuda_device_count(). Resolves via cudart64_12.dll.
 *
 * The wrapper degrades gracefully: if cudart64_12.dll cannot be loaded, it
 * returns 0 (the loader does the same for everything else when the backend is
 * absent).
 */
#ifdef _WIN32

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <windows.h>
#include <stdio.h>

typedef int (*fn_cuda_get_device_count)(int *count);

static int shim_get_available_count(void) {
    static fn_cuda_get_device_count fn = NULL;
    static HMODULE mod = NULL;
if (!fn) {
        mod = LoadLibraryA("cudart64_12.dll");
        if (!mod) return 0;
        _Pragma("GCC diagnostic push")
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"")
        fn = (fn_cuda_get_device_count)GetProcAddress(mod, "cudaGetDeviceCount");
        _Pragma("GCC diagnostic pop")
        if (!fn) { FreeLibrary(mod); mod = NULL; return 0; }
    }
    int n = 0;
    if (fn(&n) != 0) return 0;
    return n;
}

int coli_cuda_available_device_count(void) {
    return shim_get_available_count();
}

#endif