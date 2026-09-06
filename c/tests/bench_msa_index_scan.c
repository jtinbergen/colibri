/* Step-2 microbenchmark: the production M3 block-score scanner, isolated from
 * model I/O. Run separate processes so the OpenMP team size is stable:
 *   BENCH_THREADS=1 ./tests/bench_msa_index_scan
 *   BENCH_THREADS=10 ./tests/bench_msa_index_scan */
#define main coli_msa_bench_main_unused
#include "../colibri.c"
#undef main

static uint32_t state=0xC011B1u;
static float rndf(void){ state^=state<<13; state^=state>>17; state^=state<<5;
    return (float)((int)(state&1023)-512)/512.f; }
int main(void){
    enum { H=8,D=128,BLK=128 };
    int ts[]={128,1024,2048,4096,8192,65536};
    int want=getenv("BENCH_THREADS")?atoi(getenv("BENCH_THREADS")):omp_get_max_threads();
    if(want<1) want=1; omp_set_dynamic(0); omp_set_num_threads(want); g_msa_scan_min_blocks=1;
    float *q=malloc(H*D*sizeof(*q)); for(int i=0;i<H*D;i++) q[i]=rndf();
    printf("bench_msa_index_scan: production scan | requested threads %d | runtime max %d\n",want,omp_get_max_threads());
    printf("tokens blocks reps scan-ms workers\n");
    for(size_t ci=0;ci<sizeof(ts)/sizeof(ts[0]);ci++){
        int t=ts[ci], nb=(t-1)/BLK+1, reps=t<4096?200:t<16384?50:8;
        float *k=malloc((size_t)t*D*sizeof(*k)); double *s=malloc((size_t)H*nb*sizeof(*s));
        for(size_t i=0;i<(size_t)t*D;i++) k[i]=rndf();
        MsaScanRun run; msa_block_score_scan(q,k,H,D,BLK,0,t-1,s,1,&run);
        volatile double sink=s[0]; double t0=now_s();
        for(int r=0;r<reps;r++){ msa_block_score_scan(q,k,H,D,BLK,0,t-1,s,1,&run); sink+=s[r%(H*nb)]; }
        printf("%6d %6d %4d %7.3f %7d\n",t,nb,reps,(now_s()-t0)*1e3/reps,run.workers);
        if(sink==1234567.0) fprintf(stderr,"ignore %.1f\n",sink);
        free(k);free(s);
    }
    free(q); return 0;
}
