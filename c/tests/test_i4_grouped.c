/* Exactness test for the grouped-int4 kernel (fmt=4, one f32 scale per `gs`
 * elements along I) against a plain-C reference that dequantizes and multiplies
 * in double.
 *
 * Why this test exists: matmul_i4_grouped is the REFERENCE the CUDA fmt=4 path
 * (#298) is expected to reproduce, and it had no test of its own. Debugging a
 * backend against an unverified oracle means two moving targets. Anyone porting
 * fmt=4 to a new backend can now diff against a kernel that is known exact here.
 *
 * Covers: I a clean multiple of gs, I with a partial last group (the `glen`
 * clamp), odd I (the nibble tail), gs larger than I (single group), and the
 * nibble edges 0 and 15 (which decode to -8 and +7 — an offset encoding, NOT
 * two's complement; getting this backwards is silent and looks like noise).
 *
 * FP note: the kernel sums each group in f32 (AVX2 accumulator + scalar tail)
 * while the reference sums in double, so we compare against a relative epsilon
 * rather than bit-exactly. The tolerance is tight enough that a wrong scale
 * index, a wrong group boundary or a swapped nibble cannot hide under it —
 * those are O(1) relative errors, not O(1e-6). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static uint32_t rng_state=0xC0FFEEu;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }
static float frand(void){ return (float)((int)(xr()%2001)-1000)/1000.0f; }

typedef struct { float *y; const float *x; const uint8_t *q; const float *s; int I,O; } RowJob;
static void *run_rows_thread(void *vp){
    RowJob *j=(RowJob*)vp;
    matmul_i4_grouped_rows(j->y,j->x,j->q,j->s,1,j->I,j->O,64,0,j->O);
    return NULL;
}
static int check_concurrent_rows(void){
    enum { I=65,O=7, RB=(I+1)/2, NG=(I+63)/64 };
    uint8_t q1[O*RB],q2[O*RB]; float s1[O*NG],s2[O*NG],x1[I],x2[I],a[O],b[O],ra[O],rb[O];
    for(int i=0;i<O*RB;i++){ q1[i]=(uint8_t)xr(); q2[i]=(uint8_t)xr(); }
    for(int i=0;i<O*NG;i++){ s1[i]=frand(); s2[i]=frand(); }
    for(int i=0;i<I;i++){ x1[i]=frand(); x2[i]=frand(); }
    matmul_i4_grouped(ra,x1,q1,s1,1,I,O,64); matmul_i4_grouped(rb,x2,q2,s2,1,I,O,64);
    typeof(g_pq) sentinel={x1,9,11,NULL,NULL,NULL}; g_pq=sentinel;
    RowJob ja={a,x1,q1,s1,I,O}, jb={b,x2,q2,s2,I,O}; pthread_t ta,tb;
    if(pthread_create(&ta,NULL,run_rows_thread,&ja)||pthread_create(&tb,NULL,run_rows_thread,&jb)) return 1;
    pthread_join(ta,NULL); pthread_join(tb,NULL);
    if(memcmp(a,ra,sizeof(a))||memcmp(b,rb,sizeof(b))||memcmp(&g_pq,&sentinel,sizeof(g_pq))){
        fprintf(stderr,"concurrent g64 row calls interfered or touched g_pq\n"); return 1;
    }
    printf("  concurrent explicit g64 row calls ok (separate input/weight/output, g_pq unchanged)\n"); return 0;
}

/* Step 5's full-expert call must be reentrant: unlike the row-only probe above,
 * this covers fused gate/up, activation, and down projection together.  Each
 * case owns every mutable buffer, while m3_i4_context proves the inputs enter
 * through the non-planar fmt=4/g64 view gate rather than a hand-built view. */
enum { M3C_I=65, M3C_H=67, M3C_D=5, M3C_IRB=(M3C_I+1)/2,
       M3C_HRB=(M3C_H+1)/2, M3C_ING=(M3C_I+63)/64,
       M3C_HNG=(M3C_H+63)/64, M3C_CAN=8 };
typedef struct {
    uint8_t gq[M3C_H*M3C_IRB], uq[M3C_H*M3C_IRB], dq[M3C_D*M3C_HRB];
    float gs[M3C_H*M3C_ING], us[M3C_H*M3C_ING], ds[M3C_D*M3C_HNG];
    float input[M3C_I];
    float gate_store[M3C_H+2*M3C_CAN], up_store[M3C_H+2*M3C_CAN];
    float output_store[M3C_D+2*M3C_CAN];
    QT gate,up,down;
    M3I4ExpertContext context;
    M3I4ExpertScratch scratch;
} M3ConcurrentCase;
typedef struct { pthread_mutex_t lock; pthread_cond_t cv; int ready,go; } M3RunStart;
typedef struct { const M3I4ExpertContext *context; M3I4ExpertScratch *scratch;
                 float *output; M3RunStart *start; int ok; } M3RunJob;

static void m3_fill_canary(float *p,int n,float value){
    for(int i=0;i<M3C_CAN;i++){ p[i]=value; p[M3C_CAN+n+i]=value; }
}
static int m3_canary_ok(const float *p,int n,float value){
    for(int i=0;i<M3C_CAN;i++) if(p[i]!=value||p[M3C_CAN+n+i]!=value) return 0;
    return 1;
}
static int m3_concurrent_case_init(M3ConcurrentCase *e,float canary){
    for(size_t i=0;i<sizeof(e->gq);i++){ e->gq[i]=(uint8_t)xr(); e->uq[i]=(uint8_t)xr(); }
    for(size_t i=0;i<sizeof(e->dq);i++) e->dq[i]=(uint8_t)xr();
    for(int i=0;i<M3C_H*M3C_ING;i++){ e->gs[i]=frand(); e->us[i]=frand(); }
    for(int i=0;i<M3C_D*M3C_HNG;i++) e->ds[i]=frand();
    for(int i=0;i<M3C_I;i++) e->input[i]=frand();
    e->gate=(QT){.fmt=4,.q4=e->gq,.s=e->gs,.O=M3C_H,.I=M3C_I,.gs=64,.planar=0};
    e->up  =(QT){.fmt=4,.q4=e->uq,.s=e->us,.O=M3C_H,.I=M3C_I,.gs=64,.planar=0};
    e->down=(QT){.fmt=4,.q4=e->dq,.s=e->ds,.O=M3C_D,.I=M3C_H,.gs=64,.planar=0};
    m3_fill_canary(e->gate_store,M3C_H,canary);
    m3_fill_canary(e->up_store,M3C_H,canary+1.f);
    m3_fill_canary(e->output_store,M3C_D,canary+2.f);
    e->scratch=(M3I4ExpertScratch){e->gate_store+M3C_CAN,e->up_store+M3C_CAN,M3C_H,M3C_H};
    return m3_i4_context(&e->context,&e->gate,&e->up,&e->down,e->input);
}
static void *run_m3_expert_thread(void *vp){
    M3RunJob *j=(M3RunJob*)vp;
    pthread_mutex_lock(&j->start->lock);
    if(++j->start->ready==2) pthread_cond_broadcast(&j->start->cv);
    while(!j->start->go) pthread_cond_wait(&j->start->cv,&j->start->lock);
    pthread_mutex_unlock(&j->start->lock);
    j->ok=m3_i4_expert_run(j->context,j->scratch,j->output);
    return NULL;
}
static int check_concurrent_full_experts(void){
    M3ConcurrentCase a,b;
    float ar_gate[M3C_H],ar_up[M3C_H],br_gate[M3C_H],br_up[M3C_H],ar[M3C_D],br[M3C_D];
    M3I4ExpertScratch asr={ar_gate,ar_up,M3C_H,M3C_H}, bsr={br_gate,br_up,M3C_H,M3C_H};
    if(!m3_concurrent_case_init(&a,101.f)||!m3_concurrent_case_init(&b,201.f)){
        fprintf(stderr,"full-expert g64 test could not build non-planar views\n"); return 1;
    }
    /* Make the input distinction obvious even if the deterministic PRNG changes. */
    a.input[0]=-.75f; b.input[0]=.875f;
    typeof(g_pq) sentinel={a.input,9,11,NULL,NULL,NULL}; g_pq=sentinel;
    if(!m3_i4_expert_run(&a.context,&asr,ar)||!m3_i4_expert_run(&b.context,&bsr,br)){
        fprintf(stderr,"full-expert sequential reference failed\n"); return 1;
    }
    M3RunStart start={PTHREAD_MUTEX_INITIALIZER,PTHREAD_COND_INITIALIZER,0,0};
    M3RunJob ja={&a.context,&a.scratch,a.output_store+M3C_CAN,&start,0};
    M3RunJob jb={&b.context,&b.scratch,b.output_store+M3C_CAN,&start,0};
    pthread_t ta,tb;
    if(pthread_create(&ta,NULL,run_m3_expert_thread,&ja)){ fprintf(stderr,"full-expert thread A creation failed\n"); return 1; }
    if(pthread_create(&tb,NULL,run_m3_expert_thread,&jb)){
        pthread_mutex_lock(&start.lock); start.go=1; pthread_cond_broadcast(&start.cv); pthread_mutex_unlock(&start.lock);
        pthread_join(ta,NULL); pthread_cond_destroy(&start.cv); pthread_mutex_destroy(&start.lock);
        fprintf(stderr,"full-expert thread B creation failed\n"); return 1;
    }
    pthread_mutex_lock(&start.lock);
    while(start.ready!=2) pthread_cond_wait(&start.cv,&start.lock);
    start.go=1; pthread_cond_broadcast(&start.cv);
    pthread_mutex_unlock(&start.lock);
    pthread_join(ta,NULL); pthread_join(tb,NULL);
    pthread_cond_destroy(&start.cv); pthread_mutex_destroy(&start.lock);
    if(!ja.ok||!jb.ok||memcmp(a.output_store+M3C_CAN,ar,sizeof(ar))||memcmp(b.output_store+M3C_CAN,br,sizeof(br))||
       !m3_canary_ok(a.gate_store,M3C_H,101.f)||!m3_canary_ok(a.up_store,M3C_H,102.f)||
       !m3_canary_ok(b.gate_store,M3C_H,201.f)||!m3_canary_ok(b.up_store,M3C_H,202.f)||
       !m3_canary_ok(a.output_store,M3C_D,103.f)||!m3_canary_ok(b.output_store,M3C_D,203.f)||
       memcmp(&g_pq,&sentinel,sizeof(g_pq))){
         fprintf(stderr,"concurrent full g64 experts interfered, corrupted a canary, or touched g_pq\n"); return 1;
    }
    /* Gate-5 P: a deliberately narrow per-expert timing, not an executor or
     * scalability benchmark.  Keep it here so the measured work is exactly the
     * explicit scratch path exercised above. */
    const int reps=200; double t0=now_s();
    for(int i=0;i<reps;i++) if(!m3_i4_expert_run(&a.context,&asr,ar)) return 1;
    printf("  explicit g64 expert kernel %.3f us/call (single-thread overhead witness only)\n",1e6*(now_s()-t0)/reps);
    puts("  concurrent full g64 experts ok (separate non-planar views, input/scratch/output, g_pq and canaries unchanged)");
    return 0;
}
static int check_activation_limits(void){
    const float L=2.f,A=1.5f;
    float g[]={-2.01f,-2.f,-1.99f,1.99f,2.f,2.01f};
    float u[]={-2.01f,-2.f,-1.99f,1.99f,2.f,2.01f}, want[6];
    for(int i=0;i<6;i++){ float v=g[i]<L?g[i]:L, z=u[i]<-L?-L:(u[i]>L?L:u[i]); want[i]=(z+1.f)*(v/(1.f+expf(-A*v))); }
    act_glu_range(g,u,0,6,1,A,L);
    if(memcmp(g,want,sizeof(g))){ fprintf(stderr,"activation clipping boundary mismatch\n"); return 1; }
    puts("  activation clipping boundaries ok"); return 0;
}

/* Reference: dequantize nibble -> (v-8)*scale[group], accumulate in double.
 * Deliberately the dumbest possible expression of the format. */
static void ref_grouped(double *y, double *mag, const float *x, const uint8_t *q4,
                        const float *scale, int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; double a=0, m=0;
            for(int i=0;i<I;i++){
                uint8_t byte=w[i>>1];
                int nib=(i&1)?(int)(byte>>4):(int)(byte&0xF);
                double term=(double)xs[i] * (double)(nib-8) * (double)scl[i/gs];
                a += term; m += fabs(term);
            }
            y[(int64_t)s*O+o]=a;
            /* Sum of |terms|: the scale the f32 rounding error actually lives on.
             * Comparing against |result| instead would flag pure cancellation --
             * a dot product of signed terms can land near zero, and then a 1e-6
             * absolute error reads as a 1e-3 relative one. That is the accumulator's
             * precision, not a kernel defect. A wrong scale index or group boundary
             * shifts the result by a fraction OF THE TERMS, so it is caught here. */
            mag[(int64_t)s*O+o]=m;
        }
    }
}

static int check(const char *name, int S, int I, int O, int gs, int fill_edges){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    uint8_t *q4=malloc((size_t)O*rb);
    float *scale=malloc((size_t)O*ng*sizeof(float));
    float *x=malloc((size_t)S*I*sizeof(float));
    float *y=malloc((size_t)S*O*sizeof(float));
    float *ytile=malloc((size_t)S*O*sizeof(float));
    double *yr=malloc((size_t)S*O*sizeof(double));
    double *ym=malloc((size_t)S*O*sizeof(double));
    if(!q4||!scale||!x||!y||!ytile||!yr||!ym){ fprintf(stderr,"%s: OOM\n",name); return 1; }

    for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(xr()&0xFF);
    if(fill_edges){
        /* nibble extremes: 0x0F -> +7, 0x00 -> -8. A two's-complement misread
         * turns 15 into -1 instead of +7 and the error is data-dependent noise. */
        for(size_t i=0;i<(size_t)O*rb && i<64;i++) q4[i]=(i&1)?0x00:0xFF;
    }
    /* scales span a few orders of magnitude: a wrong group index shows up big */
    for(int i=0;i<O*ng;i++) scale[i]=(0.001f+(float)(xr()%1000)/1000.0f)*((xr()&1)?1.f:-1.f);
    for(int i=0;i<S*I;i++) x[i]=frand();

    matmul_i4_grouped(y,x,q4,scale,S,I,O,gs);
    memset(ytile,0xA5,(size_t)S*O*sizeof(*ytile));
    matmul_i4_grouped_rows(ytile,x,q4,scale,S,I,O,gs,0,1);
    matmul_i4_grouped_rows(ytile,x,q4,scale,S,I,O,gs,1,O-1);
    matmul_i4_grouped_rows(ytile,x,q4,scale,S,I,O,gs,O-1,O);
    matmul_i4_grouped_rows(ytile,x,q4,scale,S,I,O,gs,O,O); /* valid empty final tile */
    if(memcmp(y,ytile,(size_t)S*O*sizeof(*y))){
        fprintf(stderr,"%s: tiled grouped rows differ bitwise from full wrapper\n",name);
        free(q4);free(scale);free(x);free(y);free(ytile);free(yr);free(ym); return 1;
    }
    ref_grouped(yr,ym,x,q4,scale,S,I,O,gs);

    int bad=0; double worst=0;
    for(int i=0;i<S*O;i++){
        double d=fabs((double)y[i]-yr[i]);
        double rel = ym[i]>1e-30 ? d/ym[i] : d;   /* error relative to the summed magnitude */
        if(rel>worst) worst=rel;
        if(rel>1e-6){
            if(bad<3) fprintf(stderr,"%s: [%d] got %.9g want %.9g (|terms| %.3g, rel %.3g)\n",
                              name,i,(double)y[i],yr[i],ym[i],rel);
            bad++;
        }
    }
    free(q4);free(scale);free(x);free(y);free(ytile);free(yr);free(ym);
    if(bad){ fprintf(stderr,"%s: FAIL (%d/%d mismatched, worst rel %.3g)\n",name,bad,S*O,worst); return 1; }
    printf("  %-42s ok (S=%d I=%d O=%d gs=%d ng=%d, worst rel %.2g)\n",name,S,I,O,gs,ng,worst);
    return 0;
}

/* matmul_i4_grouped_pair (fused gate+up, #298) reads x once instead of twice.
 * Checked two ways, because "identical" is only true where it can be:
 *
 *  - Correctness, always: both outputs must match the double reference within
 *    the same magnitude-relative epsilon as the unfused kernel.
 *  - Bit-exactness, only when I % gs == 0: then every group is covered by the
 *    AVX2 body, whose accumulation order is identical to the unfused kernel, so
 *    the results agree to the last bit. This is the shape the real g64
 *    checkpoints have (I = 2048 / 6144, gs = 64), i.e. the production path.
 *
 * With a PARTIAL last group the group tail falls to scalar code, and the
 * compiler is free to contract/reassociate the fused body differently from the
 * single-matrix one. The results then differ by ~1e-7 -- rounding, not logic
 * (which of gate/up "differs" is arbitrary, the tell that it is FP luck).
 * Demanding bit-exactness there would report a compiler artifact as a bug. */
static int check_pair(const char *name, int S, int I, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    uint8_t *qg=malloc((size_t)O*rb), *qu=malloc((size_t)O*rb);
    float *sg=malloc((size_t)O*ng*sizeof(float)), *su=malloc((size_t)O*ng*sizeof(float));
    float *x=malloc((size_t)S*I*sizeof(float));
    float *yg=malloc((size_t)S*O*sizeof(float)), *yu=malloc((size_t)S*O*sizeof(float));
    float *ytg=malloc((size_t)S*O*sizeof(float)), *ytu=malloc((size_t)S*O*sizeof(float));
    float *rg=malloc((size_t)S*O*sizeof(float)), *ru=malloc((size_t)S*O*sizeof(float));
    double *dg=malloc((size_t)S*O*sizeof(double)), *du=malloc((size_t)S*O*sizeof(double));
    double *mg=malloc((size_t)S*O*sizeof(double)), *mu=malloc((size_t)S*O*sizeof(double));
    if(!qg||!qu||!sg||!su||!x||!yg||!yu||!ytg||!ytu||!rg||!ru||!dg||!du||!mg||!mu){ fprintf(stderr,"%s: OOM\n",name); return 1; }

    for(size_t i=0;i<(size_t)O*rb;i++){ qg[i]=(uint8_t)(xr()&0xFF); qu[i]=(uint8_t)(xr()&0xFF); }
    for(int i=0;i<O*ng;i++){ sg[i]=frand(); su[i]=frand(); }
    for(int i=0;i<S*I;i++) x[i]=frand();

    matmul_i4_grouped_pair(yg,yu,x,qg,sg,qu,su,S,I,O,gs);
    memset(ytg,0xA5,(size_t)S*O*sizeof(*ytg)); memset(ytu,0xA5,(size_t)S*O*sizeof(*ytu));
    matmul_i4_grouped_pair_rows(ytg,ytu,x,qg,sg,qu,su,S,I,O,gs,0,1);
    matmul_i4_grouped_pair_rows(ytg,ytu,x,qg,sg,qu,su,S,I,O,gs,1,O-1);
    matmul_i4_grouped_pair_rows(ytg,ytu,x,qg,sg,qu,su,S,I,O,gs,O-1,O);
    if(memcmp(yg,ytg,(size_t)S*O*sizeof(*yg)) || memcmp(yu,ytu,(size_t)S*O*sizeof(*yu))){
        fprintf(stderr,"%s: tiled grouped pair differs bitwise from full wrapper\n",name); return 1;
    }
    matmul_i4_grouped(rg,x,qg,sg,S,I,O,gs);
    matmul_i4_grouped(ru,x,qu,su,S,I,O,gs);
    ref_grouped(dg,mg,x,qg,sg,S,I,O,gs);
    ref_grouped(du,mu,x,qu,su,S,I,O,gs);

    int bad=0, exact=1; double worst=0;
    for(int i=0;i<S*O;i++){
        double eg = mg[i]>1e-30 ? fabs((double)yg[i]-dg[i])/mg[i] : fabs((double)yg[i]-dg[i]);
        double eu = mu[i]>1e-30 ? fabs((double)yu[i]-du[i])/mu[i] : fabs((double)yu[i]-du[i]);
        if(eg>worst) worst=eg;
        if(eu>worst) worst=eu;
        if(eg>1e-6||eu>1e-6){
            if(bad<3) fprintf(stderr,"%s: [%d] gate %.9g/%.9g up %.9g/%.9g (rel %.3g/%.3g)\n",
                              name,i,(double)yg[i],dg[i],(double)yu[i],du[i],eg,eu);
            bad++;
        }
        if(yg[i]!=rg[i]||yu[i]!=ru[i]) exact=0;
    }
    /* Aligned shapes run entirely through the AVX2 body: same order as unfused,
     * so bit-exactness is a real invariant there and worth asserting. */
    if(I%gs==0 && !exact){
        fprintf(stderr,"%s: FAIL fused != unfused bitwise on an ALIGNED shape "
                       "(no scalar tail runs here; the orders must match)\n",name);
        bad++;
    }
    free(qg);free(qu);free(sg);free(su);free(x);free(yg);free(yu);free(ytg);free(ytu);free(rg);free(ru);
    free(dg);free(du);free(mg);free(mu);
    if(bad){ fprintf(stderr,"%s: FAIL (%d mismatched)\n",name,bad); return 1; }
    printf("  %-42s ok (S=%d I=%d O=%d gs=%d, worst rel %.2g%s)\n",name,S,I,O,gs,worst,
           I%gs==0?", bit-exact vs unfused":"");
    return 0;
}

static int check_parallel_task_group(int unfused){
#ifndef _OPENMP
    (void)unfused; return 77;
#else
    M3ConcurrentCase a,b;
    float ar_gate[M3C_H],ar_up[M3C_H],br_gate[M3C_H],br_up[M3C_H],ar[M3C_D],br[M3C_D];
    M3I4ExpertScratch asr={ar_gate,ar_up,M3C_H,M3C_H}, bsr={br_gate,br_up,M3C_H,M3C_H};
    if(!m3_concurrent_case_init(&a,301.f)||!m3_concurrent_case_init(&b,401.f)) return 1;
    int old_pair=g_no_fused_pair; g_no_fused_pair=unfused;
    if(!m3_i4_expert_run(&a.context,&asr,ar)||!m3_i4_expert_run(&b.context,&bsr,br)){ g_no_fused_pair=old_pair; return 1; }
    M3DagParallelExpert pa={0},pb={0};
    atomic_init(&pa.ok,0); atomic_init(&pb.ok,0);
    pa.c=a.context; pa.scratch=a.scratch; pa.output=a.output_store+M3C_CAN; pa.output_n=M3C_D; pa.eid=11; pa.route=0; pa.tiles=2; pa.layer=1; pa.generation=1; pa.use_fused_pair=!unfused;
    pb.c=b.context; pb.scratch=b.scratch; pb.output=b.output_store+M3C_CAN; pb.output_n=M3C_D; pb.eid=12; pb.route=1; pb.tiles=2; pb.layer=1; pb.generation=1; pb.use_fused_pair=!unfused;
    /* Use sections for this harness-level fan-out.  The executor under test
     * still creates its real bounded taskgroups; avoiding a second test-only
     * task-capture layer keeps TSan focused on those groups rather than on the
     * compiler/runtime's shared-data environment for epa/epb. */
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        { pa.ok=m3_dag_parallel_expert_run(&pa); }
        #pragma omp section
        { pb.ok=m3_dag_parallel_expert_run(&pb); }
    }
    g_no_fused_pair=old_pair;
    if(!pa.ok||!pb.ok||memcmp(pa.output,ar,sizeof(ar))||memcmp(pb.output,br,sizeof(br))||
       !m3_canary_ok(a.gate_store,M3C_H,301.f)||!m3_canary_ok(a.up_store,M3C_H,302.f)||
       !m3_canary_ok(b.gate_store,M3C_H,401.f)||!m3_canary_ok(b.up_store,M3C_H,402.f)||
       !m3_canary_ok(a.output_store,M3C_D,303.f)||!m3_canary_ok(b.output_store,M3C_D,403.f)){
        fprintf(stderr,"parallel task-group mismatch or canary corruption (unfused=%d)\n",unfused); return 1;
    }
    printf("  parallel expert task graph ok (%s gate/up, private scratch/output, two workers)\n",
           unfused?"separate":"fused");
    return 0;
#endif
}
static int check_parallel_resource_guard(void){
#ifdef _OPENMP
    if(m3_dag_parallel_preflight(INT_MAX,INT_MAX,INT_MAX,1,INT_MAX)!=NULL){
        fprintf(stderr,"parallel preflight accepted an overflowing resource request\n"); return 1;
    }
    puts("  parallel preflight rejects overflowing resource request");
#endif
    return 0;
}

static void warm_step6_omp_runtime(void){
#ifdef _OPENMP
    /* TSan reports libomp's lazy worker/mutex construction when the first
     * parallel region is also the test under inspection.  Initialize that
     * runtime state in a completed region so subsequent reports belong to the
     * executor's taskgroups, not the OpenMP library bootstrap. */
    #pragma omp parallel num_threads(2)
    { }
#endif
}

int main(void){
    int fail=0;
    fail|=check_activation_limits();

    /* The standalone row-wrapper probe below intentionally exercises the
     * legacy OpenMP convenience wrapper and is useful for normal correctness
     * testing, but it is not part of Step 6's bounded executor contract.  Keep
     * the TSan gate focused on the reentrant expert path, private ownership,
     * and the actual bounded task groups so a legacy-wrapper report cannot
     * obscure the executor race signal. */
    if(getenv("COLI_STEP6_TSAN_ONLY")){
        warm_step6_omp_runtime();
        fail|=check_concurrent_full_experts();
        fail|=check_parallel_task_group(0);
        fail|=check_parallel_task_group(1);
        fail|=check_parallel_resource_guard();
        printf("test_i4_grouped: Step 6 TSan subset %s\n",fail?"FAILED":"ok");
        return fail?1:0;
    }
    fail|=check_concurrent_rows();
    fail|=check_concurrent_full_experts();
    fail|=check_parallel_task_group(0);
    fail|=check_parallel_task_group(1);
    fail|=check_parallel_resource_guard();
    printf("test_i4_grouped: matmul_i4_grouped vs plain-C dequant reference\n");

    /* the shape the g64 checkpoints actually use */
    fail|=check("gs=64, I multiple of gs",            2, 512, 8, 64, 0);
    fail|=check("gs=64, single row single token",     1, 128, 1, 64, 0);
    fail|=check("gs=64, nibble edges (0x00/0xFF)",    1, 256, 4, 64, 1);

    /* partial last group: glen clamp, the classic off-by-one */
    fail|=check("gs=64, partial last group (I=200)",  2, 200, 4, 64, 0);
    fail|=check("gs=64, I just over a group (I=65)",  1,  65, 3, 64, 0);
    fail|=check("gs=64, I one under a group (I=63)",  1,  63, 3, 64, 0);

    /* odd I: the scalar nibble tail (i+1 == I) */
    fail|=check("gs=64, odd I (I=201)",               2, 201, 4, 64, 0);
    fail|=check("gs=16, odd I (I=33)",                1,  33, 2, 16, 0);

    /* gs > I: everything in one group */
    fail|=check("gs=128 > I=64 (single group)",       1,  64, 4, 128, 0);

    /* the other documented group size */
    fail|=check("gs=128, I multiple of gs",           2, 512, 4, 128, 0);

    /* batch: S>1 exercises the per-s inner loop against a shared scale row */
    fail|=check("gs=64, batch S=8",                   8, 320, 6, 64, 0);

    printf("test_i4_grouped: matmul_i4_grouped_pair (fused gate+up) vs two separate calls\n");
    fail|=check_pair("pair: gs=64, I multiple of gs",  2, 512, 8, 64);
    fail|=check_pair("pair: gs=64, partial last group",2, 200, 4, 64);
    fail|=check_pair("pair: gs=64, odd I (I=201)",     2, 201, 4, 64);
    fail|=check_pair("pair: gs=64, decode S=1",        1, 320, 6, 64);
    fail|=check_pair("pair: gs=128, I=512",            2, 512, 4, 128);

    if(fail){ printf("test_i4_grouped: FAIL\n"); return 1; }
    printf("test_i4_grouped: ok\n");
    return 0;
}
