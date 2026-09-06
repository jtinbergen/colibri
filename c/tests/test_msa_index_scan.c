/* Step-2 M3 MSA scan contract.  This drives the production block scanner at
 * one and many OpenMP threads and compares it to a scalar reference. */
#define main coli_msa_main_unused
#include "../colibri.c"
#undef main

#include <math.h>

static int fails;
static int test_parallel=1;
static double scalar_dot(const float *a,const float *b,int n){
    double v=0; for(int i=0;i<n;i++) v+=(double)a[i]*b[i]; return v;
}
static void scalar_scores(const float *q,const float *keys,int nh,int d,int blk,
                          int st0,int pos,double *out){
    int nb=pos/blk+1;
    for(int x=0;x<nh*nb;x++) out[x]=-1e300;
    for(int t=st0;t<=pos;t++) for(int h=0;h<nh;h++){
        double v=scalar_dot(q+(int64_t)h*d,keys+(int64_t)t*d,d);
        double *dst=&out[(int64_t)h*nb+t/blk]; if(v>*dst) *dst=v;
    }
}
static void scalar_select(double *scores,int nb,int nh,int topk,int local,int pos,int blk,int *out){
    for(int h=0;h<nh;h++){
        double *s=scores+(int64_t)h*nb; int qb=pos/blk;
        for(int l=0;l<local;l++) if(qb-l>=0) s[qb-l]=1e300;
        for(int j=0;j<topk;j++) out[(int64_t)h*topk+j]=-1;
        int n=topk<nb?topk:nb;
        for(int j=0;j<n;j++){ int best=-1; double bv=-1e300;
            for(int b=0;b<nb;b++){ int taken=0;
                for(int p=0;p<j;p++) if(out[(int64_t)h*topk+p]==b) taken=1;
                if(!taken&&s[b]>bv){bv=s[b];best=b;}
            }
            out[(int64_t)h*topk+j]=best;
        }
    }
}
static int same_scores(const double *a,const double *b,int n){
    for(int i=0;i<n;i++){
        double tol=1e-11*(1.0+fabs(b[i]));
        if(fabs(a[i]-b[i])>tol){ fprintf(stderr,"score[%d] %.17g != %.17g\n",i,a[i],b[i]); return 0; }
    } return 1;
}
static void one_case(int tokens,int st0,int ties){
    enum { H=3,D=7,BLK=4,TOPK=3,LOCAL=2 };
    int pos=tokens-1, nb=pos/BLK+1, ns=H*nb;
    float *q=malloc(H*D*sizeof(*q)),*k=malloc((size_t)tokens*D*sizeof(*k));
    double *ref=malloc(ns*sizeof(*ref)),*serial=malloc(ns*sizeof(*serial)),*parallel=malloc(ns*sizeof(*parallel));
    int *sr=malloc(H*TOPK*sizeof(*sr)),*sp=malloc(H*TOPK*sizeof(*sp));
    for(int h=0;h<H;h++) for(int i=0;i<D;i++) q[h*D+i]=(float)((h+1)*(i-3))*0.125f;
    for(int t=0;t<tokens;t++) for(int i=0;i<D;i++)
        k[t*D+i]=ties?(float)(((t/BLK)%3)-1)*(float)(i+1)*0.25f:
                        (float)(((t*17+i*11)%29)-14)*0.03125f;
    scalar_scores(q,k,H,D,BLK,st0,pos,ref);
    MsaScanRun r4; msa_block_score_scan(q,k,H,D,BLK,st0,pos,parallel,test_parallel,&r4);
    MsaScanRun r1; msa_block_score_scan(q,k,H,D,BLK,st0,pos,serial,0,&r1);
    if(!same_scores(ref,serial,ns)||!same_scores(ref,parallel,ns)||memcmp(serial,parallel,ns*sizeof(*ref))){
        fprintf(stderr,"FAIL tokens=%d st0=%d ties=%d: score mismatch\n",tokens,st0,ties); fails++;
    }
    memcpy(ref,serial,ns*sizeof(*ref)); scalar_select(ref,nb,H,TOPK,LOCAL,pos,BLK,sr);
    msa_select_blocks(parallel,nb,H,TOPK,LOCAL,pos,BLK,sp);
    if(memcmp(sr,sp,H*TOPK*sizeof(*sr))){ fprintf(stderr,"FAIL tokens=%d st0=%d ties=%d: selection mismatch\n",tokens,st0,ties); fails++; }
    if(test_parallel && nb>=2 && (!r4.parallel || r4.workers<2)){ fprintf(stderr,"FAIL tokens=%d: forced route used %d workers\n",tokens,r4.workers); fails++; }
    free(q);free(k);free(ref);free(serial);free(parallel);free(sr);free(sp);
}
int main(void){
    int counts[]={1,127,128,129,2047,2048,2049,8193};
    int want=getenv("MSA_TEST_THREADS")?atoi(getenv("MSA_TEST_THREADS")):4;
    if(want<1) want=1; test_parallel=want>1;
    omp_set_dynamic(0); omp_set_num_threads(want); g_msa_scan_min_blocks=1;
    int team=0;
    #pragma omp parallel reduction(+:team)
    { team++; }
    if(test_parallel && team<2){ fprintf(stderr,"test_msa_index_scan: OpenMP team has only %d worker; cannot validate parallel route\n",team); return 77; }
    for(size_t i=0;i<sizeof(counts)/sizeof(counts[0]);i++){
        int t=counts[i]; one_case(t,0,0); one_case(t,t>8?t-7:0,1);
    }
    if(fails){ fprintf(stderr,"test_msa_index_scan: %d failure(s)\n",fails); return 1; }
    printf("test_msa_index_scan: ok (8 lengths, kv_start, ties, local blocks; forced parallel)\n"); return 0;
}
