/* COLI_PIPE_BLOCK: the pipe pool's condvar waiter must be observably
 * equivalent to the sched_yield spin it replaces — same bytes land in the
 * same ws[] slots, and no interleaving loses a wakeup (the worker RELEASE-
 * stores ready[] BEFORE taking mx to broadcast; the waiter re-checks under
 * the lock, so a flag set between its fast-path check and the wait cannot
 * be missed). Both waiters are exercised against the same on-disk fixture,
 * alternating parked waits (wait issued before the load finishes) with
 * fast-path waits (load already done), across enough generations to cycle
 * the pool's gen-tagged cursor.
 *
 * Also pins the PIPE_WORKERS => PIPE implication table: fires ONLY when
 * PIPE is unset in the env AND the platform default left the pipe off AND
 * PIPE_WORKERS parses positive (PIPE_WORKERS=0/empty/negative must NOT
 * silently enable a clamped 1-worker pipe). */
#include <errno.h>
#include <fcntl.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define COLI_PIPE_TEST
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int fail(const char *s){ fprintf(stderr,"FAIL: %s\n",s); return 1; }

enum { NE=8, LAYER=1 };            /* experts 0..NE-1 on one MoE layer */
/* per-expert file image: [gate 12][up 12][down 12][gate.qs 12][up.qs 12][down.qs 16] */
enum { WB=12, QS_G=12, QS_U=12, QS_D=16, ESZ=3*WB+QS_G+QS_U+QS_D };

static unsigned char wbyte(int e,int j){ return (unsigned char)(e*31+j+1); }
static float scale(int e,int i){ return (float)(e*8+i)+0.5f; }

#define TMPF "test_pipe_block.tmp"

static int write_fixture(void){
    FILE *w=fopen(TMPF,"wb"); if(!w) return fail("create temp");
    for(int e=0;e<NE;e++){
        unsigned char img[ESZ];
        for(int j=0;j<3*WB;j++) img[j]=wbyte(e,j);
        float sc[(QS_G+QS_U+QS_D)/4];
        for(int i=0;i<(int)(sizeof(sc)/sizeof(sc[0]));i++) sc[i]=scale(e,i);
        memcpy(img+3*WB,sc,sizeof(sc));
        if(fwrite(img,1,ESZ,w)!=ESZ){ fclose(w); return fail("expert fixture write"); }
    }
    fclose(w);
    return 0;
}

static int build_fixture(Model *m,int fd){
    m->c.hidden=4; m->c.moe_inter=3; m->ebits=8;
    m->S.n=NE*6; m->S.cap=NE*6; m->S.t=calloc(NE*6,sizeof(st_tensor));
    if(!m->S.t) return fail("tensor metadata allocation");
    const char *proj[3]={"gate_proj","up_proj","down_proj"};
    int sbytes[3]={QS_G,QS_U,QS_D};
    for(int e=0;e<NE;e++){
        int64_t wo=(int64_t)e*ESZ, so=wo+3*WB;
        for(int k=0;k<3;k++){
            char name[300];
            snprintf(name,sizeof(name),"model.layers.%d.mlp.experts.%d.%s.weight",LAYER,e,proj[k]);
            m->S.t[e*6+k]=(st_tensor){.name=strdup(name),.fd=fd,.off=wo,
                .nbytes=WB,.dtype=3 /* U8 */,.numel=WB}; wo+=WB;
            size_t n=strlen(name); memcpy(name+n,".qs",4);
            m->S.t[e*6+3+k]=(st_tensor){.name=strdup(name),.fd=fd,.off=so,
                .nbytes=sbytes[k],.dtype=2 /* F32 */,.numel=sbytes[k]/4}; so+=sbytes[k];
        }
    }
    return 0;
}

static int check_slot(ESlot *s,int e){
    if(s->eid!=e || s->g.fmt!=1 || s->u.fmt!=1 || s->d.fmt!=1){
        fprintf(stderr,"  slot: eid=%d (want %d) fmt g/u/d=%d/%d/%d (want 1/1/1)\n",
                s->eid,e,s->g.fmt,s->u.fmt,s->d.fmt);
        return 1;
    }
    const unsigned char *g=(const unsigned char*)s->g.q8,
                        *u=(const unsigned char*)s->u.q8,
                        *d=(const unsigned char*)s->d.q8;   /* q8 is int8_t; compare raw bytes */
    for(int j=0;j<WB;j++)
        if(g[j]!=wbyte(e,j) || u[j]!=wbyte(e,WB+j) || d[j]!=wbyte(e,2*WB+j)){
            fprintf(stderr,"  slot e=%d weight byte %d: g=%d/%d u=%d/%d d=%d/%d (got/want)\n",e,j,
                    g[j],wbyte(e,j),u[j],wbyte(e,WB+j),d[j],wbyte(e,2*WB+j));
            return 1;
        }
    for(int i=0;i<3;i++)
        if(s->g.s[i]!=scale(e,i) || s->u.s[i]!=scale(e,3+i)){
            fprintf(stderr,"  slot e=%d scale %d: g=%g/%g u=%g/%g (got/want)\n",e,i,
                    (double)s->g.s[i],(double)scale(e,i),(double)s->u.s[i],(double)scale(e,3+i));
            return 1;
        }
    for(int i=0;i<4;i++)
        if(s->d.s[i]!=scale(e,6+i)){
            fprintf(stderr,"  slot e=%d scale %d: d=%g/%g (got/want)\n",e,i,
                    (double)s->d.s[i],(double)scale(e,6+i));
            return 1;
        }
    return 0;
}

static int run_generations(Model *m,int block,int gens){
    g_pipe_block=block;
    for(int gen=0;gen<gens;gen++){
        int eids[NE];
        for(int q=0;q<NE;q++) eids[q]=(gen*3+q)%NE;   /* deterministic shuffle across gens */
        pipe_dispatch(m,LAYER,eids,NE);
        if(gen%4==0) usleep(300);                     /* let loads finish → fast-path wait */
        for(int i=0;i<NE;i++){
            /* odd gens wait on the LAST-dispatched slot first: with jobs this
             * small, in-order waits mostly find ready already set — reverse
             * order is what actually parks the waiter on the condvar. */
            int q=(gen&1)?NE-1-i:i;
            pipe_wait(q);
            if(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire))
                return fail(block?"blocking wait returned before ready":"spin wait returned before ready");
            if(check_slot(&m->ws[q],eids[q])) return fail(block?"slot contents (block)":"slot contents (spin)");
        }
    }
    return 0;
}

static int test_implication_table(void){
    struct { const char *pipe_env,*pw_env; int pipe_now,want; } T[]={
        {NULL,"4",0,1},   /* pool sized, pipe off, PIPE unset → imply */
        {NULL,"16",0,1},
        {NULL,"0",0,0},   /* PIPE_WORKERS=0 must NOT enable a clamped pipe */
        {NULL,"",0,0},
        {NULL,"-3",0,0},
        {"0","4",0,0},    /* explicit PIPE=0 always wins */
        {"1","4",1,0},    /* explicit PIPE=1: nothing to imply */
        {NULL,"4",1,0},   /* platform default already ON (win32) */
        {NULL,NULL,0,0},
    };
    for(size_t i=0;i<sizeof(T)/sizeof(T[0]);i++)
        if(pipe_workers_imply_pipe(T[i].pipe_env,T[i].pw_env,T[i].pipe_now)!=T[i].want){
            fprintf(stderr,"FAIL: implication row %zu (PIPE=%s PIPE_WORKERS=%s pipe_now=%d)\n",
                    i,T[i].pipe_env?T[i].pipe_env:"<unset>",T[i].pw_env?T[i].pw_env:"<unset>",T[i].pipe_now);
            return 1;
        }
    return 0;
}

typedef struct { _Atomic int waiter_entered; int q; } AnyReadyLatch;
static void *publish_latched_b(void *vp){
    AnyReadyLatch *l=(AnyReadyLatch*)vp;
    while(!atomic_load_explicit(&l->waiter_entered,memory_order_acquire)) sched_yield();
    atomic_store_explicit(&g_pp.ready[l->q],1,memory_order_release);
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv_done); pthread_mutex_unlock(&g_pp.mx);
    return NULL;
}

/* A is deliberately never released.  B is published only after wait-any has
 * begun, so success proves the predicate wait does not park behind an earlier
 * pending slot.  Exercise both the spin and condvar forms. */
static int test_wait_any_ready_first(void){
    int qof[2]={0,1};
    uint64_t gen=atomic_load_explicit(&g_pp.cur,memory_order_acquire)>>8;
    for(int block=0;block<2;block++){
        g_pipe_block=block;
        M3DagExpertTask tasks[2]={0};
        tasks[0].compute_state=M3_DAG_COMPUTE_WAITING;
        tasks[1].compute_state=M3_DAG_COMPUTE_READY;
        int ready_waited=1;
        if(m3_dag_pipe_pick_ready(tasks,2,qof,gen,&ready_waited)!=1 || ready_waited)
            return fail(block?"ready task did not beat pending I/O (block)":"ready task did not beat pending I/O (spin)");
        tasks[0].compute_state=M3_DAG_COMPUTE_COMPLETE; /* consumed slot A must not wake selection */
        tasks[1].compute_state=M3_DAG_COMPUTE_WAITING;
        atomic_store_explicit(&g_pp.ready[0],1,memory_order_release);
        atomic_store_explicit(&g_pp.ready[1],1,memory_order_release);
        ready_waited=1;
        if(m3_dag_pipe_pick_ready(tasks,2,qof,gen,&ready_waited)!=1 || ready_waited)
            return fail(block?"completed ready slot was not excluded (block)":"completed ready slot was not excluded (spin)");
        tasks[0].compute_state=M3_DAG_COMPUTE_WAITING;
        atomic_store_explicit(&g_pp.ready[0],0,memory_order_relaxed); /* held A */
        atomic_store_explicit(&g_pp.ready[1],0,memory_order_relaxed);
        AnyReadyLatch l={0,1}; pthread_t th;
        g_pipe_test_wait_any_enter=&l.waiter_entered;
        if(pthread_create(&th,NULL,publish_latched_b,&l)) return fail("wait-any latch thread");
        int event=0,shared_start=++event,shared_end=++event,waited=0;
        int got=m3_dag_pipe_pick_ready(tasks,2,qof,gen,&waited);
        int b_start=++event,b_end=++event;            /* production selects B here */
        pthread_join(th,NULL);
        g_pipe_test_wait_any_enter=NULL;
        int a_was_held=!atomic_load_explicit(&g_pp.ready[0],memory_order_acquire);
        int a_release=++event;
        atomic_store_explicit(&g_pp.ready[0],1,memory_order_release);
        if(got!=1 || !waited || !a_was_held || !(shared_start<shared_end && shared_end<b_start && b_start<b_end && b_end<a_release))
            return fail(block?"ready-first event order (block)":"ready-first event order (spin)");
    }
    return 0;
}

typedef struct { _Atomic int a_entered,a_release,drain_entered,wait_a_entered,drain_after_a,drain_done; } DrainLatch;
static DrainLatch *g_drain_latch;
static void hold_load_a(int q){
    if(q!=0) return;
    atomic_store_explicit(&g_drain_latch->a_entered,1,memory_order_release);
    while(!atomic_load_explicit(&g_drain_latch->a_release,memory_order_acquire)) sched_yield();
}
static void *drain_published_slots(void *vp){
    DrainLatch *l=(DrainLatch*)vp;
    m3_dag_pipe_drain_slots(2);
    atomic_store_explicit(&l->drain_done,1,memory_order_release);
    return NULL;
}

/* This is the exact drain helper used by the Step-4 fatal scheduler branch.
 * A worker owns slot A while B publishes.  Once a hypothetical B-run failure
 * enters the drain boundary, it cannot complete (and therefore cannot free or
 * reuse slots) until A is explicitly released. */
static int test_failure_drain_live_load(Model *m){
    for(int block=0;block<2;block++){
        g_pipe_block=block;
        DrainLatch l={0}; g_drain_latch=&l;
        g_pipe_test_before_load=hold_load_a;
        int eids[2]={2,3}; pipe_dispatch(m,LAYER,eids,2);
        int rc=0, drain_started=0; pthread_t th;
        while(!atomic_load_explicit(&l.a_entered,memory_order_acquire)) sched_yield();
        pipe_wait(1);                              /* B is ready while A remains live */
        M3DagExpertTask tasks[2]={0}; int qof[2]={0,1}, waited=0;
        tasks[0].compute_state=tasks[1].compute_state=M3_DAG_COMPUTE_WAITING;
        if(m3_dag_pipe_pick_ready(tasks,2,qof,atomic_load_explicit(&g_pp.cur,memory_order_acquire)>>8,&waited)!=1)
            rc=fail(block?"failure drain did not select ready B (block)":"failure drain did not select ready B (spin)");
        if(rc) goto cleanup;
        g_pipe_test_drain_enter=&l.drain_entered;
        g_pipe_test_wait_slot_enter=&l.wait_a_entered;
        g_pipe_test_drain_after_slot=&l.drain_after_a;
        if(pthread_create(&th,NULL,drain_published_slots,&l)){ rc=fail("failure drain thread"); goto cleanup; }
        drain_started=1;
        while(!atomic_load_explicit(&l.drain_entered,memory_order_acquire)) sched_yield();
        while(!atomic_load_explicit(&l.wait_a_entered,memory_order_acquire)) sched_yield();
        if(atomic_load_explicit(&l.drain_after_a,memory_order_acquire)){ rc=fail("failure drain crossed held A"); goto cleanup; }
        atomic_store_explicit(&l.a_release,1,memory_order_release);
        pthread_join(th,NULL);
        drain_started=0;
        if(!atomic_load_explicit(&l.drain_done,memory_order_acquire) ||
           !atomic_load_explicit(&l.drain_after_a,memory_order_acquire) ||
           !atomic_load_explicit(&g_pp.ready[0],memory_order_acquire) ||
           !atomic_load_explicit(&g_pp.ready[1],memory_order_acquire) ||
           check_slot(&m->ws[0],eids[0]) || check_slot(&m->ws[1],eids[1]))
            rc=fail(block?"failure drain did not acknowledge both slots (block)":"failure drain did not acknowledge both slots (spin)");
cleanup:
        atomic_store_explicit(&l.a_release,1,memory_order_release);
        if(drain_started) pthread_join(th,NULL);
        else { pipe_wait(0); pipe_wait(1); }
        g_pipe_test_drain_after_slot=NULL; g_pipe_test_wait_slot_enter=NULL;
        g_pipe_test_drain_enter=NULL; g_pipe_test_before_load=NULL; g_drain_latch=NULL;
        if(rc) return rc;
    }
    return 0;
}

/* Gate C/M integration: drive the real ready-first moe() scheduler through a
 * task-run failure.  A remains in its worker while B is selected and fails;
 * the test-only fatal callback is reached only after the production branch has
 * drained A and B, so setjmp avoids turning this unit test into a subprocess. */
static jmp_buf g_scheduler_fatal;
typedef struct { DrainLatch *l; _Atomic int crossed_while_held; } SchedulerCtl;
static void force_grouped_slot(int q){
    ESlot *s=&g_pp.m->ws[q];
    s->g.fmt=s->u.fmt=s->d.fmt=4;
    s->g.gs=s->u.gs=s->d.gs=64;
    s->g.I=s->u.I=4; s->g.O=s->u.O=3;
    s->d.I=3; s->d.O=4;
    /* The fixture is normally loaded as the legacy format.  Give the test
     * seam valid grouped views so the parallel path reaches its post-load
     * failure hook without executing made-up weights. */
    s->g.q4=(uint8_t*)s->g.q8; s->u.q4=(uint8_t*)s->u.q8; s->d.q4=(uint8_t*)s->d.q8;
}
static void *release_after_scheduler_wait(void *vp){
    SchedulerCtl *c=(SchedulerCtl*)vp;
    while(!atomic_load_explicit(&c->l->wait_a_entered,memory_order_acquire)) sched_yield();
    if(atomic_load_explicit(&c->l->drain_after_a,memory_order_acquire))
        atomic_store_explicit(&c->crossed_while_held,1,memory_order_release);
    atomic_store_explicit(&c->l->a_release,1,memory_order_release);
    return NULL;
}
static void scheduler_fatal(int layer){ (void)layer; longjmp(g_scheduler_fatal,1); }

static int test_scheduler_failure_drain(Model *m,int parallel){
    for(int block=0;block<2;block++){
        Layer l={0}; float x[4]={1,2,3,4}, out[4]={0}; float dummy[4]={0};
        DrainLatch dl={0}; SchedulerCtl ctl={.l=&dl}; pthread_t th;
        int pre_idx[2]={0,1}, pre_keff[1]={2}; float pre_w[2]={.5f,.5f};
        m->c=(Cfg){.arch=ARCH_M3,.hidden=4,.n_layers=1,.n_experts=2,.topk=2,
                    .moe_inter=3,.n_shared=1,.shared_inter=3,.norm_topk=1};
        m->ebits=8; m->ecap=2;
        l.sh_gate=l.sh_up=(QT){.fmt=4,.q4=(uint8_t*)dummy,.s=dummy,.O=3,.I=4,.gs=64};
        l.sh_down=(QT){.fmt=4,.q4=(uint8_t*)dummy,.s=dummy,.O=4,.I=3,.gs=64};
        if(!m->eheat){
            m->eheat=calloc(2,sizeof(*m->eheat)); m->elast=calloc(2,sizeof(*m->elast));
            m->eroute=calloc(2,sizeof(*m->eroute)); m->enr=calloc(2,sizeof(*m->enr));
            if(!m->eheat||!m->elast||!m->eroute||!m->enr) return fail("scheduler fixture bookkeeping");
            for(int i=0;i<2;i++){ m->eheat[i]=calloc(2,sizeof(**m->eheat)); m->elast[i]=calloc(2,sizeof(**m->elast)); m->eroute[i]=calloc(2,sizeof(**m->eroute)); }
        }
        g_pipe_block=block; g_m3_dag_serial=1; g_m3_dag_parallel=parallel; g_m3_dag_pipe_ready=1;
        g_pre_idx=pre_idx; g_pre_w=pre_w; g_pre_keff=pre_keff;
        g_drain_latch=&dl; g_pipe_test_before_load=hold_load_a;
        g_pipe_test_after_load=force_grouped_slot; g_pipe_test_m3_run=1;
        g_pipe_test_wait_slot_enter=&dl.wait_a_entered;
        g_pipe_test_wait_any_enter=&dl.wait_a_entered;
        g_pipe_test_drain_after_slot=&dl.drain_after_a; g_pipe_test_fatal=scheduler_fatal;
        if(pthread_create(&th,NULL,release_after_scheduler_wait,&ctl)) return fail("scheduler drain controller");
        int jumped=setjmp(g_scheduler_fatal);
        if(!jumped) moe(m,&l,LAYER,x,1,out,1);
        pthread_join(th,NULL);
        g_pre_idx=NULL; g_pre_w=NULL; g_pre_keff=NULL;
        g_pipe_test_fatal=NULL; g_pipe_test_m3_run=0; g_pipe_test_after_load=NULL;
        g_pipe_test_drain_after_slot=NULL; g_pipe_test_wait_slot_enter=NULL;
        g_pipe_test_wait_any_enter=NULL;
        g_pipe_test_before_load=NULL; g_drain_latch=NULL;
        g_m3_dag_serial=0; g_m3_dag_parallel=0; g_m3_dag_pipe_ready=0;
        if(!jumped || atomic_load_explicit(&ctl.crossed_while_held,memory_order_acquire) ||
           !atomic_load_explicit(&dl.drain_after_a,memory_order_acquire) ||
           !atomic_load_explicit(&g_pp.ready[0],memory_order_acquire) ||
           !atomic_load_explicit(&g_pp.ready[1],memory_order_acquire))
            return fail(block?"scheduler failure drain (block)":"scheduler failure drain (spin)");
    }
    return 0;
}

int main(void){
    if(test_implication_table()) return 1;

    /* Relative to the CWD, like test_compat_direct's TMPF — NOT "/tmp/...":
     * the windows job builds native .exe files and "/tmp" is not a Windows
     * path. fwrite then reopen read-only: Windows compat has pread, not pwrite. */
    if(write_fixture()) return 1;
    int fd=open(TMPF,COMPAT_O_RDONLY);
    if(fd<0) return fail("open temp");

    static Model m;                                   /* zeroed: buffered pread path, no mmap/cuda */
    if(build_fixture(&m,fd)){ close(fd); remove(TMPF); return 1; }

    g_pipe=1; g_pipe_nw=4;
    pipe_init(&m);

    if(getenv("COLI_TEST_PARALLEL_FAILURE")){
        int rc=test_scheduler_failure_drain(&m,1);
        for(int q=0;q<NE;q++){ compat_aligned_free(m.ws[q].slab); free(m.ws[q].fslab); }
        for(int i=0;i<m.S.n;i++) free(m.S.t[i].name);
        free(m.S.t); close(fd); remove(TMPF);
        if(rc) return 1;
        puts("test_pipe_block: parallel failure drain ok");
        return 0;
    }

    /* spin waiter first (control), then the condvar waiter under the same
     * dispatch pattern; 200 generations each cycles the gen-tagged cursor
     * and alternates parked/fast-path waits. */
    if(run_generations(&m,0,200) || run_generations(&m,1,200) || test_wait_any_ready_first() ||
       test_failure_drain_live_load(&m) || test_scheduler_failure_drain(&m,0) ||
       test_scheduler_failure_drain(&m,1)){
        close(fd); remove(TMPF); return 1;
    }

    for(int q=0;q<NE;q++){ compat_aligned_free(m.ws[q].slab); free(m.ws[q].fslab); }
    for(int i=0;i<m.S.n;i++) free(m.S.t[i].name);
    free(m.S.t);
    close(fd);
    remove(TMPF);
    puts("test_pipe_block: ok");
    return 0;
}
