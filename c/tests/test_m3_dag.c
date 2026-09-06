/* Contract/lifecycle gate for the serial MiniMax-M3 DAG executor. */
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdatomic.h>
#include "../m3_dag.h"

typedef struct { int acquire, release, run, commit, sum, fail_run, cancel_in_run; } Probe;
static int acquire(void *p, M3DagExpertTask *t){ (void)t; ((Probe*)p)->acquire++; return 1; }
static void release(void *p, M3DagExpertTask *t){ (void)t; ((Probe*)p)->release++; }
static int run(void *p, M3DagExpertTask *t){ Probe *q=p; q->run++; if(q->cancel_in_run) return 1; return !q->fail_run; }
static int commit(void *p, M3DagExpertTask *t){ Probe *q=p; q->commit++; q->sum=q->sum*10+t->expert_id+1; return 1; }
static int fail(const char *s){ fprintf(stderr,"FAIL: %s\n",s); return 1; }
static void ready(M3DagExecutionContext *c, M3DagExpertTask *t, void *handle){
    t->input=(const void*)t; t->input_ready=1; t->route_committed=t->kind==M3_DAG_ROUTED;
    if(!m3_dag_weights_queued(t) || !m3_dag_weights_loading(t) || !m3_dag_weights_resident(t,handle) || !m3_dag_task_ready(c,t))
        fprintf(stderr,"setup failure\n");
}

typedef struct {
    M3DagExecutionContext *ctx;
    _Atomic int start;
    _Atomic int armed;
    _Atomic int allow_fail;
} FailurePublicationProbe;

static void *publish_failure_after_readers_start(void *opaque){
    FailurePublicationProbe *p=(FailurePublicationProbe*)opaque;
    while(!atomic_load_explicit(&p->start,memory_order_acquire)) sched_yield();
    atomic_store_explicit(&p->armed,1,memory_order_release);
    while(!atomic_load_explicit(&p->allow_fail,memory_order_acquire)) sched_yield();
    m3_dag_context_fail(p->ctx);
    return NULL;
}

/* Regression for the producer/worker publication edge: task_ready() must use
 * the same atomic failure publication as the worker.  The worker is held
 * until the producer has performed many readiness reads, then publishes the
 * failure while those reads continue.  Once the producer observes the
 * publication, no later readiness probe may accept the task. */
static int check_failure_publication_interleave(M3DagIdentity id, void *handle){
    M3DagExecutionContext ctx={.id=id};
    M3DagExpertTask task;
    m3_dag_task_init(&task,id,M3_DAG_ROUTED,12,0);
    ready(&ctx,&task,handle);
    task.compute_state=M3_DAG_COMPUTE_WAITING;
    FailurePublicationProbe probe={.ctx=&ctx};
    pthread_t worker;
    if(pthread_create(&worker,NULL,publish_failure_after_readers_start,&probe))
        return fail("failure-publication probe thread creation");
    atomic_store_explicit(&probe.start,1,memory_order_release);
    while(!atomic_load_explicit(&probe.armed,memory_order_acquire)) sched_yield();

    int observed=0, late_ready=0;
    for(int i=0;i<200000;i++){
        task.compute_state=M3_DAG_COMPUTE_WAITING;
        if(i==1000) atomic_store_explicit(&probe.allow_fail,1,memory_order_release);
        int accepted=m3_dag_task_ready(&ctx,&task);
        if(m3_dag_context_failed(&ctx)){
            observed=1;
            if(accepted) late_ready=1;
        }
    }
    pthread_join(worker,NULL);
    task.compute_state=M3_DAG_COMPUTE_WAITING;
    if(!observed || !m3_dag_context_failed(&ctx) || late_ready || m3_dag_task_ready(&ctx,&task))
        return fail("failure publication allowed readiness after worker failure");
    puts("  atomic failure publication/readiness interleave ok");
    return 0;
}

int main(void){
    M3DagIdentity id={.model_id=7,.forward_id=11,.generation=13,.layer=17};
    M3DagExecutionContext ctx={.id=id,.resident_budget=99};
    M3DagExpertTask tasks[3];
    m3_dag_task_init(&tasks[0],id,M3_DAG_ROUTED,2,0);
    m3_dag_task_init(&tasks[1],id,M3_DAG_ROUTED,5,1);
    m3_dag_task_init(&tasks[2],id,M3_DAG_SHARED,9,-1);
    int handles[3]={1,2,3}; for(int i=0;i<3;i++) ready(&ctx,&tasks[i],&handles[i]);
    if(check_failure_publication_interleave(id,&handles[0])) return 1;
    Probe p={0}; M3DagSerialOps ops={acquire,release,run,commit,&p};
    for(int i=0;i<3;i++) if(!m3_dag_serial_execute(&ctx,&tasks[i],&ops)) return fail("serial task did not execute");
    if(p.sum!=370 || p.acquire!=3 || p.release!=3 || p.commit!=3 || !m3_dag_layer_retired(&ctx,tasks,3)) return fail("baseline order/release");
    if(m3_dag_serial_execute(&ctx,&tasks[2],&ops) || p.commit!=3) return fail("duplicate completion committed output");

    M3DagExecutionContext old_ctx={.id={7,11,12,17}}; M3DagExpertTask stale;
    m3_dag_task_init(&stale,old_ctx.id,M3_DAG_ROUTED,1,0);
    ready(&old_ctx,&stale,&handles[0]);
    if(m3_dag_serial_execute(&ctx,&stale,&ops) || p.acquire!=3) return fail("stale generation accepted");

    M3DagExecutionContext cancelled={.id=id}; M3DagExpertTask ct;
    m3_dag_task_init(&ct,id,M3_DAG_ROUTED,3,0); ready(&cancelled,&ct,&handles[0]); m3_dag_context_cancel(&cancelled);
    if(m3_dag_serial_execute(&cancelled,&ct,&ops) || ct.compute_state!=M3_DAG_COMPUTE_CANCELLED || p.acquire!=3) return fail("cancelled task ran");

    M3DagExecutionContext failed={.id=id}; M3DagExpertTask ft,blocked; Probe fp={.fail_run=1};
    M3DagSerialOps fops={acquire,release,run,commit,&fp};
    m3_dag_task_init(&ft,id,M3_DAG_ROUTED,4,0); ready(&failed,&ft,&handles[0]);
    m3_dag_task_init(&blocked,id,M3_DAG_ROUTED,6,1); ready(&failed,&blocked,&handles[1]);
    if(m3_dag_serial_execute(&failed,&ft,&fops) || !m3_dag_context_failed(&failed) || ft.weight_lease || fp.acquire!=1 || fp.release!=1 || fp.commit) return fail("error path lease/commit");
    if(m3_dag_serial_execute(&failed,&blocked,&fops) || fp.acquire!=1) return fail("failed context ran another task");
    M3DagExecutionContext parallel_ctx={.id=id}; M3DagExpertTask pt;
    m3_dag_task_init(&pt,id,M3_DAG_ROUTED,8,0); ready(&parallel_ctx,&pt,&handles[0]);
    pt.weight_lease=1;
    if(!m3_dag_parallel_submit(&pt) || !m3_dag_parallel_start(&pt) ||
       !m3_dag_parallel_output_ready(&pt) || (pt.weight_lease=0, !m3_dag_parallel_reduce(&pt)) ||
       !m3_dag_task_terminal(&pt) || m3_dag_parallel_reduce(&pt)) return fail("parallel lifecycle/duplicate reduce");
    M3DagExpertTask pf; m3_dag_task_init(&pf,id,M3_DAG_ROUTED,10,0); ready(&parallel_ctx,&pf,&handles[0]);
    pf.weight_lease=1;
    if(!m3_dag_parallel_submit(&pf) || !m3_dag_parallel_fail(&pf) || (pf.weight_lease=0, !m3_dag_task_terminal(&pf)) ||
       m3_dag_parallel_output_ready(&pf)) return fail("parallel failure lifecycle");
    printf("test_m3_dag: ok (serial order, shared once, stale, cancel, failure, publication race)\n");
    return 0;
}
