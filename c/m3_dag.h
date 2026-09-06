/*
 * Internal MiniMax-M3 execution-DAG contract.
 *
 * Step 3 owns the state machine and serial adapter; Step 6 adds lifecycle
 * transitions for a bounded OpenMP task team.  The caller owns model storage,
 * input/output, and the numerical kernel.  A task's weight_handle is opaque
 * here so an ESlot cannot become its identity.  PIPE/parallel callers must
 * retain these identity, readiness, and release checks.
 */
#ifndef COLIBRI_M3_DAG_H
#define COLIBRI_M3_DAG_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    M3_DAG_WEIGHT_ABSENT = 0,
    M3_DAG_WEIGHT_QUEUED,
    M3_DAG_WEIGHT_LOADING,
    M3_DAG_WEIGHT_RESIDENT,
    M3_DAG_WEIGHT_FAILED
} M3DagWeightState;

typedef enum {
    M3_DAG_COMPUTE_WAITING = 0,
    M3_DAG_COMPUTE_READY,
    M3_DAG_COMPUTE_SUBMITTED,
    M3_DAG_COMPUTE_RUNNING,
    M3_DAG_COMPUTE_OUTPUT_READY,
    M3_DAG_COMPUTE_COMPLETE,
    M3_DAG_COMPUTE_FAILED,
    M3_DAG_COMPUTE_CANCELLED
} M3DagComputeState;

typedef enum {
    M3_DAG_ROUTED = 0,
    M3_DAG_SHARED = 1
} M3DagTaskKind;

typedef struct {
    uint64_t model_id;
    uint64_t forward_id;
    uint64_t generation;
    uint32_t layer;
} M3DagIdentity;

typedef struct {
    M3DagIdentity id;
    size_t resident_budget;
    int cancelled;
    int failed;
} M3DagExecutionContext;

typedef struct {
    M3DagIdentity id;
    M3DagTaskKind kind;
    int expert_id;
    int route_index;             /* exact committed router position; -1 for shared */
    void *weight_handle;         /* borrowed model/cache handle, never task identity */
    const void *input;
    void *scratch;
    void *output;
    M3DagWeightState weight_state;
    M3DagComputeState compute_state;
    unsigned input_ready;
    unsigned route_committed;
    unsigned weight_lease;
    unsigned output_committed;
} M3DagExpertTask;

typedef int (*M3DagWeightAcquire)(void *opaque, M3DagExpertTask *task);
typedef void (*M3DagWeightRelease)(void *opaque, M3DagExpertTask *task);
typedef int (*M3DagRunTask)(void *opaque, M3DagExpertTask *task);
typedef int (*M3DagCommitTask)(void *opaque, M3DagExpertTask *task);

typedef struct {
    M3DagWeightAcquire acquire;
    M3DagWeightRelease release;
    M3DagRunTask run;
    M3DagCommitTask commit;
    void *opaque;
} M3DagSerialOps;

static inline int m3_dag_identity_equal(M3DagIdentity a, M3DagIdentity b){
    return a.model_id==b.model_id && a.forward_id==b.forward_id &&
           a.generation==b.generation && a.layer==b.layer;
}

static inline void m3_dag_task_init(M3DagExpertTask *task, M3DagIdentity id,
                                    M3DagTaskKind kind, int expert_id, int route_index){
    *task=(M3DagExpertTask){0};
    task->id=id; task->kind=kind; task->expert_id=expert_id; task->route_index=route_index;
    task->weight_state=M3_DAG_WEIGHT_ABSENT;
    task->compute_state=M3_DAG_COMPUTE_WAITING;
}

/* State transitions are intentionally monotonic.  A caller that wants a new
 * generation must create a new task; it must never recycle a live descriptor. */
static inline int m3_dag_weights_queued(M3DagExpertTask *task){
    if(task->weight_state!=M3_DAG_WEIGHT_ABSENT) return 0;
    task->weight_state=M3_DAG_WEIGHT_QUEUED; return 1;
}
static inline int m3_dag_weights_loading(M3DagExpertTask *task){
    if(task->weight_state!=M3_DAG_WEIGHT_QUEUED) return 0;
    task->weight_state=M3_DAG_WEIGHT_LOADING; return 1;
}
static inline int m3_dag_weights_resident(M3DagExpertTask *task, void *handle){
    if(task->weight_state!=M3_DAG_WEIGHT_LOADING || !handle) return 0;
    task->weight_handle=handle; task->weight_state=M3_DAG_WEIGHT_RESIDENT; return 1;
}
static inline int m3_dag_weights_fail(M3DagExpertTask *task){
    if(task->weight_state!=M3_DAG_WEIGHT_LOADING && task->weight_state!=M3_DAG_WEIGHT_QUEUED) return 0;
    task->weight_state=M3_DAG_WEIGHT_FAILED; task->compute_state=M3_DAG_COMPUTE_FAILED; return 1;
}
static inline int m3_dag_task_ready(M3DagExecutionContext *ctx, M3DagExpertTask *task){
    if(!ctx || !task || ctx->cancelled || ctx->failed || !m3_dag_identity_equal(ctx->id,task->id) ||
       !task->input_ready || !task->input || task->weight_state!=M3_DAG_WEIGHT_RESIDENT ||
       !task->weight_handle || task->compute_state!=M3_DAG_COMPUTE_WAITING) return 0;
    if(task->kind==M3_DAG_ROUTED && (!task->route_committed || task->route_index<0)) return 0;
    if(task->kind==M3_DAG_SHARED && task->route_index!=-1) return 0;
    task->compute_state=M3_DAG_COMPUTE_READY;
    return 1;
}

/* Execute one already-ready task.  Compute completion and reduction commitment
 * are separate so a duplicate notification cannot add an output twice. */
static inline int m3_dag_serial_execute(M3DagExecutionContext *ctx, M3DagExpertTask *task,
                                        const M3DagSerialOps *ops){
    if(!ctx || !task || !ops || !ops->run || !ops->commit ||
       !m3_dag_identity_equal(ctx->id,task->id)) return 0;
    if(ctx->failed || task->compute_state==M3_DAG_COMPUTE_COMPLETE ||
       task->compute_state==M3_DAG_COMPUTE_FAILED || task->compute_state==M3_DAG_COMPUTE_CANCELLED) return 0;
    if(ctx->cancelled){
        if(task->compute_state==M3_DAG_COMPUTE_WAITING || task->compute_state==M3_DAG_COMPUTE_READY)
            task->compute_state=M3_DAG_COMPUTE_CANCELLED;
        return 0;
    }
    if(task->compute_state!=M3_DAG_COMPUTE_READY || task->weight_lease || task->output_committed) return 0;
    if(ops->acquire && !ops->acquire(ops->opaque,task)){ ctx->failed=1; task->compute_state=M3_DAG_COMPUTE_FAILED; return 0; }
    task->weight_lease=1;
    task->compute_state=M3_DAG_COMPUTE_RUNNING;
    if(!ops->run(ops->opaque,task)){
        ctx->failed=1; task->compute_state=M3_DAG_COMPUTE_FAILED;
        task->weight_lease=0; if(ops->release) ops->release(ops->opaque,task); return 0;
    }
    task->compute_state=M3_DAG_COMPUTE_COMPLETE;
    if(ctx->cancelled || !ops->commit(ops->opaque,task)){
        if(!ctx->cancelled) ctx->failed=1;
        task->compute_state=ctx->cancelled?M3_DAG_COMPUTE_CANCELLED:M3_DAG_COMPUTE_FAILED;
        task->weight_lease=0; if(ops->release) ops->release(ops->opaque,task); return 0;
    }
    task->output_committed=1;
    task->weight_lease=0;
    if(ops->release) ops->release(ops->opaque,task);
    return 1;
}

/* Parallel-task lifecycle.  The OpenMP producer owns submission and the
 * reducer owns commitment; one coordinator owns the intervening transitions.
 * These helpers intentionally remain non-atomic: taskwait is the publication
 * boundary and no descriptor is touched by two owners at once. */
static inline int m3_dag_parallel_submit(M3DagExpertTask *task){
    if(!task || task->compute_state!=M3_DAG_COMPUTE_READY || !task->weight_lease ||
       task->output_committed) return 0;
    task->compute_state=M3_DAG_COMPUTE_SUBMITTED; return 1;
}
static inline int m3_dag_context_failed(const M3DagExecutionContext *ctx){
    return ctx && __atomic_load_n(&ctx->failed,__ATOMIC_ACQUIRE);
}
static inline int m3_dag_context_cancelled(const M3DagExecutionContext *ctx){
    return ctx && __atomic_load_n(&ctx->cancelled,__ATOMIC_ACQUIRE);
}
static inline void m3_dag_context_fail(M3DagExecutionContext *ctx){
    if(ctx) __atomic_store_n(&ctx->failed,1,__ATOMIC_RELEASE);
}
static inline void m3_dag_context_cancel(M3DagExecutionContext *ctx){
    if(ctx) __atomic_store_n(&ctx->cancelled,1,__ATOMIC_RELEASE);
}
static inline int m3_dag_parallel_start(M3DagExpertTask *task){
    if(!task || task->compute_state!=M3_DAG_COMPUTE_SUBMITTED) return 0;
    task->compute_state=M3_DAG_COMPUTE_RUNNING; return 1;
}
static inline int m3_dag_parallel_output_ready(M3DagExpertTask *task){
    if(!task || task->compute_state!=M3_DAG_COMPUTE_RUNNING) return 0;
    task->compute_state=M3_DAG_COMPUTE_OUTPUT_READY; return 1;
}
static inline int m3_dag_parallel_fail(M3DagExpertTask *task){
    if(!task || (task->compute_state!=M3_DAG_COMPUTE_SUBMITTED &&
                 task->compute_state!=M3_DAG_COMPUTE_RUNNING)) return 0;
    task->compute_state=M3_DAG_COMPUTE_FAILED; return 1;
}
static inline int m3_dag_parallel_reduce(M3DagExpertTask *task){
    if(!task || task->compute_state!=M3_DAG_COMPUTE_OUTPUT_READY ||
       task->output_committed) return 0;
    task->output_committed=1; task->compute_state=M3_DAG_COMPUTE_COMPLETE; return 1;
}

static inline int m3_dag_task_terminal(const M3DagExpertTask *task){
    if(!task || task->weight_lease) return 0;
    if(task->compute_state==M3_DAG_COMPUTE_COMPLETE) return task->output_committed!=0;
    return task->compute_state==M3_DAG_COMPUTE_FAILED || task->compute_state==M3_DAG_COMPUTE_CANCELLED;
}
static inline int m3_dag_layer_retired(const M3DagExecutionContext *ctx,
                                       const M3DagExpertTask *tasks, size_t count){
    if(!ctx || !tasks) return 0;
    for(size_t i=0;i<count;i++) if(!m3_dag_identity_equal(ctx->id,tasks[i].id) || !m3_dag_task_terminal(&tasks[i])) return 0;
    return 1;
}

#endif
