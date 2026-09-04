/* qwen36_tier.c -- CUDA VRAM expert tier for the qwen36 engine. See header. */
#ifdef COLI_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#ifdef __linux__
#include <errno.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "qwen36_tier.h"
#include "qwen36_topology.h"
#include "backend_cuda.h"
#include "tier.h"

#define QT_MAX_DEV 8
#define QT_QCAP 48            /* upload queue depth (staging ~1.6 MB/entry) */
#define QT_STAGE_MAX 128      /* execution-cache experts per GPU island */

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    uint32_t heat;
    uint8_t resident, queued, planned;
    /* raw RAM pointers (slots are never evicted when cap==n_experts) -- lets
     * warmstart, lookahead and LFRU swaps run without an engine callback */
    const uint8_t *g4,*u4,*d4; const float *gs,*us,*ds;
    int disk_replica, disk_controller;
} QSlot;

typedef struct {
    ColiCudaTensor *qkv;
    ColiCudaTensor *z;
    ColiCudaTensor *out;
    int qkv_o, z_o, out_o;
    ColiCudaTensor *attn_q, *attn_k, *attn_v, *attn_o;
    int attn_q_o, attn_k_o, attn_v_o, attn_o_o, attn_o_i;
    ColiCudaTensor *shared_g, *shared_u, *shared_d;
    int shared_g_o, shared_u_o, shared_d_o, shared_d_i;
    /* Full DeltaNet island resources. These are device allocations, not
     * expert-cache slots, and survive across decode tokens. */
    float *conv, *b, *a, *dtbias, *alog, *norm;
    float *rec, *ring;
    int full;
} QtDenseLayer;

static struct {
    int on, nl, ne, D, Ih, topk, ndev;
    int egs; size_t sc_gu, sc_d;   /* expert group size + per-matrix scale counts (gs64) */
    /* Formato dei pesi che il tier spedisce in VRAM: 4 = int4 raggruppato
     * (container gs64), 1 = int8 per-riga. Prima era cablato a 4 in ogni punto,
     * e su un container int8 -- dove s->g4 e' NULL perche' non c'e' nulla da
     * impacchettare -- il tier riservava budget, marcava planned=1 e non
     * promuoveva mai niente, senza dire una parola (#1331). backend_cuda sa
     * gia' leggere fmt=1: mancava solo che glielo offrissimo. */
    int wfmt;
    int dev[QT_MAX_DEV];
    size_t budget[QT_MAX_DEV], used[QT_MAX_DEV];
    size_t exp_bytes;                     /* estimated VRAM bytes per expert */
    QSlot *slot;                          /* [nl*ne] */
    pthread_mutex_t mx;
    pthread_t th;
    int th_stop;
    /* upload ring with staging copies */
    struct { int layer, eid; uint8_t *w; float *s; int v_layer, v_eid; } q[QT_QCAP];
    int qh, qt_, qn;
    pthread_cond_t cv;
    /* statistics */
    uint64_t hits[QT_MAX_DEV], miss, uploads, q_full_skips;
    uint64_t batch_calls, batch_routes;
    /* issue state of the (single) decode thread */
    int is_cnt[QT_MAX_DEV];
    int is_k[QT_MAX_DEV][64];
    int is_rows[64];
    ColiCudaTensor *is_tg[QT_MAX_DEV][64],*is_tu[QT_MAX_DEV][64],*is_td[QT_MAX_DEV][64];
    float *is_x;                          /* bounded prefill input replicas per device */
    /* Stage 3A: one persistent home-device activation/partial arena.  Resident
     * groups compute on their owning GPUs and return one weighted partial per
     * device; only the final reduced row is downloaded. */
    int resident_active, resident_home, resident_nissued;
    int resident_devices[QT_MAX_DEV];
    float *resident_buf, *resident_x_dev, *resident_slots_dev, *resident_acc_dev;
    float *resident_host;
    /* M3 */
    int *fill_order; int fill_cur;        /* warmstart order (heat desc) */
    int issue_open;                       /* guard: no tensor_free while a group is in flight */
    pthread_cond_t cv_take;               /* signals qt_take done + queue space */
    uint64_t tick, swaps, pf_hits, pf_notes;
    uint64_t resident_calls, resident_experts, resident_fallbacks;
    uint64_t resident_single_direct;
    double resident_h2d_us, resident_issue_us, resident_reduce_us, resident_d2h_us;
    double resident_gpu_event_us, resident_sync_wait_us, resident_take_api_us;
    double resident_reduce_event_us, resident_d2h_diag_us, resident_take_total_us;
    uint64_t resident_timing_calls;
    double qt_take_diag_us;
    uint64_t qt_take_diag_calls;
    double qt_last_gpu_event_us, qt_last_sync_wait_us, qt_last_take_wall_us;
    double qt_last_gpu_lower_ms, qt_last_gpu_upper_ms;
    double qt_last_reduce_lower_ms, qt_last_reduce_upper_ms;
    double qt_last_gpu_complete_ms, qt_last_merge_begin_ms;
    int qt_last_gpu_valid;
    double *qt_take_layer_us, *qt_take_layer_max_us;
    uint64_t *qt_take_layer_calls, qt_take_bins[8];
    uint64_t *qt_issue_layer_routes, *qt_issue_layer_gpu;
    int timing_layer;
    int resident_timing_decode;
    uint32_t *heat0;                      /* heat table loaded from HEAT_FILE */
    QtTopology topology;                  /* GPU/RAM/storage locality snapshot */
    double timing_pre_us, timing_issue_us;
    /* Execution-plane counters.  These deliberately count the new adapter
     * boundary, not CUDA diagnostic calls: one descriptor is a layer/token
     * submission and one island batch is one local executor invocation. */
    uint64_t exec_descriptors, exec_island_batches, exec_island_experts;
    /* Optional execution-only overflow slots. These do not become resident
     * cache entries: a routed CPU miss is copied into a reserved temporary
     * tensor and consumed in the current layer batch. Disabled unless
     * COLI_CUDA_STAGE_MISSES=1. */
    int stage_slots;
    ColiCudaTensor *stage_g[QT_MAX_DEV][QT_STAGE_MAX];
    ColiCudaTensor *stage_u[QT_MAX_DEV][QT_STAGE_MAX];
    ColiCudaTensor *stage_d[QT_MAX_DEV][QT_STAGE_MAX];
    uint8_t *stage_w[QT_MAX_DEV][QT_STAGE_MAX];
    float *stage_s[QT_MAX_DEV][QT_STAGE_MAX];
    int stage_layer[QT_MAX_DEV][QT_STAGE_MAX];
    int stage_eid[QT_MAX_DEV][QT_STAGE_MAX];
    uint64_t stage_lru[QT_MAX_DEV][QT_STAGE_MAX];
    uint64_t stage_clock, stage_cache_hits;
    uint64_t stage_calls, stage_updates, stage_failures;
    /* Dense island prototype. The projection mode batches qkv/z; the full
     * mode keeps DeltaNet recurrent state and the convolution ring on the
     * device. This is intentionally separate from expert slots. */
    QtDenseLayer *dense;
    ColiCudaTensor *dense_lm_head;
    int dense_lm_I, dense_lm_O;
    int dense_device, dense_enabled;
    float *dense_pack; size_t dense_pack_cap;
    uint64_t dense_registers, dense_calls, dense_fallbacks;
    uint64_t dense_full_calls, dense_full_fallbacks;
    uint64_t dense_lm_calls, dense_lm_fallbacks;
    uint64_t dense_attn_calls, dense_attn_fallbacks;
    uint64_t dense_shared_calls, dense_shared_fallbacks;
    uint64_t dense_shared_coarse_calls, dense_shared_coarse_fallbacks;
    int dense_vh, dense_vk, dense_kdim, dense_vdim, dense_convk, dense_conv_dim;
    float dense_eps;
} G;

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return qt_topology_home_gpu(&G.topology, eid); }
static double qt_now_us(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec*1000000.0+(double)ts.tv_nsec/1000.0;
}

static void qt_record_take_diag(double us){
    if(!G.resident_timing_decode) return;
    G.qt_take_diag_us += us;
    G.qt_take_diag_calls++;
    int b = us < 1000.0 ? 0 : us < 2000.0 ? 1 : us < 4000.0 ? 2 :
            us < 8000.0 ? 3 : us < 16000.0 ? 4 : us < 32000.0 ? 5 :
            us < 64000.0 ? 6 : 7;
    G.qt_take_bins[b]++;
    int l=G.timing_layer;
    if(l>=0 && l<G.nl && G.qt_take_layer_us && G.qt_take_layer_calls){
        G.qt_take_layer_us[l]+=us;
        G.qt_take_layer_calls[l]++;
        if(us>G.qt_take_layer_max_us[l]) G.qt_take_layer_max_us[l]=us;
    }
}

static void qt_record_issue_diag(int layer,int routes,uint32_t mask){
    if(!G.resident_timing_decode || layer<0 || layer>=G.nl ||
       !G.qt_issue_layer_routes || !G.qt_issue_layer_gpu) return;
    G.qt_issue_layer_routes[layer]+=(uint64_t)routes;
    for(int k=0;k<routes;k++) if(mask&(1u<<k)) G.qt_issue_layer_gpu[layer]++;
}

/* ------------------------------------------------------------------------- */
/* Layer/island execution boundary                                           */

/* This is the first concrete split between the Colibri control plane and an
 * island execution plane.  qt_issue() owns routing-derived placement and
 * fills one of these records per active device.  The executor owns the local
 * submission details.  The initial implementation intentionally delegates to
 * the already validated resident CUDA entry point; keeping this boundary
 * exact lets later implementations replace the CUDA envelope without
 * changing routing, placement, CPU fallback, or reduction order. */
typedef struct {
    int layer;
    int device;
    int count;
    int hidden;
    int intermediate;
    const float *input_dev;
    const float *route_weights;
    int route_index[64];
    ColiCudaTensor *gate[64];
    ColiCudaTensor *up[64];
    ColiCudaTensor *down[64];
} QtIslandWork;

/* Compute-Islands v0 layer descriptor.  The control plane still chooses the
 * resident subset per device; this descriptor only makes the complete local
 * work assignment explicit before submission.  The executor below preserves
 * the existing island order and calls the existing validated resident entry
 * point, so it cannot change arithmetic or reduction ordering. */
typedef struct {
    int layer;
    int home_device;
    int count;
    QtIslandWork islands[QT_MAX_DEV];
} QtLayerWork;

static int qt_execute_gpu_island(const QtIslandWork *work,
                                 int home_device, float *partial_slot_dev) {
    if (!work || work->count < 1 || work->count > 64 ||
        !work->input_dev || !work->route_weights || !partial_slot_dev)
        return 0;

    float weights[64];
    for (int i = 0; i < work->count; ++i) {
        int route = work->route_index[i];
        if (route < 0 || route >= 64 || !work->gate[i] ||
            !work->up[i] || !work->down[i])
            return 0;
        weights[i] = work->route_weights[route];
    }

    G.exec_island_batches++;
    G.exec_island_experts += (uint64_t)work->count;
    return coli_cuda_expert_group_resident_issue(
        work->gate, work->up, work->down, weights, work->count,
        home_device, work->input_dev, partial_slot_dev);
}

static int qt_execute_gpu_layer(const QtLayerWork *layer,
                                float *partial_slots_dev,
                                int *devices, int *nissued,
                                int timing_on, double *issue_us) {
    if (!layer || layer->count < 1 || layer->count > QT_MAX_DEV ||
        !partial_slots_dev || !devices || !nissued)
        return 0;
    int n = 0;
    double elapsed = 0.0;
    for (int i = 0; i < layer->count; ++i) {
        double start = timing_on ? qt_now_us() : 0.0;
        if (!qt_execute_gpu_island(&layer->islands[i], layer->home_device,
                                   partial_slots_dev + (size_t)n * G.D)) {
            *nissued = n;
            if (issue_us) *issue_us += elapsed;
            return 0;
        }
        if (timing_on) elapsed += qt_now_us() - start;
        devices[n++] = layer->islands[i].device;
    }
    *nissued = n;
    if (issue_us) *issue_us += elapsed;
    return n > 0;
}

/* Staging: packed int4 (g|u|d) two's-complement -> offset-binary (XOR 0x88,
 * the upload format of backend_cuda fmt=2) + copy the scales (gs|us|ds). */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                  const float *gs,const float *us,const float *ds){
    size_t mb = (size_t)G.D*G.Ih/(G.wfmt==1?1:2);
    if(G.wfmt==1){
        /* int8: il formato del backend e' gia' quello in RAM, si copia e basta.
         * Niente XOR: quello serve a portare i nibble int4 da complemento a due
         * a binario sfalsato, e su byte interi sarebbe corruzione. */
        memcpy(dw,        g4, mb);
        memcpy(dw+mb,     u4, mb);
        memcpy(dw+2*mb,   d4, mb);
    } else {
    const uint64_t X=0x8888888888888888ull;
    const uint64_t *sg=(const uint64_t*)g4,*su=(const uint64_t*)u4,*sd=(const uint64_t*)d4;
    uint64_t *w0=(uint64_t*)dw,*w1=(uint64_t*)(dw+mb),*w2=(uint64_t*)(dw+2*mb);
    for(size_t i=0;i<mb/8;i++){ w0[i]=sg[i]^X; w1[i]=su[i]^X; w2[i]=sd[i]^X; }
    }
    memcpy(dsc,                 gs, G.sc_gu*sizeof(float));
    memcpy(dsc+G.sc_gu,         us, G.sc_gu*sizeof(float));
    memcpy(dsc+2*G.sc_gu,       ds, G.sc_d *sizeof(float));
}

static void qt_stage_shutdown(void) {
    for (int di=0; di<G.ndev; di++) for (int si=0; si<G.stage_slots; si++) {
        if (G.stage_g[di][si]) coli_cuda_tensor_free(G.stage_g[di][si]);
        if (G.stage_u[di][si]) coli_cuda_tensor_free(G.stage_u[di][si]);
        if (G.stage_d[di][si]) coli_cuda_tensor_free(G.stage_d[di][si]);
        G.stage_g[di][si]=G.stage_u[di][si]=G.stage_d[di][si]=NULL;
        free(G.stage_w[di][si]); G.stage_w[di][si]=NULL;
        free(G.stage_s[di][si]); G.stage_s[di][si]=NULL;
    }
    G.stage_slots=0;
}

/* Reserve a small set of execution-only tensors.  Their contents are updated
 * per layer, but their device allocations never participate in the residency
 * table or LFRU.  The budget subtraction happens before warmstart, so enabling
 * this experiment cannot silently exceed the configured VRAM cap. */
static int qt_stage_init(void) {
    if (G.stage_slots <= 0 || !G.egs) return 0;
    size_t mb=(size_t)G.D*G.Ih/2;
    size_t wb=3*mb;
    size_t scn=2*G.sc_gu+G.sc_d;
    uint8_t *zero_w=(uint8_t*)calloc(wb,1);
    float *one_s=(float*)malloc(scn*sizeof(float));
    if(!zero_w || !one_s){ free(zero_w); free(one_s); qt_stage_shutdown(); return 0; }
    for(size_t i=0;i<scn;i++) one_s[i]=1.f;
    for(int di=0;di<G.ndev;di++) for(int si=0;si<G.stage_slots;si++) {
        G.stage_layer[di][si]=-1;
        G.stage_eid[di][si]=-1;
        G.stage_lru[di][si]=0;
    }
    int ok=1;
    for(int di=0;di<G.ndev && ok;di++) for(int si=0;si<G.stage_slots;si++) {
        G.stage_w[di][si]=(uint8_t*)malloc(wb);
        G.stage_s[di][si]=(float*)malloc(scn*sizeof(float));
        if(!G.stage_w[di][si] || !G.stage_s[di][si]) { ok=0; break; }
        memcpy(G.stage_w[di][si],zero_w,wb);
        memcpy(G.stage_s[di][si],one_s,scn*sizeof(float));
        int dv=G.dev[di];
        ok = coli_cuda_tensor_upload_g(&G.stage_g[di][si],G.stage_w[di][si],
                                       G.stage_s[di][si],4,G.D,G.Ih,dv,G.egs) &&
             coli_cuda_tensor_upload_g(&G.stage_u[di][si],G.stage_w[di][si]+mb,
                                       G.stage_s[di][si]+G.sc_gu,4,G.D,G.Ih,dv,G.egs) &&
             coli_cuda_tensor_upload_g(&G.stage_d[di][si],G.stage_w[di][si]+2*mb,
                                       G.stage_s[di][si]+2*G.sc_gu,4,G.Ih,G.D,dv,G.egs);
        if(!ok) break;
    }
    free(zero_w); free(one_s);
    if(!ok){ qt_stage_shutdown(); return 0; }
    fprintf(stderr,"[qtier] transient GPU execution staging: %d slots/device (cache unchanged)\n",
            G.stage_slots);
    return 1;
}

static void *uploader(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&G.mx);
        while(G.qn==0 && !G.th_stop) pthread_cond_wait(&G.cv,&G.mx);
        if(G.th_stop && G.qn==0){ pthread_mutex_unlock(&G.mx); return NULL; }
        int layer=G.q[G.qh].layer, eid=G.q[G.qh].eid;
        int vl=G.q[G.qh].v_layer, ve=G.q[G.qh].v_eid;
        uint8_t *w=G.q[G.qh].w; float *sc=G.q[G.qh].s;
        G.qh=(G.qh+1)%QT_QCAP; G.qn--;
        pthread_cond_broadcast(&G.cv_take);          /* queue space available */
        if(ve>=0){
            /* LFRU swap: free the victim only when no group is in flight */
            while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
            QSlot *v=qs(vl,ve);
            ColiCudaTensor *a=v->tg,*b=v->tu,*ct=v->td;
            v->tg=v->tu=v->td=NULL;
            pthread_mutex_unlock(&G.mx);
            if(a) coli_cuda_tensor_free(a);
            if(b) coli_cuda_tensor_free(b);
            if(ct) coli_cuda_tensor_free(ct);
        } else pthread_mutex_unlock(&G.mx);

        int dv = G.dev[home(eid)];
        /* passo fra le tre matrici nello staging: int4 impacchettato = mezzo
         * byte per elemento, int8 = uno. */
        size_t mb=(size_t)G.D*G.Ih/(G.wfmt==1?1:2);
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok;
        if(G.wfmt==1){
            /* int8, scale per riga: qt_init ha gia' rifiutato il caso raggruppato,
             * che questo formato non sa esprimere. */
            ok = coli_cuda_tensor_upload(&tg, w,      sc,          1, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   sc+G.Ih,     1, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, sc+2*G.Ih,   1, G.Ih, G.D,  dv);
        } else if(G.egs){
            ok = coli_cuda_tensor_upload_g(&tg, w,      sc,             4, G.D,  G.Ih, dv, G.egs)
              && coli_cuda_tensor_upload_g(&tu, w+mb,   sc+G.sc_gu,     4, G.D,  G.Ih, dv, G.egs)
              && coli_cuda_tensor_upload_g(&td, w+2*mb, sc+2*G.sc_gu,   4, G.Ih, G.D,  dv, G.egs);
        } else {
            ok = coli_cuda_tensor_upload(&tg, w,      sc,          2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   sc+G.Ih,     2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, sc+2*G.Ih,   2, G.Ih, G.D,  dv);
        }
        free(w); free(sc);
        pthread_mutex_lock(&G.mx);
        QSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid); G.used[hd]-=G.exp_bytes;
                G.budget[hd]=G.used[hd];   /* device genuinely full: stop trying */ }
        s->queued=0;
        pthread_mutex_unlock(&G.mx);
    }
}

int qt_init(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs,
            int expert_is_int4){
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(cap != ne){
        fprintf(stderr,"[qtier] cap=%d != n_experts=%d -> tier disabled (needs full RAM residency)\n",cap,ne);
        return 0;
    }
    if(topk>32){ fprintf(stderr,"[qtier] topk>32 unsupported\n"); return 0; }
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk;

    /* devices: COLI_GPUS="0,1" (default: first two visible devices) */
    const char *gl=getenv("COLI_GPUS");
    if (gl && *gl) {
        char buf[128]; snprintf(buf,sizeof buf,"%s",gl);
        for(char *t=strtok(buf,","); t && G.ndev<QT_MAX_DEV; t=strtok(NULL,","))
            G.dev[G.ndev++]=atoi(t);
    } else {
        int available=coli_cuda_available_device_count();
        int want=available<2?available:2;
        for(int i=0;i<want && i<QT_MAX_DEV;i++) G.dev[G.ndev++]=i;
        fprintf(stderr,"[qtier] COLI_GPUS unset: selecting %d visible device(s)\n",G.ndev);
    }
    if(G.ndev<1){ fprintf(stderr,"[qtier] no visible CUDA devices -> CPU path\n"); return 0; }
    if(!coli_cuda_init(G.dev,G.ndev)){ fprintf(stderr,"[qtier] coli_cuda_init failed -> CPU path\n"); return 0; }
    int have=coli_cuda_device_count();
    if(have<G.ndev){ G.ndev=have; }
    if(G.ndev<1){ fprintf(stderr,"[qtier] no CUDA devices -> CPU path\n"); return 0; }
    qt_topology_init(&G.topology, G.dev, G.ndev);
    for(int i=0;i<G.ndev;i++) {
        G.topology.gpu[i].integrated = coli_cuda_device_integrated(G.dev[i]);
        int domain=-1,bus=-1,pci_device=-1,function=-1;
        if(coli_cuda_device_pci(G.dev[i],&domain,&bus,&pci_device,&function))
            qt_topology_set_gpu_pci(&G.topology,i,domain,bus,pci_device,function);
    }
    for(int i=0;i<G.ndev;i++) for(int j=0;j<G.ndev;j++)
        qt_topology_set_peer(&G.topology,i,j,
            i==j || coli_cuda_peer_access(G.dev[i],G.dev[j]));
    qt_topology_report(&G.topology, "[qtier]");

    /* per-device budget: CUDA_EXPERT_GB, or auto = free minus 1 GB headroom.
     * Scale counts follow the container: per-row (expert_gs=0) or grouped
     * (gs64: [O, ceil(I/gs)] per projection). */
    G.wfmt = expert_is_int4 ? 4 : 1;
    /* fmt=1 non ha scale raggruppate: backend_cuda le onora solo per fmt=4
     * (want_gs = (fmt==4 && ...)). Un container int8 con scale a gruppi non e'
     * esprimibile sulla GPU, e promuoverlo lo stesso darebbe numeri sbagliati:
     * meglio restare su CPU dicendolo. Rifiutare qui, PRIMA di riservare
     * qualunque budget, e' il punto giusto -- il difetto di #1331 era proprio
     * riservare per poi non promuovere.  */
    if(G.wfmt==1 && expert_gs>0){
        fprintf(stderr,"[qtier] int8 experts with grouped scales (gs=%d) cannot be "
                       "expressed on the GPU (fmt=1 is per-row only) -> CPU path\n", expert_gs);
        return 0;
    }
    G.egs = expert_gs;
    G.sc_gu = expert_gs ? (size_t)Ih * ((D + expert_gs - 1)/expert_gs) : (size_t)Ih;
    G.sc_d  = expert_gs ? (size_t)D  * ((Ih + expert_gs - 1)/expert_gs) : (size_t)D;
    /* int8 occupa il doppio dell'int4 impacchettato: il budget deve saperlo,
     * se no si promettono il doppio degli esperti che ci stanno. */
    G.exp_bytes = (G.wfmt==1 ? 3ull*D*Ih : 3ull*D*Ih/2)
                + (2*G.sc_gu+G.sc_d)*sizeof(float) + 4096; /* + allocation slack */
    const char *bg=getenv("CUDA_EXPERT_GB");
    const char *dg=getenv("COLI_CUDA_DENSE_GB");
    const char *stage_env=getenv("COLI_CUDA_STAGE_MISSES");
    if(stage_env && atoi(stage_env)) {
        G.stage_slots=8;
        const char *ss=getenv("COLI_CUDA_STAGE_SLOTS");
        if(ss && atoi(ss)>0) G.stage_slots=atoi(ss);
        if(G.stage_slots>QT_STAGE_MAX) G.stage_slots=QT_STAGE_MAX;
    }
    size_t stage_reserve=(size_t)G.stage_slots*G.exp_bytes;
    size_t dense_reserve=(dg && atof(dg)>0.0)
                       ? (size_t)(atof(dg)*1024.0*1024.0*1024.0) : 0;
    for(int i=0;i<G.ndev;i++){
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        size_t b = (bg && strcmp(bg,"auto") && atof(bg)>0)
                   ? (size_t)(atof(bg)*1024.0*1024.0*1024.0)
                   : (freeb>(1ull<<30) ? freeb-(1ull<<30) : 0);
        if(dense_reserve>=b) b=0; else b-=dense_reserve;
        if(stage_reserve>=b) b=0; else b-=stage_reserve;
        G.budget[i]=b;
        fprintf(stderr,"[qtier] dev %d: %.1f GB free, expert budget %.1f GB (~%zu experts), dense reserve %.1f GB, stage reserve %.1f MB\n",
                G.dev[i], freeb/1073741824.0, b/1073741824.0, b/G.exp_bytes,
                dense_reserve/1073741824.0, stage_reserve/1048576.0);
    }
    G.slot=calloc((size_t)nl*ne,sizeof(QSlot));
    G.is_x=malloc((size_t)QT_MAX_DEV*64*D*sizeof(float));
    G.resident_host=malloc((size_t)D*sizeof(float));
    G.qt_take_layer_us=calloc((size_t)nl,sizeof(double));
    G.qt_take_layer_max_us=calloc((size_t)nl,sizeof(double));
    G.qt_take_layer_calls=calloc((size_t)nl,sizeof(uint64_t));
    G.qt_issue_layer_routes=calloc((size_t)nl,sizeof(uint64_t));
    G.qt_issue_layer_gpu=calloc((size_t)nl,sizeof(uint64_t));
    G.timing_layer=-1;
    for(int i=0;i<64;i++) G.is_rows[i]=1;
    if(!G.slot||!G.is_x||!G.resident_host||!G.qt_take_layer_us||
       !G.qt_take_layer_max_us||!G.qt_take_layer_calls||
       !G.qt_issue_layer_routes||!G.qt_issue_layer_gpu) return 0;
    for(size_t i=0;i<(size_t)nl*ne;i++){
        G.slot[i].disk_replica=-1;
        G.slot[i].disk_controller=-1;
    }
    /* load learned heat (HEAT_FILE): warmstart order + initial values */
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"rb");
        if(f){
            uint32_t hdr[3]={0,0,0};
            if(fread(hdr,4,3,f)==3 && hdr[0]==0x51544831u && hdr[1]==(uint32_t)nl && hdr[2]==(uint32_t)ne){
                G.heat0=malloc((size_t)nl*ne*4);
                if(G.heat0 && fread(G.heat0,4,(size_t)nl*ne,f)==(size_t)nl*ne){
                    for(size_t i=0;i<(size_t)nl*ne;i++) G.slot[i].heat=G.heat0[i]>>1; /* decay */
                    fprintf(stderr,"[qtier] HEAT_FILE loaded: %s\n",hf);
                } else { free(G.heat0); G.heat0=NULL; }
            }
            fclose(f);
        }
    }
    pthread_mutex_init(&G.mx,NULL); pthread_cond_init(&G.cv,NULL); pthread_cond_init(&G.cv_take,NULL);
    if(G.stage_slots && !qt_stage_init()) {
        for(int i=0;i<G.ndev;i++) G.budget[i]+=stage_reserve;
        fprintf(stderr,"[qtier] transient GPU staging unavailable; disabled\n");
    }
    if(pthread_create(&G.th,NULL,uploader,NULL)!=0) return 0;
    G.on=1;
    fprintf(stderr,"[qtier] CUDA VRAM expert tier active: %d device(s), %.2f MB/expert\n",
            G.ndev, G.exp_bytes/1048576.0);
    return 1;
}

int qt_ready(void){ return G.on; }

int qt_dense_init(int n_layers){
    if(!G.on || n_layers!=G.nl) return 0;
    free(G.dense);
    G.dense=(QtDenseLayer*)calloc((size_t)n_layers,sizeof(*G.dense));
    if(!G.dense) return 0;
    G.dense_device=G.dev[0];
    G.dense_enabled=1;
    G.dense_registers=G.dense_calls=G.dense_fallbacks=0;
    return 1;
}

int qt_dense_register(int layer,int kind,const int8_t *weights,
                      const float *scales,int I,int O){
    if(!G.dense_enabled || !G.dense || layer<0 || layer>=G.nl ||
       !weights || !scales || I<1 || O<1) return 0;
    QtDenseLayer *d=&G.dense[layer];
    ColiCudaTensor **dst=NULL; int *out_o=NULL;
    if(kind==QT_DENSE_DN_QKV){ dst=&d->qkv; out_o=&d->qkv_o; }
    else if(kind==QT_DENSE_DN_Z){ dst=&d->z; out_o=&d->z_o; }
    else if(kind==QT_DENSE_DN_OUT){ dst=&d->out; out_o=&d->out_o; }
    else if(kind==QT_DENSE_ATTN_Q){ dst=&d->attn_q; out_o=&d->attn_q_o; }
    else if(kind==QT_DENSE_ATTN_K){ dst=&d->attn_k; out_o=&d->attn_k_o; }
    else if(kind==QT_DENSE_ATTN_V){ dst=&d->attn_v; out_o=&d->attn_v_o; }
    else if(kind==QT_DENSE_ATTN_O){ dst=&d->attn_o; out_o=&d->attn_o_o; d->attn_o_i=I; }
    else if(kind==QT_DENSE_SHARED_G){ dst=&d->shared_g; out_o=&d->shared_g_o; }
    else if(kind==QT_DENSE_SHARED_U){ dst=&d->shared_u; out_o=&d->shared_u_o; }
    else if(kind==QT_DENSE_SHARED_D){ dst=&d->shared_d; out_o=&d->shared_d_o; d->shared_d_i=I; }
    else return 0;
    if(!coli_cuda_tensor_upload(dst,weights,scales,1,I,O,G.dense_device)) return 0;
    *out_o=O;
    G.dense_registers++;
    return 1;
}

int qt_dense_deltanet_proj(int layer,const float *x,float *qkv,float *z){
    if(!G.dense_enabled || !G.dense || !x || !qkv || !z ||
       layer<0 || layer>=G.nl) return 0;
    QtDenseLayer *d=&G.dense[layer];
    if(!d->qkv || !d->z || d->qkv_o<1 || d->z_o<1) return 0;
    size_t need=(size_t)d->qkv_o+(size_t)d->z_o;
    if(need>G.dense_pack_cap){
        float *p=(float*)realloc(G.dense_pack,need*sizeof(float));
        if(!p) return 0;
        G.dense_pack=p; G.dense_pack_cap=need;
    }
    ColiCudaTensor *ts[2]={d->qkv,d->z};
    int off[2]={0,d->qkv_o};
    if(!coli_cuda_pipe_dense_batch(ts,off,2,G.D,x,G.dense_pack,
                                   (int)need,G.dense_device)){
        G.dense_fallbacks++;
        return 0;
    }
    memcpy(qkv,G.dense_pack,(size_t)d->qkv_o*sizeof(float));
    memcpy(z,G.dense_pack+(size_t)d->qkv_o,(size_t)d->z_o*sizeof(float));
    G.dense_calls++;
    return 1;
}

static float *dense_blob_upload(const float *src,size_t n,int device){
    float *d=(float*)coli_cuda_pipe_alloc(device,n*sizeof(float));
    if(!d) return NULL;
    if(!coli_cuda_pipe_upload(device,d,src,n*sizeof(float))){
        coli_cuda_pipe_free(device,d); return NULL;
    }
    return d;
}

int qt_dense_register_deltanet_full(int layer,
        const float *conv,const float *b,const float *a,
        const float *dtbias,const float *alog,const float *norm,
        const float *rec,const float *ring,
        int vheads,int kheads,int kdim,int vdim,int convk,int conv_dim,float eps){
    if(!G.dense_enabled || !G.dense || layer<0 || layer>=G.nl ||
       !conv||!b||!a||!dtbias||!alog||!norm||!rec||!ring ||
       vheads<1||kheads<1||vheads%kheads||kdim<1||vdim<1||convk<2||conv_dim<1)
        return 0;
    QtDenseLayer *d=&G.dense[layer];
    float *conv_d=dense_blob_upload(conv,(size_t)conv_dim*convk,G.dense_device);
    float *b_d=dense_blob_upload(b,(size_t)vheads*G.D,G.dense_device);
    float *a_d=dense_blob_upload(a,(size_t)vheads*G.D,G.dense_device);
    float *db_d=dense_blob_upload(dtbias,(size_t)vheads,G.dense_device);
    float *al_d=dense_blob_upload(alog,(size_t)vheads,G.dense_device);
    float *no_d=dense_blob_upload(norm,(size_t)vdim,G.dense_device);
    float *re_d=dense_blob_upload(rec,(size_t)vheads*kdim*vdim,G.dense_device);
    float *ri_d=dense_blob_upload(ring,(size_t)conv_dim*(convk-1),G.dense_device);
    if(!conv_d||!b_d||!a_d||!db_d||!al_d||!no_d||!re_d||!ri_d){
        float *p[]={conv_d,b_d,a_d,db_d,al_d,no_d,re_d,ri_d};
        for(size_t i=0;i<sizeof(p)/sizeof(p[0]);i++) if(p[i]) coli_cuda_pipe_free(G.dense_device,p[i]);
        return 0;
    }
    d->conv=conv_d; d->b=b_d; d->a=a_d; d->dtbias=db_d; d->alog=al_d;
    d->norm=no_d; d->rec=re_d; d->ring=ri_d; d->full=1;
    G.dense_vh=vheads; G.dense_vk=kheads; G.dense_kdim=kdim;
    G.dense_vdim=vdim; G.dense_convk=convk; G.dense_conv_dim=conv_dim;
    G.dense_eps=eps;
    return 1;
}

int qt_dense_deltanet_full(int layer,const float *x,float *out){
    if(!G.dense_enabled || !G.dense || layer<0 || layer>=G.nl || !x || !out)
        return 0;
    QtDenseLayer *d=&G.dense[layer];
    if(!d->full || !d->qkv || !d->z || !d->out) return 0;
    if(!coli_cuda_pipe_deltanet_layer(d->qkv,d->z,d->out,d->conv,d->b,d->a,
            d->dtbias,d->alog,d->norm,d->rec,d->ring,x,out,G.D,
            G.dense_vh,G.dense_vk,G.dense_kdim,G.dense_vdim,G.dense_convk,
            G.dense_conv_dim,G.dense_eps,G.dense_device)){
        G.dense_full_fallbacks++; return 0;
    }
    G.dense_full_calls++;
    return 1;
}

int qt_dense_deltanet_state(int layer,const float *qkv,const float *z,
                            const float *b,const float *a,float *norm_out){
    if(!G.dense_enabled || !G.dense || layer<0 || layer>=G.nl ||
       !qkv || !z || !b || !a || !norm_out) return 0;
    QtDenseLayer *d=&G.dense[layer];
    if(!d->full) return 0;
    if(!coli_cuda_pipe_deltanet_state(qkv,z,b,a,d->conv,d->dtbias,d->alog,
            d->norm,d->rec,d->ring,norm_out,G.dense_vh,G.dense_vk,
            G.dense_kdim,G.dense_vdim,G.dense_convk,G.dense_conv_dim,
            G.dense_eps,G.dense_device)){
        G.dense_full_fallbacks++;
        return 0;
    }
    G.dense_full_calls++;
    return 1;
}

int qt_dense_register_lm_head(const int8_t *weights,const float *scales,int I,int O){
    if(!G.dense_enabled || !weights || !scales || I<1 || O<1) return 0;
    if(G.dense_lm_head) coli_cuda_tensor_free(G.dense_lm_head);
    G.dense_lm_head=NULL;
    if(!coli_cuda_tensor_upload(&G.dense_lm_head,weights,scales,1,I,O,G.dense_device)) return 0;
    G.dense_lm_I=I; G.dense_lm_O=O;
    return 1;
}

int qt_dense_lm_head(const float *x,float *logits,int O){
    if(!G.dense_enabled || !G.dense_lm_head || !x || !logits ||
       O!=G.dense_lm_O || G.dense_lm_I!=G.D) return 0;
    ColiCudaTensor *t[1]={G.dense_lm_head}; int off[1]={0};
    if(!coli_cuda_pipe_dense_batch(t,off,1,G.D,x,logits,O,G.dense_device)){
        G.dense_lm_fallbacks++; return 0;
    }
    G.dense_lm_calls++;
    return 1;
}

int qt_dense_attention_proj(int layer,const float *x,float *q,float *k,float *v){
    if(!G.dense_enabled || !G.dense || layer<0 || layer>=G.nl || !x || !q || !k || !v)
        return 0;
    QtDenseLayer *d=&G.dense[layer];
    if(!d->attn_q || !d->attn_k || !d->attn_v) return 0;
    ColiCudaTensor *ts[3]={d->attn_q,d->attn_k,d->attn_v};
    int off[3]={0,d->attn_q_o,d->attn_q_o+d->attn_k_o};
    int total=d->attn_q_o+d->attn_k_o+d->attn_v_o;
    if((size_t)total>G.dense_pack_cap){
        float *p=(float*)realloc(G.dense_pack,(size_t)total*sizeof(float));
        if(!p) return 0;
        G.dense_pack=p; G.dense_pack_cap=(size_t)total;
    }
    if(!coli_cuda_pipe_dense_batch(ts,off,3,G.D,x,G.dense_pack,
                                   total,G.dense_device)){
        G.dense_attn_fallbacks++; return 0;
    }
    memcpy(q,G.dense_pack,(size_t)d->attn_q_o*sizeof(float));
    memcpy(k,G.dense_pack+(size_t)d->attn_q_o,(size_t)d->attn_k_o*sizeof(float));
    memcpy(v,G.dense_pack+(size_t)d->attn_q_o+d->attn_k_o,
           (size_t)d->attn_v_o*sizeof(float));
    G.dense_attn_calls++;
    return 1;
}

int qt_dense_attention_out(int layer,const float *x,float *out){
    if(!G.dense_enabled || !G.dense || layer<0 || layer>=G.nl || !x || !out)
        return 0;
    QtDenseLayer *d=&G.dense[layer];
    if(!d->attn_o || d->attn_o_i<1 || d->attn_o_o<1) return 0;
    ColiCudaTensor *ts[1]={d->attn_o}; int off[1]={0};
    if(!coli_cuda_pipe_dense_batch(ts,off,1,d->attn_o_i,x,out,
                                   d->attn_o_o,G.dense_device)){
        G.dense_attn_fallbacks++; return 0;
    }
    G.dense_attn_calls++;
    return 1;
}

int qt_dense_shared(int layer,const float *x,float *out){
    if(!G.dense_enabled || !G.dense || layer<0 || layer>=G.nl || !x || !out)
        return 0;
    QtDenseLayer *d=&G.dense[layer];
    if(!d->shared_g || !d->shared_u || !d->shared_d ||
       d->shared_g_o<1 || d->shared_g_o!=d->shared_u_o || d->shared_d_i!=d->shared_g_o)
        return 0;
    int I=d->shared_g_o, total=2*I;
    const char *coarse=getenv("COLI_DENSE_SHARED_COARSE");
    if(coarse && *coarse=='1'){
        if(coli_cuda_pipe_dense_mlp(d->shared_g,d->shared_u,d->shared_d,x,out,
                                    G.D,I,d->shared_d_o,G.dense_device)){
            G.dense_shared_coarse_calls++;
            return 1;
        }
        G.dense_shared_coarse_fallbacks++;
    }
    if((size_t)total>G.dense_pack_cap){
        float *p=(float*)realloc(G.dense_pack,(size_t)total*sizeof(float));
        if(!p) return 0;
        G.dense_pack=p; G.dense_pack_cap=(size_t)total;
    }
    ColiCudaTensor *ts[2]={d->shared_g,d->shared_u}; int off[2]={0,I};
    if(!coli_cuda_pipe_dense_batch(ts,off,2,G.D,x,G.dense_pack,total,G.dense_device)){
        G.dense_shared_fallbacks++; return 0;
    }
    for(int i=0;i<I;i++){
        float sv=G.dense_pack[i];
        G.dense_pack[i]=(sv/(1.f+expf(-sv)))*G.dense_pack[I+i];
    }
    ColiCudaTensor *down[1]={d->shared_d}; int down_off[1]={0};
    if(!coli_cuda_pipe_dense_batch(down,down_off,1,I,G.dense_pack,out,
                                   d->shared_d_o,G.dense_device)){
        G.dense_shared_fallbacks++; return 0;
    }
    G.dense_shared_calls++;
    return 1;
}

void qt_dense_stats(void){
    if(!G.dense_enabled || !G.dense) return;
    uint64_t qkv=0,z=0,out=0;
    for(int i=0;i<G.nl;i++){ qkv+=G.dense[i].qkv!=NULL; z+=G.dense[i].z!=NULL; out+=G.dense[i].out!=NULL; }
    uint64_t full=0;
    for(int i=0;i<G.nl;i++) full+=G.dense[i].full!=0;
    uint64_t aq=0,ak=0,av=0,ao=0,sg=0,su=0,sd=0;
    for(int i=0;i<G.nl;i++){ aq+=G.dense[i].attn_q!=NULL; ak+=G.dense[i].attn_k!=NULL;
        av+=G.dense[i].attn_v!=NULL; ao+=G.dense[i].attn_o!=NULL;
        sg+=G.dense[i].shared_g!=NULL; su+=G.dense[i].shared_u!=NULL; sd+=G.dense[i].shared_d!=NULL; }
    fprintf(stderr,"[qtier-dense] device %d | registered qkv=%llu z=%llu out=%llu full=%llu lm=%d attn=%llu/%llu/%llu/%llu shared=%llu/%llu/%llu | projection batches %llu failures %llu | full calls %llu failures %llu | lm calls %llu failures %llu | attn calls %llu failures %llu shared calls %llu failures %llu coarse shared %llu failures %llu\n",
            G.dense_device,(unsigned long long)qkv,(unsigned long long)z,
            (unsigned long long)out,(unsigned long long)full,G.dense_lm_head!=NULL,
            (unsigned long long)aq,(unsigned long long)ak,(unsigned long long)av,(unsigned long long)ao,
            (unsigned long long)sg,(unsigned long long)su,(unsigned long long)sd,
            (unsigned long long)G.dense_calls,(unsigned long long)G.dense_fallbacks,
            (unsigned long long)G.dense_full_calls,(unsigned long long)G.dense_full_fallbacks,
            (unsigned long long)G.dense_lm_calls,(unsigned long long)G.dense_lm_fallbacks,
            (unsigned long long)G.dense_attn_calls,(unsigned long long)G.dense_attn_fallbacks,
            (unsigned long long)G.dense_shared_calls,(unsigned long long)G.dense_shared_fallbacks,
            (unsigned long long)G.dense_shared_coarse_calls,(unsigned long long)G.dense_shared_coarse_fallbacks);
}

/* Is (layer,eid) currently VRAM-resident? (used to free RAM-side int8 copies) */
int qt_is_resident(int layer,int eid){
    if(!G.on) return 0;
    pthread_mutex_lock(&G.mx);
    int r = qs(layer,eid)->resident;
    pthread_mutex_unlock(&G.mx);
    return r;
}

int qt_expert_location(int layer, int eid, QtExpertLocation *out){
    if(!G.on || !out || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return 0;
    memset(out, 0, sizeof(*out));
    out->home_gpu = -1;
    out->resident_gpu = -1;
    out->ram_node = -1;
    out->disk_replica = -1;
    out->disk_controller = -1;
    int hi = home(eid);
    if(hi>=0 && hi<G.topology.gpu_count){
        const QtGpuTopology *g=&G.topology.gpu[hi];
        out->home_gpu=g->ordinal;
        out->ram_node=G.topology.numa_enabled ? -2 : g->numa_node;
        if(out->ram_node<0) out->ram_node=0;
    }
    if(G.topology.storage_count>0)
        out->disk_controller=-1;
    pthread_mutex_lock(&G.mx);
    QSlot *s=qs(layer,eid);
    out->resident=s->resident;
    out->queued=s->queued;
    out->disk_replica=s->disk_replica;
    out->disk_controller=s->disk_controller;
    if(s->resident && hi>=0 && hi<G.ndev) out->resident_gpu=G.dev[hi];
    pthread_mutex_unlock(&G.mx);
    return 1;
}

/* internal, G.mx held: enqueue one upload. victim=-1: plain upload (budget is
 * reserved here); victim>=0: LFRU swap (budget neutral). */
static int enqueue_locked(int layer,int eid,int v_layer,int v_eid,int reserved){
    QSlot *s=qs(layer,eid);
    if(s->resident||s->queued||!s->g4) return 0;
    if(G.qn>=QT_QCAP){ G.q_full_skips++; return 0; }
    int hd=home(eid);
    if(!reserved && v_eid<0 && G.used[hd]+G.exp_bytes>G.budget[hd]) return 0;
    size_t mb=(size_t)G.D*G.Ih/(G.wfmt==1?1:2);   /* buffer di staging: int8 = 1 byte/elemento */
    uint8_t *w=malloc(3*mb); float *sc=malloc((2*G.sc_gu+G.sc_d)*sizeof(float));
    if(!w||!sc){ free(w); free(sc); return 0; }
    if(!reserved && v_eid<0) G.used[hd]+=G.exp_bytes;
    s->queued=1;
    stage(w,sc,s->g4,s->u4,s->d4,s->gs,s->us,s->ds);
    G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid; G.q[G.qt_].w=w; G.q[G.qt_].s=sc;
    G.q[G.qt_].v_layer=v_layer; G.q[G.qt_].v_eid=v_eid;
    G.qt_=(G.qt_+1)%QT_QCAP; G.qn++;
    pthread_cond_signal(&G.cv);
    return 1;
}

void qt_note(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    if(s->heat<0xFFFFFFFFu) s->heat++;
    enqueue_locked(layer,eid,-1,-1,0);
    pthread_mutex_unlock(&G.mx);
}

/* blocking variant for the warmstart (waits for queue space). */
void qt_note_block(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    enqueue_locked(layer,eid,-1,-1,0);
    pthread_mutex_unlock(&G.mx);
}

/* warmstart order -- heat descending (HEAT_FILE) or natural order.
 * Returns 0 once all budgets are full or the list is exhausted. */
static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    if (ha != hb) return ha<hb ? 1 : -1;
    /* qsort is otherwise free to permute equal-heat entries.  Stable
     * tie-breaking matters for reproducible placement files and makes the
     * balanced policy deterministic across CRT/libc implementations. */
    int ia=*(const int*)a, ib=*(const int*)b;
    return ia>ib ? 1 : ia<ib ? -1 : 0;
}

/* Build the warmstart order once.  The historical policy is a single global
 * heat sort (or layer-major order without HEAT_FILE).  The explicit balanced
 * policy keeps the same heat objective inside each layer but interleaves the
 * layer lists, so a finite budget cannot accidentally consume the first N
 * layers in full.  Because expert home is eid % gpu_count, the same order is
 * also naturally spread across future GPU islands. */
static void fill_order_init_locked(size_t n) {
    if (G.fill_order) return;
    G.fill_order=malloc(n*sizeof(int));
    if (!G.fill_order) return;
    const char *placement=getenv("QTIER_PLACEMENT");
    int balanced=placement && strcmp(placement,"layer_balanced")==0;
    if (!balanced) {
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
        G.fill_cur=0;
        return;
    }
    int *by_layer=(int*)malloc(n*sizeof(int));
    if (!by_layer) {
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        G.fill_cur=0;
        return;
    }
    for(int l=0;l<G.nl;l++) {
        int *list=by_layer+(size_t)l*G.ne;
        for(int e=0;e<G.ne;e++) list[e]=l*G.ne+e;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(list,G.ne,sizeof(int),cmp_heat_desc); }
    }
    size_t out=0;
    for(int rank=0;rank<G.ne;rank++)
        for(int l=0;l<G.nl;l++)
            G.fill_order[out++]=by_layer[(size_t)l*G.ne+rank];
    free(by_layer);
    G.fill_cur=0;
}
int qt_fill_next(int *layer,int *eid){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    pthread_mutex_lock(&G.mx);
    fill_order_init_locked(n);
    if(!G.fill_order){ pthread_mutex_unlock(&G.mx); return 0; }
    while((size_t)G.fill_cur<n){
        int gi=G.fill_order[G.fill_cur];
        int l=gi/G.ne, e=gi%G.ne, hd=home(e);
        QSlot *s=qs(l,e);
        int full=1; for(int i=0;i<G.ndev;i++) if(G.used[i]+G.exp_bytes<=G.budget[i]) full=0;
        if(full){ pthread_mutex_unlock(&G.mx); return 0; }
        G.fill_cur++;
        if(s->resident||s->queued) continue;
        if(G.used[hd]+G.exp_bytes>G.budget[hd]) continue;   /* dieses Device voll */
        *layer=l; *eid=e;
        pthread_mutex_unlock(&G.mx);
        return 1;
    }
    pthread_mutex_unlock(&G.mx);
    return 0;
}

/* Plan the whole warmstart set in one pass -- same heat order and budget
 * reservation as qt_fill_next, but without loading. The experts are then
 * loaded by any number of threads and handed over via qt_note_planned. */
int qt_plan_fill(int *layers,int *eids,int max){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    int cnt=0;
    pthread_mutex_lock(&G.mx);
    fill_order_init_locked(n);
    if(!G.fill_order){ pthread_mutex_unlock(&G.mx); return 0; }
    while((size_t)G.fill_cur<n && cnt<max){
        int full=1; for(int i=0;i<G.ndev;i++) if(G.used[i]+G.exp_bytes<=G.budget[i]) full=0;
        if(full) break;
        int gi=G.fill_order[G.fill_cur++];
        int l=gi/G.ne, e=gi%G.ne, hd=home(e);
        QSlot *s=qs(l,e);
        if(s->resident||s->queued||s->planned) continue;
        if(G.used[hd]+G.exp_bytes>G.budget[hd]) continue;
        G.used[hd]+=G.exp_bytes;          /* reserve */
        s->planned=1;
        layers[cnt]=l; eids[cnt]=e; cnt++;
    }
    pthread_mutex_unlock(&G.mx);
    return cnt;
}

/* Thread-safe (callable from multiple loader threads): stage + enqueue one
 * expert reserved by qt_plan_fill; blocks only while the queue is full. */
/* g/u/d: i pesi COME STANNO IN RAM -- int4 impacchettati su un container gs64,
 * int8 su un container int8. Il formato lo decide qt_init dal container, e da
 * li' in poi staging e upload lo seguono. */
void qt_note_planned(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    if(!enqueue_locked(layer,eid,-1,-1,1)){
        /* not enqueueable (e.g. already resident): return the reservation */
        if(s->planned) G.used[home(eid)]-=G.exp_bytes;
    }
    s->planned=0;
    pthread_mutex_unlock(&G.mx);
}

/* waits until the upload queue is drained (end of warmstart). */
void qt_fill_wait(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    while(G.qn>0 && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
}

/* Adaptive swap check (every 16 ticks = tokens): per device, coldest resident
 * vs hottest non-resident. Decay every 1024 ticks so an old workload cannot
 * permanently own the tier; admission uses the shared tier.h contract. */
static void qt_lfru_tick_locked(void){
    size_t n=(size_t)G.nl*G.ne;
    G.tick++;
    if(!(G.tick%1024))
        for(size_t i=0;i<n;i++) G.slot[i].heat=tier_decay_value(G.slot[i].heat);
    if(G.tick%16) return;
    for(int di=0;di<G.ndev;di++){
        int cold=-1, hot=-1; uint32_t ch=0, hh=0;
        for(size_t i=0;i<n;i++){
            QSlot *s=&G.slot[i];
            int e=(int)(i%G.ne);
            if(home(e)!=di) continue;
            if(s->resident && !s->queued){ if(cold<0||s->heat<ch){ cold=(int)i; ch=s->heat; } }
            else if(!s->resident && !s->queued && s->g4){ if(hot<0||s->heat>hh){ hot=(int)i; hh=s->heat; } }
        }
        if(cold<0||hot<0) continue;
        if(!tier_should_promote(hh,ch)) continue;
        QSlot *v=&G.slot[cold];
        v->resident=0;                                    /* CPU fallback from now on */
        if(enqueue_locked(hot/G.ne,hot%G.ne,cold/G.ne,cold%G.ne,0)) G.swaps++;
        else v->resident=1;                               /* queue full: revert */
    }
}

uint32_t qt_issue(int layer,const int *eids,int K,const float *weights,const float *x){
    if(!G.on||K>32) return 0;
    if(!weights) return 0;
    G.exec_descriptors++;
    G.timing_layer=layer;
    const int timing_on = getenv("QTIER_TIMING_DUMP") && atoi(getenv("QTIER_TIMING_DUMP"));
    struct timespec ts0, ts1;
    double t_issue_call_us = 0.0;
    if (timing_on) clock_gettime(CLOCK_MONOTONIC, &ts0);
    uint32_t mask=0;
    ColiCudaTensor *tg[QT_MAX_DEV][32],*tu[QT_MAX_DEV][32],*td[QT_MAX_DEV][32];
    static int rows[32]={0};
    if(!rows[0]) for(int i=0;i<32;i++) rows[i]=1;
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    pthread_mutex_lock(&G.mx);
    const char *lfru_env=getenv("QTIER_LFRU");
    if(layer==0 && !(lfru_env && atoi(lfru_env)==0)) qt_lfru_tick_locked();
    G.issue_open=1;
    for(int k=0;k<K;k++){
        QSlot *s=qs(layer,eids[k]);
        if(s->resident){
            int di=home(eids[k]); int c=G.is_cnt[di];
            tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
            G.is_k[di][c]=k; G.is_cnt[di]=c+1;
            mask|=1u<<k; G.hits[di]++;
        } else G.miss++;
    }
    pthread_mutex_unlock(&G.mx);

    /* Save the device descriptor set before the resident fast path.  If a
     * device-side reduction ever fails, qt_take can replay these exact groups
     * through the proven host-staged launcher. */
    for(int di=0;di<G.ndev;di++) if(G.is_cnt[di]){
        float *xr=G.is_x+(size_t)di*64*G.D;
        for(int j=0;j<G.is_cnt[di];j++)
            memcpy(xr+(size_t)j*G.D,x,(size_t)G.D*sizeof(float));
        for(int j=0;j<G.is_cnt[di];j++){
            G.is_tg[di][j]=tg[di][j]; G.is_tu[di][j]=tu[di][j]; G.is_td[di][j]=td[di][j];
        }
    }

    /* Optional execution-only staging: let the current layer's nonresident
     * routes join the same DP4A island batch without admitting them to the
     * residency/LFRU table.  This is deliberately opt-in because it changes
     * the CPU/GPU work split; the normal path leaves every miss on CPU.
     *
     * The stage arena is an execution cache, not a second residency table.
     * A slot is keyed by (layer,eid) and its device weights remain valid until
     * that slot is reused.  qt_take completes the previous group before the
     * next qt_issue, so an inactive slot is safe to refresh. */
    uint8_t stage_active[QT_MAX_DEV][QT_STAGE_MAX];
    memset(stage_active,0,sizeof stage_active);
    int stage_new[QT_MAX_DEV]={0};
    int stage_slot_for_route[64];
    uint8_t stage_route_new[64]={0};
    int stage_batch_ok[QT_MAX_DEV];
    int stage_slots_new[QT_MAX_DEV][64];
    ColiCudaTensor *stage_batch_g[QT_MAX_DEV][64];
    ColiCudaTensor *stage_batch_u[QT_MAX_DEV][64];
    ColiCudaTensor *stage_batch_d[QT_MAX_DEV][64];
    const void *stage_batch_w[QT_MAX_DEV][64];
    const float *stage_batch_s[QT_MAX_DEV][64];
    for(int k=0;k<64;k++) stage_slot_for_route[k]=-1;
    for(int di=0;di<QT_MAX_DEV;di++) stage_batch_ok[di]=1;
    int stage_limit=G.stage_slots;
    const char *stage_limit_env=getenv("COLI_CUDA_STAGE_LIMIT");
    if(stage_limit_env && atoi(stage_limit_env)>=0 && atoi(stage_limit_env)<stage_limit)
        stage_limit=atoi(stage_limit_env);
    if(G.stage_slots>0 && G.egs) for(int k=0;k<K;k++) if(!(mask&(1u<<k))) {
        int eid=eids[k], di=home(eid);
        if(di<0 || di>=G.ndev || G.stage_slots<=0) continue;
        if(G.is_cnt[di]>=64) {
            if(getenv("COLI_STAGE_BATCH_DEBUG"))
                fprintf(stderr,"[qtier-stage-debug] descriptor full layer=%d route=%d di=%d count=%d K=%d\n",
                        layer,k,di,G.is_cnt[di],K);
            G.stage_failures++; continue;
        }
        const uint8_t *g4=NULL,*u4=NULL,*d4=NULL;
        const float *gs=NULL,*us=NULL,*ds=NULL;
        pthread_mutex_lock(&G.mx);
        QSlot *s=qs(layer,eid);
        g4=s->g4; u4=s->u4; d4=s->d4; gs=s->gs; us=s->us; ds=s->ds;
        pthread_mutex_unlock(&G.mx);
        if(!g4 || !u4 || !d4 || !gs || !us || !ds) continue;

        int si=-1;
        /* First prefer a matching cached expert.  stage_active also handles
         * duplicate route entries in a malformed/experimental route list. */
        for(int j=0;j<G.stage_slots;j++)
            if(G.stage_layer[di][j]==layer && G.stage_eid[di][j]==eid){
                si=j;
                if(!stage_active[di][j]){
                    stage_active[di][j]=1;
                    G.stage_clock++;
                    G.stage_lru[di][j]=G.stage_clock;
                    G.stage_cache_hits++;
                }
                break;
            }
        if(si<0){
            /* A new upload is the only operation subject to the per-call
             * limit.  Cache hits remain usable even after the limit is met. */
            if(stage_new[di]>=stage_limit) continue;
            uint64_t oldest=UINT64_MAX;
            for(int j=0;j<G.stage_slots;j++) if(!stage_active[di][j]){
                if(G.stage_layer[di][j]<0){ si=j; break; }
                if(G.stage_lru[di][j]<oldest){ oldest=G.stage_lru[di][j]; si=j; }
            }
            if(si<0) continue; /* all cache entries are used by this layer */
            /* Preserve the established staging representation: stage() makes
             * two's-complement nibbles offset-binary, then the CUDA refresh
             * converts them back to signed nibbles on device.  The batch
             * backend changes only the launch boundary, not this contract. */
            stage(G.stage_w[di][si],G.stage_s[di][si],g4,u4,d4,gs,us,ds);
            stage_active[di][si]=1; /* reserve this slot for this issue */
            int n=stage_new[di]++;
            stage_slots_new[di][n]=si;
            stage_batch_g[di][n]=G.stage_g[di][si];
            stage_batch_u[di][n]=G.stage_u[di][si];
            stage_batch_d[di][n]=G.stage_d[di][si];
            stage_batch_w[di][n]=G.stage_w[di][si];
            stage_batch_s[di][n]=G.stage_s[di][si];
            stage_slot_for_route[k]=si;
            stage_route_new[k]=1;
            continue;
        }
        stage_slot_for_route[k]=si;
    }

    /* Refresh all new execution-cache entries on a device as one host
     * operation. The destination tensors are still independent cache slots,
     * but the conversion boundary is now one launch per island batch. */
    for(int di=0;di<G.ndev;di++) if(stage_new[di]){
        int batch_ok=coli_cuda_expert_update_batch_async(
                stage_batch_g[di],stage_batch_u[di],stage_batch_d[di],
                stage_batch_w[di],stage_batch_s[di],stage_new[di]);
        if(!batch_ok){
            if(getenv("COLI_STAGE_BATCH_DEBUG"))
                fprintf(stderr,"[qtier-stage-debug] batch rejected di=%d count=%d g0=%p u0=%p d0=%p w0=%p s0=%p\n",
                        di,stage_new[di],(void*)stage_batch_g[di][0],
                        (void*)stage_batch_u[di][0],(void*)stage_batch_d[di][0],
                        stage_batch_w[di][0],(void*)stage_batch_s[di][0]);
            stage_batch_ok[di]=0;
            G.stage_failures+=(uint64_t)stage_new[di];
            for(int n=0;n<stage_new[di];n++){
                int si=stage_slots_new[di][n];
                G.stage_layer[di][si]=-1;
                G.stage_eid[di][si]=-1;
                G.stage_lru[di][si]=0;
            }
        }
    }
    /* Publish the key for each successful new slot. The route arrays are
     * revisited rather than carrying another parallel key array. */
    for(int k=0;k<K;k++) if(stage_route_new[k]){
        int di=home(eids[k]), si=stage_slot_for_route[k];
        if(di<0 || di>=G.ndev || si<0 || !stage_batch_ok[di]) continue;
        G.stage_layer[di][si]=layer;
        G.stage_eid[di][si]=eids[k];
        G.stage_clock++;
        G.stage_lru[di][si]=G.stage_clock;
        G.stage_updates++;
    }
    /* Append cache hits and successfully refreshed entries to the island
     * descriptor after all stream-ordered refreshes have been submitted. */
    for(int k=0;k<K;k++) if(stage_slot_for_route[k]>=0){
        int di=home(eids[k]), si=stage_slot_for_route[k];
        if(di<0 || di>=G.ndev || (stage_route_new[k] && !stage_batch_ok[di])) continue;
        int c=G.is_cnt[di];
        if(c>=64){ G.stage_failures++; continue; }
        G.is_tg[di][c]=G.stage_g[di][si];
        G.is_tu[di][c]=G.stage_u[di][si];
        G.is_td[di][c]=G.stage_d[di][si];
        G.is_k[di][c]=k; G.is_cnt[di]=c+1;
        if(!stage_route_new[k]) stage_active[di][si]=1;
        else stage_active[di][si]=1;
        mask|=1u<<k; G.stage_calls++;
    }

    /* Stage 3A resident island: make one host->hub transfer, then let each
     * owning device consume the hub row through the resident CUDA primitive.
     * The primitive folds router weights before returning a single partial, so
     * qt_take() has no per-expert device->host traffic to perform. */
    G.resident_active=0;
    if(getenv("COLI_CUDA_RESIDENT") && atoi(getenv("COLI_CUDA_RESIDENT")) && G.ndev>0){
        int active_idx[QT_MAX_DEV], nactive=0;
        for(int di=0;di<G.ndev;di++) if(G.is_cnt[di]) active_idx[nactive++]=di;
        int hub_pos=qt_topology_pick_hub(&G.topology,active_idx,nactive);
        int hub=hub_pos>=0?G.dev[hub_pos]:G.dev[0], n=0, ok=1;
        size_t bytes=(size_t)(G.ndev+2)*G.D*sizeof(float);
        float *buf=coli_cuda_pipe_scratch(hub,25,bytes);
        double th0=qt_now_us();
        int uploaded=buf && G.resident_host &&
                     coli_cuda_pipe_upload(hub,buf,x,(size_t)G.D*sizeof(float));
        if(uploaded) G.resident_h2d_us+=qt_now_us()-th0;
        if(uploaded){
            G.resident_buf=buf;
            G.resident_x_dev=buf;
            G.resident_slots_dev=buf+G.D;
            G.resident_acc_dev=buf+(size_t)(G.ndev+1)*G.D;
            QtLayerWork layer_work;
            memset(&layer_work,0,sizeof layer_work);
            layer_work.layer=layer;
            layer_work.home_device=hub;
            for(int di=0;di<G.ndev;di++) if(G.is_cnt[di]){
                if (layer_work.count >= QT_MAX_DEV) { ok=0; break; }
                QtIslandWork *work=&layer_work.islands[layer_work.count++];
                memset(work,0,sizeof *work);
                work->layer=layer;
                work->device=G.dev[di];
                work->count=G.is_cnt[di];
                work->hidden=G.D;
                work->intermediate=G.Ih;
                work->input_dev=G.resident_x_dev;
                work->route_weights=weights;
                for(int q=0;q<work->count;q++){
                    work->route_index[q]=G.is_k[di][q];
                    work->gate[q]=G.is_tg[di][q];
                    work->up[q]=G.is_tu[di][q];
                    work->down[q]=G.is_td[di][q];
                }
            }
            if (ok && layer_work.count > 0) {
                double resident_issue_start=qt_now_us();
                ok=qt_execute_gpu_layer(&layer_work,G.resident_slots_dev,
                                        G.resident_devices,&n,timing_on,
                                        &t_issue_call_us);
                G.resident_issue_us+=qt_now_us()-resident_issue_start;
            }
            if(ok && n>0){
                G.resident_home=hub; G.resident_nissued=n; G.resident_active=1;
                /* With one active compute island, the reduction is an
                 * identity operation. Alias the accumulator to the sole
                 * partial slot so resident_take can wait for completion and
                 * download it without launching sum_slots. Keep the switch
                 * explicit for matched A/B runs and leave multi-island
                 * reduction semantics unchanged. */
                const char *direct_env=getenv("COLI_CUDA_DIRECT_SINGLE");
                /* The alias is currently proven only for one-row decode.
                 * Keep multi-row/prefill on the established accumulator path
                 * until its reduction ordering has a dedicated parity test. */
                if(n==1 && G.resident_timing_decode &&
                   direct_env && atoi(direct_env)) {
                    G.resident_acc_dev=G.resident_slots_dev;
                    G.resident_single_direct++;
                } else {
                    G.resident_acc_dev=buf+(size_t)(G.ndev+1)*G.D;
                }
                G.resident_calls++;
                for(int di=0;di<G.ndev;di++) G.resident_experts+=(uint64_t)G.is_cnt[di];
                if (timing_on) {
                    clock_gettime(CLOCK_MONOTONIC, &ts1);
                    G.timing_pre_us = (ts1.tv_sec-ts0.tv_sec)*1e6 +
                                      (ts1.tv_nsec-ts0.tv_nsec)/1e3;
                    G.timing_issue_us = t_issue_call_us;
                }
                qt_record_issue_diag(layer,K,mask);
                return mask;
            }
            /* A launch failure is a performance fallback, never a reason to
             * drop an expert. Drain any groups already queued before retrying
             * them through the established host-staged path. */
            if(n>0){
                (void)coli_cuda_expert_group_resident_take(hub,G.resident_devices,n,
                                                           G.resident_slots_dev,G.resident_acc_dev,G.D);
                (void)coli_cuda_pipe_sync(hub);
            }
            G.resident_fallbacks++;
        } else if(buf || G.resident_host){
            G.resident_fallbacks++;
        }
    }

    /* Host-side timing instrumentation (QTIER_TIMING_DUMP). */
    double t_pre_loop_end = 0;
    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x + (size_t)di*64*G.D;              /* per-device input block */
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        struct timespec ti0, ti1;
        if (timing_on) clock_gettime(CLOCK_MONOTONIC, &ti0);
        if(!coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr)){
            /* issue failed -> hand these k back to the CPU */
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
        if (timing_on) { clock_gettime(CLOCK_MONOTONIC, &ti1); t_issue_call_us += (ti1.tv_sec-ti0.tv_sec)*1e6 + (ti1.tv_nsec-ti0.tv_nsec)/1e3; }
    }
    if (timing_on) { clock_gettime(CLOCK_MONOTONIC, &ts1); t_pre_loop_end = (ts1.tv_sec-ts0.tv_sec)*1e6 + (ts1.tv_nsec-ts0.tv_nsec)/1e3; }
    G.timing_pre_us = t_pre_loop_end;
    G.timing_issue_us = t_issue_call_us;
    qt_record_issue_diag(layer,K,mask);
    return mask;
}

void qt_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!G.on) return;
    const int timing_on = getenv("QTIER_TIMING_DUMP") && atoi(getenv("QTIER_TIMING_DUMP"));
    const int mprof_on = getenv("COLI_TIMERS") && atoi(getenv("COLI_TIMERS"));
    const int island_timing = getenv("COLI_ISLAND_TIMING") &&
                              atoi(getenv("COLI_ISLAND_TIMING"));
    const int resident_diag = (mprof_on || island_timing) && G.resident_timing_decode;
    double diag_take0 = resident_diag ? qt_now_us() : 0.0;
    if (resident_diag) {
        G.qt_last_gpu_event_us=0.0;
        G.qt_last_sync_wait_us=0.0;
        G.qt_last_take_wall_us=0.0;
        G.qt_last_gpu_complete_ms=0.0;
        G.qt_last_merge_begin_ms=0.0;
        G.qt_last_gpu_valid=0;
    }
    struct timespec ts0, ts1;
    double t_take_call_us = 0.0;
    if (timing_on) clock_gettime(CLOCK_MONOTONIC, &ts0);

    if(G.resident_active){
        double take0=qt_now_us();
        int ok=coli_cuda_expert_group_resident_take(G.resident_home,G.resident_devices,
                                                    G.resident_nissued,G.resident_slots_dev,
                                                    G.resident_acc_dev,G.D);
        double take_api_us=qt_now_us()-take0;
        G.resident_reduce_us+=take_api_us; /* legacy aggregate retained for qt_stats */
        if (resident_diag) G.resident_take_api_us+=take_api_us;
        double wait0=qt_now_us();
        if (ok && resident_diag) ok=coli_cuda_expert_group_resident_sync(G.resident_home);
        if (resident_diag) G.resident_sync_wait_us+=qt_now_us()-wait0;
        if (resident_diag) G.qt_last_sync_wait_us=qt_now_us()-wait0;
        if (resident_diag && ok) {
            G.qt_last_gpu_complete_ms=qt_now_us()/1000.0;
            G.qt_last_merge_begin_ms=G.qt_last_gpu_complete_ms;
        }
        if (ok && resident_diag) {
            double gpu_ms=0.0, reduce_ms=0.0;
            if (coli_cuda_expert_group_resident_timing(
                    G.resident_home,G.resident_devices,G.resident_nissued,
                    &gpu_ms,&reduce_ms)) {
                G.resident_gpu_event_us+=gpu_ms*1000.0;
                G.resident_reduce_event_us+=reduce_ms*1000.0;
                G.qt_last_gpu_event_us=gpu_ms*1000.0;
                G.qt_last_gpu_valid=1;
            }
            uint64_t gpu_lower_ns=0, gpu_upper_ns=0;
            uint64_t reduce_lower_ns=0, reduce_upper_ns=0;
            if (coli_cuda_expert_group_resident_host_timing(
                    G.resident_home,G.resident_devices,G.resident_nissued,
                    &gpu_lower_ns,&gpu_upper_ns,
                    &reduce_lower_ns,&reduce_upper_ns)) {
                G.qt_last_gpu_lower_ms=(double)gpu_lower_ns/1000000.0;
                G.qt_last_gpu_upper_ms=(double)gpu_upper_ns/1000000.0;
                G.qt_last_reduce_lower_ms=(double)reduce_lower_ns/1000000.0;
                G.qt_last_reduce_upper_ms=(double)reduce_upper_ns/1000000.0;
            }
        }
        double td0=qt_now_us();
        if (resident_diag && ok) G.qt_last_merge_begin_ms=td0/1000.0;
        if(ok) ok=coli_cuda_pipe_download(G.resident_home,G.resident_acc_dev,
                                          G.resident_host,(size_t)G.D*sizeof(float));
        if(ok) {
            double d2h_us=qt_now_us()-td0;
            G.resident_d2h_us+=d2h_us;
            if (resident_diag) G.resident_d2h_diag_us+=d2h_us;
        }
        if(ok){
            for(int d=0;d<G.D;d++) out[d]+=G.resident_host[d];
            if (resident_diag) {
                G.resident_take_total_us+=qt_now_us()-take0;
                G.resident_timing_calls++;
            }
            G.resident_active=0;
            pthread_mutex_lock(&G.mx);
            G.issue_open=0;
            pthread_cond_broadcast(&G.cv_take);
            pthread_mutex_unlock(&G.mx);
            if (timing_on) {
                clock_gettime(CLOCK_MONOTONIC, &ts1);
                double total_us=(ts1.tv_sec-ts0.tv_sec)*1e6+(ts1.tv_nsec-ts0.tv_nsec)/1e3;
                const char *path=getenv("QTIER_TIMING_FILE");
                FILE *fp=path?fopen(path,"a"):stderr;
                if(fp){
                    fprintf(fp,"tier_pre_us=%.2f tier_issue_us=%.2f tier_take_us=%.2f tier_total_us=%.2f resident=1\n",
                            G.timing_pre_us,G.timing_issue_us,total_us,total_us);
                    if(path) fclose(fp);
                }
            }
            if (resident_diag) {
                /* Include the final mutex/bookkeeping and optional timing
                 * dump in the caller-visible qt_take wall span.  The core
                 * resident_take_total_us above intentionally ends before
                 * this tail so the two costs remain distinguishable. */
                G.qt_last_take_wall_us=qt_now_us()-diag_take0;
                qt_record_take_diag(G.qt_last_take_wall_us);
            }
            return;
        }
        /* Recover on the exact same tensor set if a resident reduction or
         * download fails.  This is deliberately conservative: correctness is
         * more important than retaining the optimization on a sick device. */
        G.resident_active=0; G.resident_fallbacks++;
        for(int di=0;di<G.ndev;di++) if(G.is_cnt[di]){
            float *xr=G.is_x+(size_t)di*64*G.D;
            if(!coli_cuda_expert_group_issue(G.is_tg[di],G.is_tu[di],G.is_td[di],
                                              G.is_rows,G.is_cnt[di],xr)){
                for(int q=0;q<G.is_cnt[di];q++) mask &= ~(1u<<G.is_k[di][q]);
                G.is_cnt[di]=0;
            }
        }
    }
    if(mask) for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        struct timespec tti0, tti1;
        if (timing_on) clock_gettime(CLOCK_MONOTONIC, &tti0);
        const float *y=coli_cuda_expert_group_take(G.dev[di]);
        if (timing_on) {
            clock_gettime(CLOCK_MONOTONIC, &tti1);
            t_take_call_us += (tti1.tv_sec-tti0.tv_sec)*1e6 +
                              (tti1.tv_nsec-tti0.tv_nsec)/1e3;
        }
        if(!y) continue;
        for(int j=0;j<c;j++){
            float w=val[G.is_k[di][j]];
            const float *row=y+(size_t)j*G.D;
            for(int d=0;d<G.D;d++) out[d]+=w*row[d];
        }
        G.is_cnt[di]=0;
    }
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
    if (timing_on) {
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double total_us = (ts1.tv_sec-ts0.tv_sec)*1e6 + (ts1.tv_nsec-ts0.tv_nsec)/1e3;
        const char *path = getenv("QTIER_TIMING_FILE");
        FILE *fp = path ? fopen(path, "a") : stderr;
        if (fp) {
            fprintf(fp, "tier_pre_us=%.2f tier_issue_us=%.2f tier_take_us=%.2f tier_total_us=%.2f\n",
                    G.timing_pre_us, G.timing_issue_us, t_take_call_us, total_us);
            if (path) fclose(fp);
        }
    }
    if (resident_diag) {
        G.qt_last_take_wall_us=qt_now_us()-diag_take0;
        qt_record_take_diag(G.qt_last_take_wall_us);
    }
}

void qt_resident_timing_totals(double *gpu_us, double *sync_wait_us,
                               double *take_api_us, double *reduce_us,
                               double *d2h_us, double *take_total_us,
                               uint64_t *take_calls, double *qt_take_us,
                               uint64_t *qt_take_calls){
    if(gpu_us) *gpu_us=G.resident_gpu_event_us;
    if(sync_wait_us) *sync_wait_us=G.resident_sync_wait_us;
    if(take_api_us) *take_api_us=G.resident_take_api_us;
    if(reduce_us) *reduce_us=G.resident_reduce_event_us;
    if(d2h_us) *d2h_us=G.resident_d2h_diag_us;
    if(take_total_us) *take_total_us=G.resident_take_total_us;
    if(take_calls) *take_calls=G.resident_timing_calls;
    if(qt_take_us) *qt_take_us=G.qt_take_diag_us;
    if(qt_take_calls) *qt_take_calls=G.qt_take_diag_calls;
}

void qt_resident_timing_call_report(void){
    if(!G.resident_timing_decode || !G.qt_take_diag_calls) return;
    static const char *bn[8]={"<1","1-2","2-4","4-8","8-16","16-32","32-64",">=64"};
    fprintf(stderr,"[timers]   resident-call-dist: n=%llu avg=%.2f ms |",
            (unsigned long long)G.qt_take_diag_calls,
            G.qt_take_diag_us/(1000.0*(double)G.qt_take_diag_calls));
    for(int i=0;i<8;i++) fprintf(stderr," %s=%llu",bn[i],(unsigned long long)G.qt_take_bins[i]);
    fputc('\n',stderr);
    if(getenv("COLI_TIMERS_DETAIL") && atoi(getenv("COLI_TIMERS_DETAIL"))){
        int used[5]={0,0,0,0,0};
        for(int rank=0;rank<5;rank++){
            int best=-1;
            for(int l=0;l<G.nl;l++){
                if(!G.qt_take_layer_calls[l]) continue;
                int seen=0; for(int j=0;j<rank;j++) if(used[j]==l) seen=1;
                if(seen) continue;
                if(best<0 || G.qt_take_layer_max_us[l]>G.qt_take_layer_max_us[best]) best=l;
            }
            if(best<0) break;
            used[rank]=best;
            fprintf(stderr,"[timers]   resident-call-tail: layer=%d n=%llu avg=%.2f max=%.2f ms\n",
                    best,(unsigned long long)G.qt_take_layer_calls[best],
                    G.qt_take_layer_us[best]/(1000.0*(double)G.qt_take_layer_calls[best]),
                    G.qt_take_layer_max_us[best]/1000.0);
            uint64_t routes=G.qt_issue_layer_routes[best], gpu=G.qt_issue_layer_gpu[best];
            fprintf(stderr,"[timers]   resident-call-mix: layer=%d gpu=%llu/%llu (%.1f%%)\n",
                    best,(unsigned long long)gpu,(unsigned long long)routes,
                    routes?100.0*(double)gpu/(double)routes:0.0);
        }
    }
}

void qt_resident_timing_take_layers(double *take_us, uint64_t *calls, int n){
    if(!take_us || !calls || n<1) return;
    int lim=n<G.nl?n:G.nl;
    for(int l=0;l<lim;l++){
        take_us[l]=G.qt_take_layer_us?G.qt_take_layer_us[l]:0.0;
        calls[l]=G.qt_take_layer_calls?G.qt_take_layer_calls[l]:0;
    }
}

void qt_resident_timing_last(double *gpu_us, double *sync_us,
                             double *take_wall_us, int *gpu_valid){
    if(gpu_us) *gpu_us=G.qt_last_gpu_event_us;
    if(sync_us) *sync_us=G.qt_last_sync_wait_us;
    if(take_wall_us) *take_wall_us=G.qt_last_take_wall_us;
    if(gpu_valid) *gpu_valid=G.qt_last_gpu_valid;
}

void qt_resident_timing_last_host(double *gpu_lower_ms, double *gpu_upper_ms,
                                  double *reduce_lower_ms, double *reduce_upper_ms){
    if(gpu_lower_ms) *gpu_lower_ms=G.qt_last_gpu_lower_ms;
    if(gpu_upper_ms) *gpu_upper_ms=G.qt_last_gpu_upper_ms;
    if(reduce_lower_ms) *reduce_lower_ms=G.qt_last_reduce_lower_ms;
    if(reduce_upper_ms) *reduce_upper_ms=G.qt_last_reduce_upper_ms;
}

void qt_resident_timing_last_boundaries(double *gpu_complete_ms,
                                        double *merge_begin_ms){
    if(gpu_complete_ms) *gpu_complete_ms=G.qt_last_gpu_complete_ms;
    if(merge_begin_ms) *merge_begin_ms=G.qt_last_merge_begin_ms;
}

void qt_resident_timing_scope(int decode){
    G.resident_timing_decode=decode!=0;
}

void qt_resident_timing_reset(void){
    G.resident_gpu_event_us=0.0;
    G.resident_sync_wait_us=0.0;
    G.resident_take_api_us=0.0;
    G.resident_reduce_event_us=0.0;
    G.resident_d2h_diag_us=0.0;
    G.resident_take_total_us=0.0;
    G.resident_timing_calls=0;
    G.qt_take_diag_us=0.0;
    G.qt_take_diag_calls=0;
    G.qt_last_gpu_event_us=0.0;
    G.qt_last_sync_wait_us=0.0;
    G.qt_last_take_wall_us=0.0;
    G.qt_last_gpu_lower_ms=0.0;
    G.qt_last_gpu_upper_ms=0.0;
    G.qt_last_gpu_complete_ms=0.0;
    G.qt_last_merge_begin_ms=0.0;
    G.qt_last_reduce_lower_ms=0.0;
    G.qt_last_reduce_upper_ms=0.0;
    G.qt_last_gpu_valid=0;
    memset(G.qt_take_bins,0,sizeof G.qt_take_bins);
    if(G.qt_take_layer_us) memset(G.qt_take_layer_us,0,(size_t)G.nl*sizeof(double));
    if(G.qt_take_layer_max_us) memset(G.qt_take_layer_max_us,0,(size_t)G.nl*sizeof(double));
    if(G.qt_take_layer_calls) memset(G.qt_take_layer_calls,0,(size_t)G.nl*sizeof(uint64_t));
    if(G.qt_issue_layer_routes) memset(G.qt_issue_layer_routes,0,(size_t)G.nl*sizeof(uint64_t));
    if(G.qt_issue_layer_gpu) memset(G.qt_issue_layer_gpu,0,(size_t)G.nl*sizeof(uint64_t));
}

uint64_t qt_issue_batch(int layer, const int *eids, int routes,
                        const float *x, int rows){
    if(!G.on || !eids || !x || rows<2 || rows>8 || routes<1 || routes>64 ||
       routes%rows!=0 || routes/rows>32) return 0;
    G.batch_calls++; G.batch_routes+=(uint64_t)routes;
    int topk=routes/rows;
    uint64_t mask=0;
    ColiCudaTensor *tg[QT_MAX_DEV][64],*tu[QT_MAX_DEV][64],*td[QT_MAX_DEV][64];
    int one_rows[64];
    for(int i=0;i<64;i++) one_rows[i]=1;
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    pthread_mutex_lock(&G.mx);
    G.issue_open=1;
    for(int r=0;r<routes;r++){
        int eid=eids[r];
        if(eid<0 || eid>=G.ne){ G.miss++; continue; }
        QSlot *s=qs(layer,eid);
        if(!s->resident){ G.miss++; continue; }
        int di=home(eid), c=(di>=0 && di<G.ndev)?G.is_cnt[di]:64;
        if(c>=64){ G.miss++; continue; }
        tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
        G.is_k[di][c]=r; G.is_cnt[di]=c+1;
        mask |= 1ull<<r; G.hits[di]++;
    }
    pthread_mutex_unlock(&G.mx);

    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x+(size_t)di*64*G.D;
        for(int j=0;j<c;j++){
            int r=G.is_k[di][j];
            memcpy(xr+(size_t)j*G.D,
                   x+(size_t)(r/topk)*G.D,
                   (size_t)G.D*sizeof(float));
        }
        if(!coli_cuda_expert_group_issue_batch(tg[di],tu[di],td[di],
                                               one_rows,c,xr)){
            for(int j=0;j<c;j++) mask &= ~(1ull<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
    }
    return mask;
}

void qt_take_batch(uint64_t mask,const float *val,int routes,float *out,int rows){
    if(!G.on || !val || !out || routes<1 || routes>64 || rows<1) return;
    int topk=routes/rows;
    if(mask) for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        const float *y=coli_cuda_expert_group_take(G.dev[di]);
        if(!y) continue;
        for(int j=0;j<c;j++){
            int r=G.is_k[di][j];
            int row=r/topk;
            if(row<0 || row>=rows) continue;
            float w=val[r];
            const float *src=y+(size_t)j*G.D;
            float *dst=out+(size_t)row*G.D;
            for(int d=0;d<G.D;d++) dst[d]+=w*src[d];
        }
        G.is_cnt[di]=0;
    }
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}

void qt_stats(void){
    if(!G.on) return;
    uint64_t hits=0; size_t res=0;
    uint64_t resident_by_dev[QT_MAX_DEV]={0}, queued_by_dev[QT_MAX_DEV]={0};
    pthread_mutex_lock(&G.mx);
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++) {
        int e=(int)(i%G.ne), di=home(e);
        res += G.slot[i].resident;
        if(di>=0 && di<G.ndev) {
            resident_by_dev[di] += G.slot[i].resident;
            queued_by_dev[di] += G.slot[i].queued;
        }
    }
    pthread_mutex_unlock(&G.mx);
    fprintf(stderr,"[qtier] resident %zu/%d experts | uploads %llu | miss(CPU) %llu | q_skips %llu\n",
            res, G.nl*G.ne, (unsigned long long)G.uploads,
            (unsigned long long)G.miss, (unsigned long long)G.q_full_skips);
    for(int i=0;i<G.ndev;i++)
        fprintf(stderr,"[qtier]   placement GPU%d island%d: resident %llu | queued %llu | home experts %d\n",
                G.dev[i], G.topology.gpu[i].island,
                (unsigned long long)resident_by_dev[i],
                (unsigned long long)queued_by_dev[i], G.ne/G.ndev + (i<G.ne%G.ndev));
    if(G.batch_calls)
        fprintf(stderr,"[qtier] prefill batches %llu | route rows %llu | avg routes %.1f\n",
                (unsigned long long)G.batch_calls,
                (unsigned long long)G.batch_routes,
                (double)G.batch_routes/(double)G.batch_calls);
    if(G.resident_calls)
        fprintf(stderr,"[qtier] resident island: %llu calls | %llu resident experts | fallbacks %llu | H2D %.3f ms | issue/P2P %.3f ms | reduce %.3f ms | D2H %.3f ms\n",
                (unsigned long long)G.resident_calls,
                (unsigned long long)G.resident_experts,
                (unsigned long long)G.resident_fallbacks,
                G.resident_h2d_us/1000.0,G.resident_issue_us/1000.0,
                G.resident_reduce_us/1000.0,G.resident_d2h_us/1000.0);
    if(G.resident_single_direct)
        fprintf(stderr,"[qtier-exec] single-island direct accumulators %llu (sum_slots skipped)\n",
                (unsigned long long)G.resident_single_direct);
    if (G.exec_descriptors)
        fprintf(stderr,"[qtier-exec] layer descriptors %llu | GPU island batches %llu | GPU island experts %llu\n",
                (unsigned long long)G.exec_descriptors,
                (unsigned long long)G.exec_island_batches,
                (unsigned long long)G.exec_island_experts);
    if (G.stage_calls || G.stage_updates || G.stage_cache_hits || G.stage_failures)
        fprintf(stderr,"[qtier-exec] execution-cache routes %llu | new uploads %llu | cache hits %llu | update failures %llu | slots/device %d\n",
                (unsigned long long)G.stage_calls,
                (unsigned long long)G.stage_updates,
                (unsigned long long)G.stage_cache_hits,
                (unsigned long long)G.stage_failures, G.stage_slots);
    for(int i=0;i<G.ndev;i++){
        size_t tc=0,tb=0,freeb=0,totalb=0;
        coli_cuda_stats(G.dev[i],&tc,&tb);
        coli_cuda_mem_info(G.dev[i],&freeb,&totalb);
        hits+=G.hits[i];
        uint64_t gc=0,ge=0,gr=0; double gh=0,gk=0,gd=0;
        coli_cuda_group_stats_device(G.dev[i],&gc,&ge,&gr,&gh,&gk,&gd);
        /* Keep the group counters in the same line as the device that owns
         * them; this is the machine-readable unit for later Brain and
         * multi-island dashboards. */
        fprintf(stderr,"[qtier]   dev %d: hits %llu | %zu tensors, %.2f GB VRAM used "
                       "(budget %.2f GB, free %.2f GB/%.2f GB) | groups calls %llu "
                       "experts %llu rows %llu | H2D %.1f ms K %.1f ms D2H %.1f ms\n",
                G.dev[i], (unsigned long long)G.hits[i], tc, tb/1073741824.0,
                G.budget[i]/1073741824.0, freeb/1073741824.0, totalb/1073741824.0,
                (unsigned long long)gc, (unsigned long long)ge,
                (unsigned long long)gr, gh, gk, gd);
    }
    for(int i=0;i<G.topology.storage_count;i++){
        const QtStorageTopology *s=&G.topology.storage[i];
        fprintf(stderr,"[qtier]   storage%d: reads %llu | %.3f GB | busy %.3f ms | controller %d root %d\n",
                s->replica, (unsigned long long)s->read_ops,
                s->read_bytes/1073741824.0, s->busy_ns/1000000.0,
                s->controller, s->pcie_root);
    }
    double tot=(double)(hits+G.miss);
    fprintf(stderr,"[qtier] VRAM hit rate: %.1f %% | LFRU swaps %llu\n",
            tot>0? 100.0*hits/tot : 0.0, (unsigned long long)G.swaps);
    { uint64_t calls=0,ex=0,rows=0; double h2d=0,kms=0,d2h=0;
      coli_cuda_group_stats(&calls,&ex,&rows,&h2d,&kms,&d2h);
      if(calls) fprintf(stderr,"[qtier] group_stats: %llu calls, %llu experts | h2d %.0f ms, kernel %.0f ms, d2h %.0f ms\n",
              (unsigned long long)calls,(unsigned long long)ex,h2d,kms,d2h); }
}

void qt_record_storage_read(int replica, uint64_t bytes, uint64_t busy_ns){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    qt_topology_record_storage_read(&G.topology,replica,bytes,busy_ns);
    pthread_mutex_unlock(&G.mx);
}

void qt_record_storage_read_path(const char *path, uint64_t bytes, uint64_t busy_ns){
    if(!G.on || !path) return;
    pthread_mutex_lock(&G.mx);
    qt_topology_record_storage_read_path(&G.topology,path,bytes,busy_ns);
    pthread_mutex_unlock(&G.mx);
}

void qt_set_expert_storage(int layer, int eid, const char *path){
    qt_set_expert_storage_source(layer,eid,path,0);
}

void qt_set_expert_storage_source(int layer, int eid, const char *path, int replica){
    if(!G.on || !path || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return;
    pthread_mutex_lock(&G.mx);
    QSlot *s=qs(layer,eid);
    if(replica>0){
        for(int i=0;i<G.topology.storage_count;i++)
            if(G.topology.storage[i].replica==replica){
                s->disk_replica=replica;
                s->disk_controller=G.topology.storage[i].controller;
                pthread_mutex_unlock(&G.mx);
                return;
            }
    }
    for(int i=0;i<G.topology.storage_count;i++){
        const char *root=G.topology.storage[i].path; size_t n=strlen(root), plen=strlen(path);
        if(n && n<=plen && strncmp(path,root,n)==0){
            char c=path[n];
            if(c && c!='/' && c!='\\') continue;
            s->disk_replica=G.topology.storage[i].replica;
            s->disk_controller=G.topology.storage[i].controller;
            break;
        }
    }
    pthread_mutex_unlock(&G.mx);
}

int qt_numa_bind_arena(void *base, size_t bytes){
#ifdef __linux__
    if(!G.on || !base || !bytes || !G.topology.numa_requested ||
       G.topology.numa_nodes<2) return 0;
    uintptr_t lo=((uintptr_t)base+4095u)&~(uintptr_t)4095u;
    uintptr_t hi=((uintptr_t)base+bytes)&~(uintptr_t)4095u;
    if(hi<=lo) return 0;
    unsigned long mask = G.topology.numa_nodes >= sizeof(mask)*8
                       ? ~0ul : ((1ul<<G.topology.numa_nodes)-1ul);
    errno=0;
    long rc=syscall(SYS_mbind,lo,(size_t)(hi-lo),3/*MPOL_INTERLEAVE*/,
                    &mask,(unsigned long)(G.topology.numa_nodes+1),0);
    if(rc==0){
        pthread_mutex_lock(&G.mx);
        G.topology.numa_enabled=1;
        pthread_mutex_unlock(&G.mx);
        qt_topology_report(&G.topology,"[qtier]");
        return 1;
    }
    fprintf(stderr,"[qtier] NUMA arena interleave unavailable: %s\n",strerror(errno));
#else
    (void)base; (void)bytes;
#endif
    return 0;
}

void qt_shutdown(void){
    if(!G.on) return;
    const char *hf=getenv("HEAT_FILE");
    const char *heat_readonly=getenv("QTIER_HEAT_READONLY");
    if(hf && !(heat_readonly && atoi(heat_readonly))){
        FILE *f=fopen(hf,"wb");
        if(f){
            uint32_t hdr[3]={0x51544831u,(uint32_t)G.nl,(uint32_t)G.ne};
            fwrite(hdr,4,3,f);
            for(size_t i=0;i<(size_t)G.nl*G.ne;i++) fwrite(&G.slot[i].heat,4,1,f);
            fclose(f);
            fprintf(stderr,"[qtier] HEAT_FILE saved: %s\n",hf);
        }
    } else if(hf){
        fprintf(stderr,"[qtier] HEAT_FILE read-only: %s\n",hf);
    }
    pthread_mutex_lock(&G.mx); G.th_stop=1; pthread_cond_signal(&G.cv); pthread_mutex_unlock(&G.mx);
    pthread_join(G.th,NULL);
    G.on=0;
    free(G.qt_take_layer_us);
    free(G.qt_take_layer_max_us);
    free(G.qt_take_layer_calls);
    free(G.qt_issue_layer_routes);
    free(G.qt_issue_layer_gpu);
    G.qt_take_layer_us=G.qt_take_layer_max_us=NULL;
    G.qt_take_layer_calls=NULL;
    G.qt_issue_layer_routes=G.qt_issue_layer_gpu=NULL;
    if(G.dense){
        for(int i=0;i<G.nl;i++){
            QtDenseLayer *d=&G.dense[i];
            float *p[]={d->conv,d->b,d->a,d->dtbias,d->alog,d->norm,d->rec,d->ring};
            for(size_t k=0;k<sizeof(p)/sizeof(p[0]);k++)
                if(p[k]) coli_cuda_pipe_free(G.dense_device,p[k]);
            d->conv=d->b=d->a=d->dtbias=d->alog=d->norm=d->rec=d->ring=NULL;
            d->full=0;
        }
    }
    if(G.dense_lm_head){
        coli_cuda_tensor_free(G.dense_lm_head);
        G.dense_lm_head=NULL; G.dense_lm_I=G.dense_lm_O=0;
    }
    qt_stage_shutdown();
    coli_cuda_shutdown();
    free(G.dense_pack); G.dense_pack=NULL; G.dense_pack_cap=0;
    free(G.dense); G.dense=NULL; G.dense_enabled=0;
}

#endif /* COLI_CUDA */
