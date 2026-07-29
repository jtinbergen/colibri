/* qwen36_tier.c — M-QTIER: VRAM-Experten-Tier (Implementierung). Siehe Header. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "qwen36_tier.h"
#include "backend_cuda.h"

#define QT_MAX_DEV 8
#define QT_QCAP 48            /* Upload-Queue-Tiefe (Staging ~1,6 MB/Eintrag) */

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    uint32_t heat;
    uint8_t resident, queued;
} QSlot;

static struct {
    int on, nl, ne, D, Ih, topk, ndev;
    int dev[QT_MAX_DEV];
    size_t budget[QT_MAX_DEV], used[QT_MAX_DEV];
    size_t exp_bytes;                     /* VRAM-Bedarf je Experte (geschätzt) */
    QSlot *slot;                          /* [nl*ne] */
    pthread_mutex_t mx;
    pthread_t th;
    int th_stop;
    /* Upload-Queue (Ring) mit Staging-Kopien */
    struct { int layer, eid; uint8_t *w; float *s; } q[QT_QCAP];
    int qh, qt_, qn;
    pthread_cond_t cv;
    /* Statistik */
    uint64_t hits[QT_MAX_DEV], miss, uploads, q_full_skips;
    /* Issue-Zustand des Decode-Threads (single-threaded) */
    int is_cnt[QT_MAX_DEV];
    int is_k[QT_MAX_DEV][32];
    float *is_x;                          /* count*D Replikate je Device */
} G;

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return eid % G.ndev; }

/* Staging: packed int4 (g|u|d) Zweierkomplement -> Offset-Binary (XOR 0x88)
 * + Scales (gs|us|ds) kopieren. */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                  const float *gs,const float *us,const float *ds){
    size_t mb = (size_t)G.D*G.Ih/2;
    const uint64_t X=0x8888888888888888ull;
    const uint64_t *sg=(const uint64_t*)g4,*su=(const uint64_t*)u4,*sd=(const uint64_t*)d4;
    uint64_t *w0=(uint64_t*)dw,*w1=(uint64_t*)(dw+mb),*w2=(uint64_t*)(dw+2*mb);
    for(size_t i=0;i<mb/8;i++){ w0[i]=sg[i]^X; w1[i]=su[i]^X; w2[i]=sd[i]^X; }
    memcpy(dsc,            gs, (size_t)G.Ih*sizeof(float));
    memcpy(dsc+G.Ih,       us, (size_t)G.Ih*sizeof(float));
    memcpy(dsc+2*G.Ih,     ds, (size_t)G.D *sizeof(float));
}

static void *uploader(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&G.mx);
        while(G.qn==0 && !G.th_stop) pthread_cond_wait(&G.cv,&G.mx);
        if(G.th_stop && G.qn==0){ pthread_mutex_unlock(&G.mx); return NULL; }
        int layer=G.q[G.qh].layer, eid=G.q[G.qh].eid;
        uint8_t *w=G.q[G.qh].w; float *sc=G.q[G.qh].s;
        G.qh=(G.qh+1)%QT_QCAP; G.qn--;
        pthread_mutex_unlock(&G.mx);

        int dv = G.dev[home(eid)];
        size_t mb=(size_t)G.D*G.Ih/2;
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok = coli_cuda_tensor_upload(&tg, w,      sc,          2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   sc+G.Ih,     2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, sc+2*G.Ih,   2, G.Ih, G.D,  dv);
        free(w); free(sc);
        pthread_mutex_lock(&G.mx);
        QSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid); G.used[hd]-=G.exp_bytes;
                G.budget[hd]=G.used[hd];   /* VRAM real voll: nicht weiter versuchen */ }
        s->queued=0;
        pthread_mutex_unlock(&G.mx);
    }
}

int qt_init(int nl, int ne, int D, int Ih, int cap, int topk){
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(cap != ne){
        fprintf(stderr,"[qtier] cap=%d != n_experts=%d -> Tier aus (volle RAM-Residenz nötig, Z5)\n",cap,ne);
        return 0;
    }
    if(topk>32){ fprintf(stderr,"[qtier] topk>32 nicht unterstützt\n"); return 0; }
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk;

    /* Geräte: COLI_GPUS="0,1" (Default: 0,1 falls vorhanden, sonst 0) */
    const char *gl=getenv("COLI_GPUS");
    char buf[128]; snprintf(buf,sizeof buf,"%s", gl?gl:"0,1");
    for(char *t=strtok(buf,","); t && G.ndev<QT_MAX_DEV; t=strtok(NULL,","))
        G.dev[G.ndev++]=atoi(t);
    if(!coli_cuda_init(G.dev,G.ndev)){ fprintf(stderr,"[qtier] coli_cuda_init FAIL -> CPU\n"); return 0; }
    int have=coli_cuda_device_count();
    if(have<G.ndev){ G.ndev=have; }
    if(G.ndev<1){ fprintf(stderr,"[qtier] keine CUDA-Devices -> CPU\n"); return 0; }

    /* Budget je Device: CUDA_EXPERT_GB oder auto = frei - 1 GB Headroom */
    G.exp_bytes = 3ull*D*Ih/2 + (size_t)(2*Ih+D)*sizeof(float) + 4096; /* + Alloc-Verschnitt */
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        size_t b = (bg && strcmp(bg,"auto") && atof(bg)>0)
                   ? (size_t)(atof(bg)*1024.0*1024.0*1024.0)
                   : (freeb>(1ull<<30) ? freeb-(1ull<<30) : 0);
        G.budget[i]=b;
        fprintf(stderr,"[qtier] dev %d: %.1f GB frei, Budget %.1f GB (~%zu Experten)\n",
                G.dev[i], freeb/1073741824.0, b/1073741824.0, b/G.exp_bytes);
    }
    G.slot=calloc((size_t)nl*ne,sizeof(QSlot));
    G.is_x=malloc((size_t)32*D*sizeof(float));
    if(!G.slot||!G.is_x) return 0;
    pthread_mutex_init(&G.mx,NULL); pthread_cond_init(&G.cv,NULL);
    if(pthread_create(&G.th,NULL,uploader,NULL)!=0) return 0;
    G.on=1;
    fprintf(stderr,"[qtier] VRAM-Experten-Tier aktiv: %d Device(s), %.2f MB/Experte\n",
            G.ndev, G.exp_bytes/1048576.0);
    return 1;
}

int qt_ready(void){ return G.on; }

void qt_note(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(s->heat<0xFFFFFFFFu) s->heat++;
    int want = !s->resident && !s->queued;
    int hd = home(eid);
    if(want && G.used[hd]+G.exp_bytes<=G.budget[hd] && G.qn<QT_QCAP){
        size_t mb=(size_t)G.D*G.Ih/2;
        uint8_t *w=malloc(3*mb); float *sc=malloc((size_t)(2*G.Ih+G.D)*sizeof(float));
        if(w&&sc){
            G.used[hd]+=G.exp_bytes;            /* reservieren */
            s->queued=1;
            stage(w,sc,g4,u4,d4,gs,us,ds);
            G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid; G.q[G.qt_].w=w; G.q[G.qt_].s=sc;
            G.qt_=(G.qt_+1)%QT_QCAP; G.qn++;
            pthread_cond_signal(&G.cv);
        } else { free(w); free(sc); }
    } else if(want && G.qn>=QT_QCAP){
        G.q_full_skips++;
    }
    pthread_mutex_unlock(&G.mx);
}

uint32_t qt_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on||K>32) return 0;
    uint32_t mask=0;
    ColiCudaTensor *tg[QT_MAX_DEV][32],*tu[QT_MAX_DEV][32],*td[QT_MAX_DEV][32];
    static int rows[32]={0};
    if(!rows[0]) for(int i=0;i<32;i++) rows[i]=1;
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    pthread_mutex_lock(&G.mx);
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

    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x + (size_t)di*8*G.D;               /* je Device eigener x-Block */
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        if(!coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr)){
            /* Issue fehlgeschlagen -> diese k zurück an die CPU */
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
    }
    return mask;
}

void qt_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!G.on||!mask) return;
    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        const float *y=coli_cuda_expert_group_take(G.dev[di]);
        if(!y) continue;
        for(int j=0;j<c;j++){
            float w=val[G.is_k[di][j]];
            const float *row=y+(size_t)j*G.D;
            for(int d=0;d<G.D;d++) out[d]+=w*row[d];
        }
        G.is_cnt[di]=0;
    }
}

void qt_stats(void){
    if(!G.on) return;
    uint64_t hits=0; size_t res=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++) res += G.slot[i].resident;
    fprintf(stderr,"[qtier] resident %zu/%d Experten | uploads %llu | miss(CPU) %llu | q_skips %llu\n",
            res, G.nl*G.ne, (unsigned long long)G.uploads,
            (unsigned long long)G.miss, (unsigned long long)G.q_full_skips);
    for(int i=0;i<G.ndev;i++){
        size_t tc=0,tb=0; coli_cuda_stats(G.dev[i],&tc,&tb);
        hits+=G.hits[i];
        fprintf(stderr,"[qtier]   dev %d: hits %llu | %zu Tensoren, %.2f GB VRAM belegt (Budget %.2f GB)\n",
                G.dev[i], (unsigned long long)G.hits[i], tc, tb/1073741824.0, G.budget[i]/1073741824.0);
    }
    double tot=(double)(hits+G.miss);
    fprintf(stderr,"[qtier] VRAM-Hit-Rate: %.1f %%\n", tot>0? 100.0*hits/tot : 0.0);
}

void qt_shutdown(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx); G.th_stop=1; pthread_cond_signal(&G.cv); pthread_mutex_unlock(&G.mx);
    pthread_join(G.th,NULL);
    G.on=0;
    coli_cuda_shutdown();
}
