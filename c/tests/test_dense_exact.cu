/* Model-free parity gate for the opt-in one-thread-per-output dense GEMV.
 * The reference uses the same four eight-lane accumulation streams and final
 * tree as qwen36.c's AVX2 matmul_q(), but spells the rounding points out with
 * fmaf/fadd so this test does not depend on host vectorization. */
#include "../backend_cuda.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float ref_one(const float *x,const int8_t *w,float scale,int I){
    float p[32]={};
    for(int i=0;i<I;i+=32) for(int j=0;j<32;j++)
        p[j]=fmaf(x[i+j],(float)w[i+j],p[j]);
    float b[8];
    for(int j=0;j<8;j++) b[j]=(p[j]+p[j+8])+(p[j+16]+p[j+24]);
    float s0=(b[0]+b[4])+(b[2]+b[6]);
    float s1=(b[1]+b[5])+(b[3]+b[7]);
    return (s0+s1)*scale;
}

static int one_shape(int I,int O,int salt){
    std::vector<float> x((size_t)I),sc((size_t)O),got((size_t)O),ref((size_t)O);
    std::vector<int8_t> q((size_t)I*O);
    for(int i=0;i<I;i++) x[i]=(float)(((i*37+salt*11)%257)-128)/91.0f;
    for(int o=0;o<O;o++) sc[o]=0.001f*(float)(1+(o*13+salt)%31);
    for(size_t i=0;i<q.size();i++) q[i]=(int8_t)(((int)(i*29+salt*7)%255)-127);
    ColiCudaTensor *t=nullptr;
    if(!coli_cuda_tensor_upload(&t,q.data(),sc.data(),1,I,O,0)){
        std::fprintf(stderr,"upload failed I=%d O=%d\n",I,O); return 0;
    }
    const int off[1]={0};
    for(int o=0;o<O;o++) ref[o]=ref_one(x.data(),q.data()+(size_t)o*I,sc[o],I);
    if(!coli_cuda_pipe_dense_batch(&t,off,1,I,x.data(),got.data(),O,0)){
        std::fprintf(stderr,"launch failed I=%d O=%d\n",I,O);
        coli_cuda_tensor_free(t); return 0;
    }
    int bad=0; float worst=0.f;
    for(int o=0;o<O;o++) if(std::memcmp(&ref[o],&got[o],sizeof(float))!=0){
        float d=std::fabs(ref[o]-got[o]); if(d>worst)worst=d;
        if(bad<4) std::fprintf(stderr,"diff I=%d O=%d o=%d ref=%a got=%a\n",I,O,o,ref[o],got[o]);
        bad++;
    }
    std::printf("dense exact parity: I=%d O=%d differing=%d max_abs=%g\n",I,O,bad,worst);
    coli_cuda_tensor_free(t);
    return bad==0;
}

int main(){
#ifdef _WIN32
    _putenv_s("COLI_DENSE_EXACT","1");
#else
    setenv("COLI_DENSE_EXACT","1",1);
#endif
    int devs[1]={0};
    if(!coli_cuda_init(devs,1)){std::fprintf(stderr,"CUDA init failed\n");return 2;}
    int ok=one_shape(1024,257,1)&&one_shape(2048,1024,2)&&one_shape(1024,4096,3);
    coli_cuda_shutdown();
    return ok?0:1;
}
