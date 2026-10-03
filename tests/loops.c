/* Loop-transform correctness: each work-item runs counted loops with its own
 * (start, bound, step) and the results are checked against the same loops on
 * the CPU. Order-dependent hashes catch skipped, repeated or reordered
 * iterations. Exit 0 = all match. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do{cl_int e_=(x);if(e_){printf("CL error %d at line %d\n",e_,__LINE__);exit(2);}}while(0)
static const char *src =
"kernel void s32(global const int *A, global const int *B, global const int *S, global uint *H, global int *I){\n"
"  size_t g=get_global_id(0); int a=A[g], b=B[g], s=S[g]; uint h=7; int i;\n"
"  for (i=a; i<b; i+=s) h = h*31u + (uint)i;\n"
"  H[g]=h; I[g]=i; }\n"
"kernel void s32_step1(global const int *A, global const int *B, global uint *H){\n"
"  size_t g=get_global_id(0); int a=A[g], b=B[g]; uint h=7;\n"
"  for (int i=a; i<b; i++) h = h*31u + (uint)i;\n"
"  H[g]=h; }\n"
"kernel void u32(global const uint *A, global const uint *B, global const uint *S, global uint *H){\n"
"  size_t g=get_global_id(0); uint a=A[g], b=B[g], s=S[g]; uint h=7;\n"
"  for (uint i=a; i<b; i+=s) h = (h ^ i) * 16777619u;\n"
"  H[g]=h; }\n"
"kernel void s64(global const long *A, global const long *B, global ulong *H){\n"
"  size_t g=get_global_id(0); long a=A[g], b=B[g]; ulong h=7;\n"
"  for (long i=a; i<b; i++) h = h*1099511628211ul + (ulong)i;\n"
"  H[g]=h; }\n"
"kernel void stores(global const int *A, global const int *B, global int *O, int W){\n"
"  size_t g=get_global_id(0); int a=A[g], b=B[g]; global int *o=O+g*W;\n"
"  for (int i=a; i<b; i++) o[i-a] = i*3+1;\n"
"}\n"
"kernel void branchy(global const int *A, global const int *B, global uint *H){\n"
"  size_t g=get_global_id(0); int a=A[g], b=B[g]; uint h=1;\n"
"  for (int i=a; i<b; i++) { if (i & 1) h = h*3u + (uint)i; else h ^= (uint)i << 3; }\n"
"  H[g]=h; }\n"
"kernel void constn(global uint *H){\n"
"  size_t g=get_global_id(0); uint h=(uint)g;\n"
"  for (int i=0; i<1000; i++) h = h*31u + (uint)i;\n"
"  H[g]=h; }\n"
"kernel void reduce(global const float *X, global float *O, int n, local float *t){\n"
"  size_t l=get_local_id(0), g=get_group_id(0); float acc=0;\n"
"  for (int i=(int)l; i<n; i+=(int)get_local_size(0)) acc += X[g*n+i];\n"
"  t[l]=acc; barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for (int s=1; s<(int)get_local_size(0); s<<=1) { if ((l & (2*s-1))==0) t[l]+=t[l+s]; barrier(CLK_LOCAL_MEM_FENCE);} \n"
"  if (l==0) O[g]=t[0]; }\n";
static cl_context ctx; static cl_command_queue q;
static cl_mem mk(size_t n, void *h){cl_int e;cl_mem m=clCreateBuffer(ctx,CL_MEM_READ_WRITE|(h?CL_MEM_COPY_HOST_PTR:0),n,h,&e);CK(e);return m;}
static void rd(cl_mem m,size_t n,void*h){CK(clEnqueueReadBuffer(q,m,CL_TRUE,0,n,h,0,0,0));}
static uint32_t rng=12345; static uint32_t rnd(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
int main(void){
  cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
  cl_int e;ctx=clCreateContext(0,1,&d,0,0,&e);CK(e);q=clCreateCommandQueueWithProperties(ctx,d,0,&e);CK(e);
  cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);
  if(clBuildProgram(pr,1,&d,"",0,0)){char l[16384];clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,sizeof l,l,0);puts(l);return 2;}
  int fails=0; const size_t N=8192;
  /* ---- s32 with steps, edge ranges ---- */
  { int *A=malloc(N*4),*B=malloc(N*4),*S=malloc(N*4),*I=malloc(N*4); uint32_t *H=malloc(N*4);
    for(size_t g=0;g<N;g++){ int s=1+(int)(rnd()%7); int len=(int)(rnd()%40)-5; int base;
      switch(g%6){case 0: base=(int)(rnd()%1000)-500; break; case 1: base=-2147483647+(int)(rnd()%10); break;
        case 2: base=2147483647-60; break; case 3: base=0; len=(int)(g%20); break; case 4: base=(int)(rnd()%100); len=1000+(int)(rnd()%50); break; default: base=5; len=-(int)(rnd()%5);}
      A[g]=base; long long bb=(long long)base+len; if(bb>2147483647LL-7) bb=2147483647LL-7; if(bb<-2147483647LL) bb=-2147483647LL; B[g]=(int)bb; S[g]=s; }
    cl_mem a=mk(N*4,A),b=mk(N*4,B),s=mk(N*4,S),h=mk(N*4,0),i=mk(N*4,0);
    cl_kernel k=clCreateKernel(pr,"s32",&e);CK(e);clSetKernelArg(k,0,8,&a);clSetKernelArg(k,1,8,&b);clSetKernelArg(k,2,8,&s);clSetKernelArg(k,3,8,&h);clSetKernelArg(k,4,8,&i);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&N,0,0,0,0)); rd(h,N*4,H); rd(i,N*4,I);
    size_t bad=0; for(size_t g=0;g<N;g++){ uint32_t hh=7; int ii; for(ii=A[g]; ii<B[g]; ii+=S[g]) hh=hh*31u+(uint32_t)ii;
      if(hh!=H[g]||ii!=I[g]){ if(bad<5) printf("  s32 g=%zu a=%d b=%d s=%d: gpu h=%u i=%d cpu h=%u i=%d\n",g,A[g],B[g],S[g],H[g],I[g],hh,ii); bad++; } }
    printf("%-10s %s (%zu/%zu wrong)\n","s32",bad?"FAIL":"ok",bad,N); fails+=bad!=0;
    /* step 1 variant on the same ranges */
    cl_kernel k1=clCreateKernel(pr,"s32_step1",&e);CK(e);clSetKernelArg(k1,0,8,&a);clSetKernelArg(k1,1,8,&b);clSetKernelArg(k1,2,8,&h);
    CK(clEnqueueNDRangeKernel(q,k1,1,0,&N,0,0,0,0)); rd(h,N*4,H); bad=0;
    for(size_t g=0;g<N;g++){ uint32_t hh=7; for(int ii=A[g]; ii<B[g]; ii++) hh=hh*31u+(uint32_t)ii; if(hh!=H[g]) bad++; }
    printf("%-10s %s (%zu/%zu wrong)\n","s32_step1",bad?"FAIL":"ok",bad,N); fails+=bad!=0; }
  /* ---- u32 near UINT_MAX ---- */
  { uint32_t *A=malloc(N*4),*B=malloc(N*4),*S=malloc(N*4),*H=malloc(N*4);
    for(size_t g=0;g<N;g++){ uint32_t s=1+rnd()%5; uint32_t base=(g%3==0)?0xFFFFFFFFu-100:(g%3==1)?rnd()%100:0x80000000u-20; uint32_t len=rnd()%60;
      uint64_t bb=(uint64_t)base+len; if(bb>0xFFFFFFFFull-8) bb=0xFFFFFFFFull-8; A[g]=base; B[g]=(uint32_t)bb; S[g]=s; }
    cl_mem a=mk(N*4,A),b=mk(N*4,B),s=mk(N*4,S),h=mk(N*4,0);
    cl_kernel k=clCreateKernel(pr,"u32",&e);CK(e);clSetKernelArg(k,0,8,&a);clSetKernelArg(k,1,8,&b);clSetKernelArg(k,2,8,&s);clSetKernelArg(k,3,8,&h);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&N,0,0,0,0)); rd(h,N*4,H); size_t bad=0;
    for(size_t g=0;g<N;g++){ uint32_t hh=7; for(uint32_t ii=A[g]; ii<B[g]; ii+=S[g]) hh=(hh^ii)*16777619u; if(hh!=H[g]){ if(bad<3) printf("  u32 g=%zu a=%u b=%u s=%u\n",g,A[g],B[g],S[g]); bad++;} }
    printf("%-10s %s (%zu/%zu wrong)\n","u32",bad?"FAIL":"ok",bad,N); fails+=bad!=0; }
  /* ---- s64 ---- */
  { int64_t *A=malloc(N*8),*B=malloc(N*8); uint64_t *H=malloc(N*8);
    for(size_t g=0;g<N;g++){ int64_t base=(g%2)?((int64_t)1<<40)-(int64_t)(rnd()%50):-(int64_t)(rnd()%3000); A[g]=base; B[g]=base+(int64_t)(rnd()%45)-3; }
    cl_mem a=mk(N*8,A),b=mk(N*8,B),h=mk(N*8,0);
    cl_kernel k=clCreateKernel(pr,"s64",&e);CK(e);clSetKernelArg(k,0,8,&a);clSetKernelArg(k,1,8,&b);clSetKernelArg(k,2,8,&h);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&N,0,0,0,0)); rd(h,N*8,H); size_t bad=0;
    for(size_t g=0;g<N;g++){ uint64_t hh=7; for(int64_t ii=A[g]; ii<B[g]; ii++) hh=hh*1099511628211ull+(uint64_t)ii; if(hh!=H[g]) bad++; }
    printf("%-10s %s (%zu/%zu wrong)\n","s64",bad?"FAIL":"ok",bad,N); fails+=bad!=0; }
  /* ---- stores in the body ---- */
  { const int W=48; int *A=malloc(N*4),*B=malloc(N*4),*O=malloc(N*W*4);
    for(size_t g=0;g<N;g++){ A[g]=(int)(rnd()%200)-100; B[g]=A[g]+(int)(rnd()%(W+1)); }
    memset(O,0xAB,N*W*4); cl_mem a=mk(N*4,A),b=mk(N*4,B),o=mk(N*W*4,O);
    cl_kernel k=clCreateKernel(pr,"stores",&e);CK(e);clSetKernelArg(k,0,8,&a);clSetKernelArg(k,1,8,&b);clSetKernelArg(k,2,8,&o);clSetKernelArg(k,3,4,&W);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&N,0,0,0,0)); int *R=malloc(N*W*4); rd(o,N*W*4,R); size_t bad=0;
    for(size_t g=0;g<N;g++) for(int j=0;j<W;j++){ int want=(j<B[g]-A[g])?(A[g]+j)*3+1:(int)0xABABABAB; if(R[g*W+j]!=want) bad++; }
    printf("%-10s %s (%zu/%zu wrong)\n","stores",bad?"FAIL":"ok",bad,N*W); fails+=bad!=0; }
  /* ---- if inside the body ---- */
  { int *A=malloc(N*4),*B=malloc(N*4); uint32_t *H=malloc(N*4);
    for(size_t g=0;g<N;g++){ A[g]=(int)(rnd()%100); B[g]=A[g]+(int)(rnd()%70)-4; }
    cl_mem a=mk(N*4,A),b=mk(N*4,B),h=mk(N*4,0);
    cl_kernel k=clCreateKernel(pr,"branchy",&e);CK(e);clSetKernelArg(k,0,8,&a);clSetKernelArg(k,1,8,&b);clSetKernelArg(k,2,8,&h);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&N,0,0,0,0)); rd(h,N*4,H); size_t bad=0;
    for(size_t g=0;g<N;g++){ uint32_t hh=1; for(int i=A[g];i<B[g];i++){ if(i&1) hh=hh*3u+(uint32_t)i; else hh^=(uint32_t)i<<3; } if(hh!=H[g]) bad++; }
    printf("%-10s %s (%zu/%zu wrong)\n","branchy",bad?"FAIL":"ok",bad,N); fails+=bad!=0; }
  /* ---- constant trip count ---- */
  { uint32_t *H=malloc(N*4); cl_mem h=mk(N*4,0); cl_kernel k=clCreateKernel(pr,"constn",&e);CK(e);clSetKernelArg(k,0,8,&h);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&N,0,0,0,0)); rd(h,N*4,H); size_t bad=0;
    for(size_t g=0;g<N;g++){ uint32_t hh=(uint32_t)g; for(int i=0;i<1000;i++) hh=hh*31u+(uint32_t)i; if(hh!=H[g]) bad++; }
    printf("%-10s %s (%zu/%zu wrong)\n","constn",bad?"FAIL":"ok",bad,N); fails+=bad!=0; }
  /* ---- strided loop + barriers ---- */
  { int G=64,n=1000; size_t L=128,T=G*L; float *X=malloc(G*n*4),*O=malloc(G*4);
    for(int j=0;j<G*n;j++) X[j]=(float)((j*7)%13); cl_mem x=mk(G*n*4,X),o=mk(G*4,0);
    cl_kernel k=clCreateKernel(pr,"reduce",&e);CK(e);clSetKernelArg(k,0,8,&x);clSetKernelArg(k,1,8,&o);clSetKernelArg(k,2,4,&n);clSetKernelArg(k,3,L*4,0);
    CK(clEnqueueNDRangeKernel(q,k,1,0,&T,&L,0,0,0)); rd(o,G*4,O); size_t bad=0;
    for(int g=0;g<G;g++){ double sum=0; for(int j=0;j<n;j++) sum+=X[g*n+j]; if(O[g]!=(float)sum) bad++; }
    printf("%-10s %s (%zu/%d wrong)\n","reduce",bad?"FAIL":"ok",bad,G); fails+=bad!=0; }
  printf("%s\n",fails?"FAIL":"PASS"); return fails?1:0; }
