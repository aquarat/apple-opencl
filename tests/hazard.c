/* Hazard test for dispatch overlap: every case below is a dependency that the
 * driver must order even though the dispatches share one compute batch.
 * A "slow" kernel makes the producer still be running when the consumer is
 * issued, so a missing barrier shows up as wrong data, not as luck.
 * Usage: hazard [reps]   exit 0 = all pass */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do{cl_int e_=(x);if(e_){printf("CL error %d at line %d\n",e_,__LINE__);exit(2);}}while(0)
static const char *src =
"kernel void slow_fill(global int *x, int v, int iters){\n"
"  size_t i=get_global_id(0); float f=i; for(int k=0;k<iters;k++) f=fma(f,1.0000001f,0.0f);\n"
"  x[i]=v+(int)(f-f);}\n"
"kernel void fill(global int *x, int v){ x[get_global_id(0)]=v; }\n"
"kernel void add1(global const int *x, global int *y){ size_t i=get_global_id(0); y[i]=x[i]+1; }\n"
"kernel void slow_copy(global const int *x, global int *y, int iters){\n"
"  size_t i=get_global_id(0); float f=i; for(int k=0;k<iters;k++) f=fma(f,1.0000001f,0.0f);\n"
"  y[i]=x[i]+(int)(f-f);}\n"
"kernel void scratchy(global int *y, int seed){\n"
"  int a[64]; size_t i=get_global_id(0);\n"
"  for(int k=0;k<64;k++) a[k]=seed*k+(int)i;\n"
"  int s=0; for(int k=0;k<64;k++) s+=a[(k*7+i)&63];\n"
"  y[i]=s;}\n";
static cl_context ctx; static cl_command_queue q; static cl_program pr;
static cl_mem buf(size_t n){cl_int e;cl_mem m=clCreateBuffer(ctx,CL_MEM_READ_WRITE,n*4,0,&e);CK(e);return m;}
static cl_kernel K(const char*n){cl_int e;cl_kernel k=clCreateKernel(pr,n,&e);CK(e);return k;}
static void run(cl_kernel k,size_t n){CK(clEnqueueNDRangeKernel(q,k,1,0,&n,0,0,0,0));}
static int check(cl_mem m,size_t n,int(*expect)(size_t),const char*name){
  int*h=malloc(n*4);CK(clEnqueueReadBuffer(q,m,1,0,n*4,h,0,0,0));
  size_t bad=0,first=0;for(size_t i=0;i<n;i++)if(h[i]!=expect(i)){if(!bad)first=i;bad++;}
  if(bad)printf("FAIL %-28s %zu/%zu wrong, first [%zu]=%d want %d\n",name,bad,n,first,h[first],expect(first));
  free(h);return bad?1:0;}
static int e2(size_t i){return 2;} static int e3(size_t i){return 3;} static int e8(size_t i){return 8;}
static int e101(size_t i){return 101;} static int e7(size_t i){return 7;}
static int escr(size_t i){int a[64];for(int k=0;k<64;k++)a[k]=5*k+(int)i;int s=0;for(int k=0;k<64;k++)s+=a[(k*7+i)&63];return s;}
int main(int argc,char**argv){
  int reps=argc>1?atoi(argv[1]):20, iters=20000; size_t N=1<<16, S=256;
  cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
  cl_int e;ctx=clCreateContext(0,1,&d,0,0,&e);CK(e);q=clCreateCommandQueueWithProperties(ctx,d,0,&e);CK(e);
  pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(clBuildProgram(pr,1,&d,"",0,0));
  cl_kernel ksf=K("slow_fill"),kf=K("fill"),ka=K("add1"),ksc=K("slow_copy"),kscr=K("scratchy");
  cl_mem X=buf(N),Y=buf(N),Z=buf(N);
  int fails=0;
  for(int r=0;r<reps;r++){
    int v;
    /* RAW: slow producer, independent dispatch in between, consumer */
    v=1;clSetKernelArg(ksf,0,sizeof X,&X);clSetKernelArg(ksf,1,4,&v);clSetKernelArg(ksf,2,4,&iters);run(ksf,N);
    v=9;clSetKernelArg(kf,0,sizeof Z,&Z);clSetKernelArg(kf,1,4,&v);run(kf,N);
    clSetKernelArg(ka,0,sizeof X,&X);clSetKernelArg(ka,1,sizeof Y,&Y);run(ka,N);
    fails+=check(Y,N,e2,"RAW via independent");
    /* WAR: slow reader of X, then a fast writer of X */
    v=2;clSetKernelArg(kf,0,sizeof X,&X);clSetKernelArg(kf,1,4,&v);run(kf,N);
    clSetKernelArg(ksc,0,sizeof X,&X);clSetKernelArg(ksc,1,sizeof Y,&Y);clSetKernelArg(ksc,2,4,&iters);run(ksc,N);
    v=50;clSetKernelArg(kf,0,sizeof X,&X);clSetKernelArg(kf,1,4,&v);run(kf,N);
    fails+=check(Y,N,e2,"WAR");
    /* WAW: slow writer then fast writer of the same buffer: last one wins */
    v=1;clSetKernelArg(ksf,0,sizeof X,&X);clSetKernelArg(ksf,1,4,&v);run(ksf,N);
    v=3;clSetKernelArg(kf,0,sizeof X,&X);clSetKernelArg(kf,1,4,&v);run(kf,N);
    fails+=check(X,N,e3,"WAW");
    /* Sub-buffers: two views of one allocation */
    cl_buffer_region r0={0,N*2},r1={N*2,N*2};
    cl_mem P=buf(N),A=clCreateSubBuffer(P,0,CL_BUFFER_CREATE_TYPE_REGION,&r0,&e),B=clCreateSubBuffer(P,0,CL_BUFFER_CREATE_TYPE_REGION,&r1,&e);
    v=7;clSetKernelArg(ksf,0,sizeof A,&A);clSetKernelArg(ksf,1,4,&v);run(ksf,N/2);
    clSetKernelArg(ka,0,sizeof P,&P);clSetKernelArg(ka,1,sizeof Y,&Y);run(ka,N/2); /* reads first half via parent */
    { int *h=malloc(N*2);CK(clEnqueueReadBuffer(q,Y,1,0,N*2,h,0,0,0));size_t bad=0;for(size_t i=0;i<N/2;i++)bad+=h[i]!=8;
      if(bad){printf("FAIL sub-buffer RAW            %zu wrong\n",bad);fails++;} free(h);}
    clReleaseMemObject(A);clReleaseMemObject(B);clReleaseMemObject(P);
    /* Chain through many buffers, interleaved with independent work */
    cl_mem ch[8];for(int i=0;i<8;i++)ch[i]=buf(N);
    v=100;clSetKernelArg(ksf,0,sizeof ch[0],&ch[0]);clSetKernelArg(ksf,1,4,&v);run(ksf,N);
    clSetKernelArg(ka,0,sizeof ch[0],&ch[0]);clSetKernelArg(ka,1,sizeof ch[1],&ch[1]);run(ka,N);
    fails+=check(ch[1],N,e101,"RAW chain");
    for(int i=0;i<8;i++)clReleaseMemObject(ch[i]);
    /* Scratch: many concurrent independent kernels using private arrays */
    cl_mem out[16];for(int i=0;i<16;i++){out[i]=buf(S);int s=5;clSetKernelArg(kscr,0,sizeof out[i],&out[i]);clSetKernelArg(kscr,1,4,&s);run(kscr,S);}
    for(int i=0;i<16;i++){fails+=check(out[i],S,escr,"scratch concurrent");clReleaseMemObject(out[i]);}
    /* Overflow of the tracking table: 100 independent buffers, then read all */
    cl_mem many[100];for(int i=0;i<100;i++){many[i]=buf(S);v=7;clSetKernelArg(kf,0,sizeof many[i],&many[i]);clSetKernelArg(kf,1,4,&v);run(kf,S);}
    v=1;clSetKernelArg(ksf,0,sizeof many[0],&many[0]);clSetKernelArg(ksf,1,4,&v);run(ksf,S);
    for(int i=1;i<100;i++){clSetKernelArg(ka,0,sizeof many[0],&many[0]);clSetKernelArg(ka,1,sizeof many[i],&many[i]);run(ka,S);}
    for(int i=1;i<100;i++)fails+=check(many[i],S,e2,"RAW after table overflow");
    for(int i=0;i<100;i++)clReleaseMemObject(many[i]);
    /* Fill buffer (driver helper kernel) after a slow writer, then consume */
    v=1;clSetKernelArg(ksf,0,sizeof X,&X);clSetKernelArg(ksf,1,4,&v);run(ksf,N);
    int pat=8;CK(clEnqueueFillBuffer(q,X,&pat,4,0,N*4,0,0,0));
    fails+=check(X,N,e8,"WAW vs clEnqueueFillBuffer");
    /* Copy buffer (helper) consuming a slow writer's output */
    v=7;clSetKernelArg(ksf,0,sizeof X,&X);clSetKernelArg(ksf,1,4,&v);run(ksf,N);
    CK(clEnqueueCopyBuffer(q,X,Z,0,0,N*4,0,0,0));
    fails+=check(Z,N,e7,"RAW vs clEnqueueCopyBuffer");
    (void)e3;
  }
  printf("%s: %d failures over %d reps\n",fails?"FAIL":"PASS",fails,reps);
  return fails?1:0;}
