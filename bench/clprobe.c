/* First look at a device: kernel build time, empty-kernel launch rate,
 * kernel + clFinish round trip, copy bandwidth, a dependent-FMA loop and a
 * naive tiled SGEMM. Quick sanity numbers, not peak figures. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e=(x);if(e){printf("err %d at %s:%d\n",e,__FILE__,__LINE__);exit(1);}}while(0)
static const char*src=
"kernel void empty(global float*a){}\n"
"kernel void copy(global const float4*a,global float4*b){size_t i=get_global_id(0);b[i]=a[i];}\n"
"kernel void fma_loop(global float*o){float x=get_global_id(0)*1e-6f,y=1.0001f,z=0.5f,w=0.25f;\n"
" for(int i=0;i<4096;i++){x=fma(x,y,z);w=fma(w,y,z);z=fma(z,y,x);y=fma(y,0.9999f,w);} o[get_global_id(0)]=x+y+z+w;}\n"
"#define T 16\n"
"kernel void sgemm(int N,global const float*A,global const float*B,global float*C){\n"
" local float As[T][T],Bs[T][T];int r=get_local_id(1),c=get_local_id(0),gr=get_global_id(1),gc=get_global_id(0);float acc=0;\n"
" for(int k=0;k<N;k+=T){As[r][c]=A[gr*N+k+c];Bs[r][c]=B[(k+r)*N+gc];barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for(int j=0;j<T;j++)acc+=As[r][j]*Bs[j][c];barrier(CLK_LOCAL_MEM_FENCE);} C[gr*N+gc]=acc;}\n";
int main(){
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);CK(e);
 cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);CK(e);
 double t=now();cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(e);
 e=clBuildProgram(pr,1,&d,"-cl-fast-relaxed-math",0,0);
 if(e){char log[8192];clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,sizeof log,log,0);puts(log);return 1;}
 printf("build: %.1f ms\n",(now()-t)*1e3);
 size_t n=64<<20; /* 64M floats = 256MB */
 cl_mem a=clCreateBuffer(ctx,CL_MEM_READ_WRITE,n*4,0,&e);CK(e);cl_mem b=clCreateBuffer(ctx,CL_MEM_READ_WRITE,n*4,0,&e);CK(e);
 float z=1;CK(clEnqueueFillBuffer(q,a,&z,4,0,n*4,0,0,0));clFinish(q);
 cl_kernel ke=clCreateKernel(pr,"empty",&e);CK(e);clSetKernelArg(ke,0,sizeof a,&a);
 size_t g=64;for(int i=0;i<10;i++)clEnqueueNDRangeKernel(q,ke,1,0,&g,0,0,0,0);clFinish(q);
 int K=2000;t=now();for(int i=0;i<K;i++)CK(clEnqueueNDRangeKernel(q,ke,1,0,&g,0,0,0,0));clFinish(q);
 printf("empty kernel, %d back-to-back: %.1f us/launch\n",K,(now()-t)/K*1e6);
 t=now();for(int i=0;i<200;i++){clEnqueueNDRangeKernel(q,ke,1,0,&g,0,0,0,0);clFinish(q);}
 printf("empty kernel + clFinish round trip: %.1f us\n",(now()-t)/200*1e6);
 cl_kernel kc=clCreateKernel(pr,"copy",&e);CK(e);clSetKernelArg(kc,0,sizeof a,&a);clSetKernelArg(kc,1,sizeof b,&b);
 g=n/4;clEnqueueNDRangeKernel(q,kc,1,0,&g,0,0,0,0);clFinish(q);
 t=now();for(int i=0;i<20;i++)clEnqueueNDRangeKernel(q,kc,1,0,&g,0,0,0,0);clFinish(q);
 printf("copy kernel bandwidth: %.1f GB/s (r+w)\n",20.0*n*8/(now()-t)/1e9);
 cl_kernel kf=clCreateKernel(pr,"fma_loop",&e);CK(e);clSetKernelArg(kf,0,sizeof b,&b);
 g=1<<20;clEnqueueNDRangeKernel(q,kf,1,0,&g,0,0,0,0);clFinish(q);
 t=now();for(int i=0;i<5;i++)clEnqueueNDRangeKernel(q,kf,1,0,&g,0,0,0,0);clFinish(q);
 printf("fp32 FMA peak-ish: %.0f GFLOPS\n",5.0*g*4096*4*2/(now()-t)/1e9);
 int N=2048;cl_kernel kg=clCreateKernel(pr,"sgemm",&e);CK(e);cl_mem c=clCreateBuffer(ctx,CL_MEM_READ_WRITE,(size_t)N*N*4,0,&e);
 clSetKernelArg(kg,0,4,&N);clSetKernelArg(kg,1,sizeof a,&a);clSetKernelArg(kg,2,sizeof a,&a);clSetKernelArg(kg,3,sizeof c,&c);
 size_t gg[2]={N,N},lg[2]={16,16};CK(clEnqueueNDRangeKernel(q,kg,2,0,gg,lg,0,0,0));clFinish(q);
 t=now();for(int i=0;i<5;i++)clEnqueueNDRangeKernel(q,kg,2,0,gg,lg,0,0,0);clFinish(q);
 printf("naive tiled sgemm 2048: %.0f GFLOPS\n",5.0*2*N*(double)N*N/(now()-t)/1e9);
 return 0;}
