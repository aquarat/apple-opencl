/* Per-kernel cost of a chain of tiny dependent kernels that each write a
 * buffer (the per-kernel flush), and 64 independent one-workgroup dispatches
 * against the same work as one 64-workgroup dispatch (dispatch overlap).
 * Optional argument: loop iterations per work-item (default 20000). */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e=(x);if(e){printf("err %d line %d\n",e,__LINE__);exit(1);}}while(0)
static const char*src=
"kernel void w1(global int*a){if(get_global_id(0)==0)a[0]+=1;}\n"
"kernel void serial(global float*o,int iters){float x=get_global_id(0)*1e-6f;for(int i=0;i<iters;i++)x=fma(x,1.0001f,0.5f);o[get_global_id(0)]=x;}\n";
int main(int c,char**v){
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);
 cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(clBuildProgram(pr,1,&d,"",0,0));
 cl_mem a=clCreateBuffer(ctx,CL_MEM_READ_WRITE,1<<20,0,&e);int z=0;clEnqueueFillBuffer(q,a,&z,4,0,1<<20,0,0,0);clFinish(q);
 cl_kernel k=clCreateKernel(pr,"w1",&e);clSetKernelArg(k,0,sizeof a,&a);size_t g=32;
 for(int i=0;i<10;i++)clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,0);
 clFinish(q);
 int K=2000;double t=now();for(int i=0;i<K;i++)CK(clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,0));clFinish(q);
 int r;clEnqueueReadBuffer(q,a,1,0,4,&r,0,0,0);
 printf("dependent tiny writing kernels: %.1f us/kernel (counter=%d, expect %d)\n",(now()-t)/K*1e6,r,K+10);
 /* overlap: 64 dispatches of one workgroup each, independent outputs, vs one dispatch of 64 groups */
 cl_kernel s=clCreateKernel(pr,"serial",&e);int it=atoi(c>1?v[1]:"20000");
 cl_mem bufs[64];for(int i=0;i<64;i++)bufs[i]=clCreateBuffer(ctx,CL_MEM_READ_WRITE,4096,0,&e);
 cl_mem big=clCreateBuffer(ctx,CL_MEM_READ_WRITE,64*4096,0,&e);size_t l=64,gb=64*64;
 clSetKernelArg(s,1,4,&it);
 for(int rep=0;rep<2;rep++){
 clSetKernelArg(s,0,sizeof big,&big);t=now();CK(clEnqueueNDRangeKernel(q,s,1,0,&gb,&l,0,0,0));clFinish(q);double t1=now()-t;
 t=now();for(int i=0;i<64;i++){clSetKernelArg(s,0,sizeof bufs[i],&bufs[i]);CK(clEnqueueNDRangeKernel(q,s,1,0,&l,&l,0,0,0));}clFinish(q);double t2=now()-t;
 printf("one dispatch x64 groups: %.2f ms | 64 dispatches x1 group: %.2f ms (%.1fx)\n",t1*1e3,t2*1e3,t2/t1);}
 return 0;}
