/* GPU bandwidth and host read speed: write-combined (default) vs write-back
 * (CL_MEM_ALLOC_HOST_PTR) buffers. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){printf("err %d line %d\n",e_,__LINE__);exit(1);}}while(0)
static const char*src="kernel void copy(global const float4*a,global float4*b){size_t i=get_global_id(0);b[i]=a[i]*1.0001f;}\n"
 "kernel void gather(global const int*a,global int*b,int mask){size_t i=get_global_id(0);b[i]=a[(i*2654435761u)&mask];}";
int main(){
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(clBuildProgram(pr,1,&d,"",0,0));
 cl_kernel kc=clCreateKernel(pr,"copy",&e),kg=clCreateKernel(pr,"gather",&e);
 size_t n=64<<20;cl_mem_flags fl[]={0,CL_MEM_ALLOC_HOST_PTR};const char*nm[]={"default (WC)","ALLOC_HOST_PTR (WB)"};
 char*h=malloc(n);
 for(int r=0;r<2;r++)for(int i=0;i<2;i++){
  cl_mem a=clCreateBuffer(ctx,fl[i],n,0,&e),b=clCreateBuffer(ctx,fl[i],n,0,&e);float z=1;
  clEnqueueFillBuffer(q,a,&z,4,0,n,0,0,0);clEnqueueFillBuffer(q,b,&z,4,0,n,0,0,0);clFinish(q);
  clSetKernelArg(kc,0,sizeof a,&a);clSetKernelArg(kc,1,sizeof b,&b);size_t g=n/16;
  clEnqueueNDRangeKernel(q,kc,1,0,&g,0,0,0,0);clFinish(q);
  double t=now();for(int k=0;k<10;k++)clEnqueueNDRangeKernel(q,kc,1,0,&g,0,0,0,0);clFinish(q);double bw=10.0*2*n/(now()-t)/1e9;
  int mask=(int)(n/4-1);clSetKernelArg(kg,0,sizeof a,&a);clSetKernelArg(kg,1,sizeof b,&b);clSetKernelArg(kg,2,4,&mask);g=n/4;
  clEnqueueNDRangeKernel(q,kg,1,0,&g,0,0,0,0);clFinish(q);
  t=now();for(int k=0;k<5;k++)clEnqueueNDRangeKernel(q,kg,1,0,&g,0,0,0,0);clFinish(q);double gt=(now()-t)/5*1e3;
  t=now();CK(clEnqueueReadBuffer(q,b,CL_TRUE,0,n,h,0,0,0));double rb=n/(now()-t)/1e9;
  t=now();CK(clEnqueueWriteBuffer(q,b,CL_TRUE,0,n,h,0,0,0));double wbw=n/(now()-t)/1e9;
  printf("%-20s GPU copy %6.1f GB/s | GPU random gather %6.2f ms | ReadBuffer %6.2f GB/s | WriteBuffer %6.2f GB/s\n",nm[i],bw,gt,rb,wbw);
  clReleaseMemObject(a);clReleaseMemObject(b);}
 return 0;}
