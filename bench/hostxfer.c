/* Host<->buffer transfer speed for default (write-combined on Asahi) vs
 * CL_MEM_ALLOC_HOST_PTR (write-back) buffers, in the pattern applications use:
 * GPU writes, host reads (ReadBuffer / map READ); host writes, GPU reads
 * (WriteBuffer / map WRITE_INVALIDATE). Host memory is pre-faulted. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){printf("err %d line %d\n",e_,__LINE__);exit(1);}}while(0)
static const char*src="kernel void touch(global int*a){size_t i=get_global_id(0);a[i]=a[i]+1;}";
int main(int argc,char**argv){
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(clBuildProgram(pr,1,&d,"",0,0));cl_kernel k=clCreateKernel(pr,"touch",&e);
 size_t sizes[]={64<<10,4<<20,64<<20};char*h=malloc(64<<20);memset(h,1,64<<20);
 cl_mem_flags fl[]={0,CL_MEM_ALLOC_HOST_PTR};const char*nm[]={"default","ALLOC_HOST_PTR"};
 printf("%-15s %8s | %9s %9s | %9s %9s   (GB/s, GPU touches the buffer between transfers)\n","flags","size","ReadBuf","MapRead","WriteBuf","MapWrInv");
 for(int i=0;i<2;i++)for(int s=0;s<3;s++){
  size_t n=sizes[s];int reps=s==2?8:50;cl_mem b=clCreateBuffer(ctx,CL_MEM_READ_WRITE|fl[i],n,0,&e);CK(e);
  clSetKernelArg(k,0,sizeof b,&b);size_t g=n/4;int z=0;clEnqueueFillBuffer(q,b,&z,4,0,n,0,0,0);clFinish(q);
  double tr=0,tm=0,tw=0,ti=0,t;
  for(int r=0;r<reps;r++){
   clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,0);clFinish(q);
   t=now();CK(clEnqueueReadBuffer(q,b,CL_TRUE,0,n,h,0,0,0));tr+=now()-t;
   clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,0);clFinish(q);
   t=now();char*m=clEnqueueMapBuffer(q,b,CL_TRUE,CL_MAP_READ,0,n,0,0,0,&e);CK(e);memcpy(h,m,n);clEnqueueUnmapMemObject(q,b,m,0,0,0);clFinish(q);tm+=now()-t;
   t=now();CK(clEnqueueWriteBuffer(q,b,CL_TRUE,0,n,h,0,0,0));tw+=now()-t;
   clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,0);clFinish(q);
   t=now();m=clEnqueueMapBuffer(q,b,CL_TRUE,CL_MAP_WRITE_INVALIDATE_REGION,0,n,0,0,0,&e);CK(e);memcpy(m,h,n);clEnqueueUnmapMemObject(q,b,m,0,0,0);clFinish(q);ti+=now()-t;
   clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,0);clFinish(q);}
  double f=(double)n*reps/1e9;
  printf("%-15s %8zu | %9.2f %9.2f | %9.2f %9.2f\n",nm[i],n,f/tr,f/tm,f/tw,f/ti);clReleaseMemObject(b);}
 return 0;}
