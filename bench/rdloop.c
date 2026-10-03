/* clEnqueueReadBuffer throughput in a loop, for profiling the host copy:
 * ./rdloop 0 = default buffer, ./rdloop 1 = CL_MEM_ALLOC_HOST_PTR. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int c,char**v){cl_platform_id p;cl_device_id d;clGetPlatformIDs(1,&p,0);clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0);
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 size_t n=64<<20;cl_mem b=clCreateBuffer(ctx,atoi(v[1])?CL_MEM_ALLOC_HOST_PTR:0,n,0,&e);float z=1;clEnqueueFillBuffer(q,b,&z,4,0,n,0,0,0);
 char*h=malloc(n);memset(h,1,n);double t=now();for(int i=0;i<20;i++)clEnqueueReadBuffer(q,b,CL_TRUE,0,n,h,0,0,0);
 printf("%s: %.2f GB/s\n",atoi(v[1])?"WB":"WC",20.0*n/(now()-t)/1e9);return 0;}
