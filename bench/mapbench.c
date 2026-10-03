/* Cost of clEnqueueMapBuffer/Unmap round trips by allocation flags and size,
 * and the CPU-side cost of touching the mapped pointer. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){printf("err %d line %d\n",e_,__LINE__);exit(1);}}while(0)
static const char*src="kernel void inc(global int*a){a[get_global_id(0)]+=1;}";
int main(){
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(clBuildProgram(pr,1,&d,"",0,0));cl_kernel k=clCreateKernel(pr,"inc",&e);
 struct{const char*n;cl_mem_flags f;}kinds[]={{"default",0},{"ALLOC_HOST_PTR",CL_MEM_ALLOC_HOST_PTR}};
 size_t sizes[]={4096,1<<20,64<<20};
 printf("%-15s %9s %14s %14s %14s\n","flags","size","map+unmap us","GB/s","kernel+map us");
 for(int ki=0;ki<2;ki++)for(int si=0;si<3;si++){
  size_t n=sizes[si];cl_mem b=clCreateBuffer(ctx,CL_MEM_READ_WRITE|kinds[ki].f,n,0,&e);CK(e);
  int z=0;clEnqueueFillBuffer(q,b,&z,4,0,n,0,0,0);clFinish(q);
  int reps=si==2?10:200;
  /* pure map/unmap, writing the whole buffer from the CPU */
  double t=now();
  for(int r=0;r<reps;r++){int*m=clEnqueueMapBuffer(q,b,CL_TRUE,CL_MAP_READ|CL_MAP_WRITE,0,n,0,0,0,&e);CK(e);
   m[0]+=1;clEnqueueUnmapMemObject(q,b,m,0,0,0);}
  clFinish(q);double tm=(now()-t)/reps;
  /* typical loop: kernel writes, host maps and reads one value */
  clSetKernelArg(k,0,sizeof b,&b);size_t g=n/4;t=now();
  for(int r=0;r<reps;r++){clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,0);
   int*m=clEnqueueMapBuffer(q,b,CL_TRUE,CL_MAP_READ,0,n,0,0,0,&e);CK(e);volatile int x=m[g-1];(void)x;clEnqueueUnmapMemObject(q,b,m,0,0,0);}
  clFinish(q);double tk=(now()-t)/reps;
  printf("%-15s %9zu %14.1f %14.2f %14.1f\n",kinds[ki].n,n,tm*1e6,n/tm/1e9,tk*1e6);
  clReleaseMemObject(b);}
 return 0;}
