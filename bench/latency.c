/* Round-trip latency of one tiny kernel + clFinish, and of a blocking
 * 4-byte read, with the GPU kept busy vs idle between iterations. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){printf("err %d line %d\n",e_,__LINE__);exit(1);}}while(0)
static const char*src="kernel void w(global int*a){a[get_global_id(0)]+=1;}";
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
int main(int argc,char**argv){
 int N=argc>1?atoi(argv[1]):300; int idle_us=argc>2?atoi(argv[2]):0;
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(clBuildProgram(pr,1,&d,"",0,0));cl_kernel k=clCreateKernel(pr,"w",&e);
 cl_mem b=clCreateBuffer(ctx,0,4096,0,&e);clSetKernelArg(k,0,sizeof b,&b);size_t g=32;int v;
 double*t=malloc(N*sizeof(double));
 for(int pass=0;pass<2;pass++){
  for(int i=0;i<N;i++){ if(idle_us)usleep(idle_us);
   double t0=now();
   if(pass==0){CK(clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,0));CK(clFinish(q));}
   else CK(clEnqueueReadBuffer(q,b,CL_TRUE,0,4,&v,0,0,0));
   t[i]=(now()-t0)*1e6;}
  qsort(t,N,sizeof(double),cmp);
  printf("%-22s idle %5d us: p10 %6.1f  p50 %6.1f  p90 %6.1f us\n",pass?"blocking 4B read":"kernel + clFinish",idle_us,t[N/10],t[N/2],t[N*9/10]);}
 return 0;}
