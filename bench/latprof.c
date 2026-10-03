/* Where does a kernel round trip go? Profiling timestamps (device clock)
 * against the host clock via clGetDeviceAndHostTimer. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#define CK(x) do{cl_int e_=(x);if(e_){printf("err %d line %d\n",e_,__LINE__);exit(1);}}while(0)
static const char*src="kernel void w(global int*a){a[get_global_id(0)]+=1;}";
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
int main(int argc,char**argv){
 int N=argc>1?atoi(argv[1]):200,idle=argc>2?atoi(argv[2]):0;
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);
 cl_queue_properties pp[]={CL_QUEUE_PROPERTIES,CL_QUEUE_PROFILING_ENABLE,0};
 cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,pp,&e);CK(e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(clBuildProgram(pr,1,&d,"",0,0));cl_kernel k=clCreateKernel(pr,"w",&e);
 cl_mem b=clCreateBuffer(ctx,0,4096,0,&e);clSetKernelArg(k,0,sizeof b,&b);size_t g=32;
 double *qs=malloc(N*8),*ss=malloc(N*8),*ex=malloc(N*8),*nt=malloc(N*8),*tot=malloc(N*8);
 for(int i=0;i<N+5;i++){ if(idle)usleep(idle);
  cl_ulong d0,h0,h1,dq,ds,dst,den; cl_event ev;
  CK(clGetDeviceAndHostTimer(d,&d0,&h0));
  CK(clEnqueueNDRangeKernel(q,k,1,0,&g,0,0,0,&ev));CK(clFinish(q));
  CK(clGetHostTimer(d,&h1));
  clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_QUEUED,8,&dq,0);clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_SUBMIT,8,&ds,0);
  clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,8,&dst,0);clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,8,&den,0);
  clReleaseEvent(ev); if(i<5)continue; int j=i-5;
  /* device and host timers share a base after offsetting by d0-h0 */
  double off=(double)d0-(double)h0;
  qs[j]=(ds-dq)/1e3; ss[j]=(dst-ds)/1e3; ex[j]=(den-dst)/1e3; nt[j]=((double)h1+off-(double)den)/1e3; tot[j]=(h1-h0)/1e3;}
 double*arr[]={qs,ss,ex,nt,tot};const char*nm[]={"queued->submit","submit->GPU start","GPU start->end","GPU end->clFinish returns","total"};
 printf("idle %d us between iterations (median / p90, us)\n",idle);
 for(int a=0;a<5;a++){qsort(arr[a],N,8,cmp);printf("  %-26s %8.1f %8.1f\n",nm[a],arr[a][N/2],arr[a][N*9/10]);}
 return 0;}
