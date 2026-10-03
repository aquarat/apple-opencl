/* Mixed-precision throughput: fp16 x fp16 + fp32 in a few spellings. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){printf("err %d line %d\n",e_,__LINE__);exit(1);}}while(0)
static const char *src =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#define R 1024\n"
"kernel void f32(global float*o,float a){float x=a+get_local_id(0),y=a,z=a*2,w=a*3; half h=(half)a,g=(half)(a*0.5f);\n"
" float hf=h,gf=g; for(int i=0;i<R;i++){x=fma(hf,gf,x);y=fma(hf,gf,y);z=fma(hf,gf,z);w=fma(hf,gf,w);} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void mp_cvt(global float*o,float a){float x=a+get_local_id(0),y=a,z=a*2,w=a*3; half h=(half)a,g=(half)(a*0.5f);\n"
" for(int i=0;i<R;i++){h=h*g; x=fma((float)h,(float)g,x);y=fma((float)h,(float)g,y);z=fma((float)h,(float)g,z);w=fma((float)h,(float)g,w);} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void mp_mad(global float*o,float a){float x=a+get_local_id(0),y=a,z=a*2,w=a*3; half h=(half)a,g=(half)(a*0.5f);\n"
" for(int i=0;i<R;i++){x=mad((float)h,(float)g,x);y=mad((float)h,(float)g,y);z=mad((float)h,(float)g,z);w=mad((float)h,(float)g,w);} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void mp_star(global float*o,float a){float x=a+get_local_id(0),y=a,z=a*2,w=a*3; half h=(half)a,g=(half)(a*0.5f);\n"
" for(int i=0;i<R;i++){x+=(float)h*(float)g;y+=(float)h*(float)g;z+=(float)h*(float)g;w+=(float)h*(float)g;} o[get_global_id(0)]=x+y+z+w;}\n";
int main(int argc,char**argv){
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);
 const char*opt=argc>1?argv[1]:"";
 if(clBuildProgram(pr,1,&d,opt,0,0)){char l[8192];clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,8192,l,0);puts(l);return 1;}
 size_t g=1<<20,l=256;cl_mem o=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,g*4,0,&e);float a=1.0001f;
 const char*names[]={"f32","mp_cvt","mp_mad","mp_star"};
 for(int k=0;k<4;k++){cl_kernel kk=clCreateKernel(pr,names[k],&e);CK(e);clSetKernelArg(kk,0,sizeof o,&o);clSetKernelArg(kk,1,4,&a);
  clEnqueueNDRangeKernel(q,kk,1,0,&g,&l,0,0,0);clFinish(q);double t=now();for(int i=0;i<5;i++)clEnqueueNDRangeKernel(q,kk,1,0,&g,&l,0,0,0);clFinish(q);
  printf("%-8s %7.0f GFLOPS\n",names[k],5.0*g*1024*4*2/(now()-t)/1e9);}
 return 0;}
