/* How much does loop overhead cost on AGX? Same FMA work, loop body
 * unrolled 1x/4x/8x by hand, plus "#pragma unroll 8" (clang at -O0 in
 * Rusticl: does the pragma survive?). */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){printf("err %d line %d\n",e_,__LINE__);exit(1);}}while(0)
static const char *src =
"#define B4 x=fma(x,a,b);y=fma(y,a,b);z=fma(z,a,b);w=fma(w,a,b);\n"
"#define HEAD float a=p+1e-7f,b=p*0.5f,x=get_local_id(0),y=x+1,z=x+2,w=x+3;\n"
"kernel void u1(global float*o,float p,int n){HEAD for(int i=0;i<n;i++){B4} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void u4(global float*o,float p,int n){HEAD for(int i=0;i<n;i+=4){B4 B4 B4 B4} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void u8(global float*o,float p,int n){HEAD for(int i=0;i<n;i+=8){B4 B4 B4 B4 B4 B4 B4 B4} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void pragma8(global float*o,float p,int n){HEAD\n#pragma unroll 8\n for(int i=0;i<n;i++){B4} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void chk4(global float*o,float p,int n){HEAD int i=0; for(;;){ if(i>=n)break; B4 i++; if(i>=n)break; B4 i++; if(i>=n)break; B4 i++; if(i>=n)break; B4 i++;} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void chk8(global float*o,float p,int n){HEAD int i=0; for(;;){ if(i>=n)break; B4 i++; if(i>=n)break; B4 i++; if(i>=n)break; B4 i++; if(i>=n)break; B4 i++; if(i>=n)break; B4 i++; if(i>=n)break; B4 i++; if(i>=n)break; B4 i++; if(i>=n)break; B4 i++;} o[get_global_id(0)]=x+y+z+w;}\n"
"kernel void const1(global float*o,float p,int n){HEAD for(int i=0;i<4096;i++){B4} o[get_global_id(0)]=x+y+z+w;}\n";
int main(){
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);
 if(clBuildProgram(pr,1,&d,"",0,0)){char l[8192];clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,8192,l,0);puts(l);return 1;}
 size_t g=1<<20,l=256;cl_mem o=clCreateBuffer(ctx,0,g*4,0,&e);float pv=1.0f;int n=4096;
 const char*nm[]={"u1","u4","u8","pragma8","chk4","chk8","const1"};
 for(int k=0;k<7;k++){cl_kernel kk=clCreateKernel(pr,nm[k],&e);CK(e);clSetKernelArg(kk,0,sizeof o,&o);clSetKernelArg(kk,1,4,&pv);clSetKernelArg(kk,2,4,&n);
  clEnqueueNDRangeKernel(q,kk,1,0,&g,&l,0,0,0);clFinish(q);double best=1e9;
  for(int r=0;r<5;r++){double t=now();clEnqueueNDRangeKernel(q,kk,1,0,&g,&l,0,0,0);clFinish(q);t=now()-t;if(t<best)best=t;}
  printf("%-8s %7.0f GFLOPS\n",nm[k],(double)g*n*4*2/best/1e9);}
 return 0;}
