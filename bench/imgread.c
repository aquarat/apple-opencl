/* Readback speed of a GPU-written image (darktable's pipeline buffers are
 * RGBA float images). */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e_=(x);if(e_){printf("err %d line %d\n",e_,__LINE__);exit(1);}}while(0)
static const char*src="kernel void w(write_only image2d_t o){int2 p=(int2)(get_global_id(0),get_global_id(1));write_imagef(o,p,(float4)(p.x,p.y,1,1));}";
int main(int argc,char**argv){
 cl_platform_id p;cl_device_id d;CK(clGetPlatformIDs(1,&p,0));CK(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0));
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);CK(clBuildProgram(pr,1,&d,"",0,0));cl_kernel k=clCreateKernel(pr,"w",&e);
 size_t W=argc>1?atoi(argv[1]):6048,H=argc>2?atoi(argv[2]):4024;
 cl_image_format f={CL_RGBA,CL_FLOAT};cl_image_desc dd={0};dd.image_type=CL_MEM_OBJECT_IMAGE2D;dd.image_width=W;dd.image_height=H;
 cl_mem img=clCreateImage(ctx,CL_MEM_READ_WRITE,&f,&dd,0,&e);CK(e);clSetKernelArg(k,0,sizeof img,&img);
 size_t n=W*H*16;float*h=malloc(n);memset(h,0,n);size_t g[2]={W,H},o[3]={0,0,0},r[3]={W,H,1};
 for(int i=0;i<3;i++){CK(clEnqueueNDRangeKernel(q,k,2,0,g,0,0,0,0));clFinish(q);
  double t=now();CK(clEnqueueReadImage(q,img,CL_TRUE,o,r,0,0,h,0,0,0));double dt=now()-t;
  double t2=now();CK(clEnqueueWriteImage(q,img,CL_TRUE,o,r,0,0,h,0,0,0));double dt2=now()-t2;
  size_t bad=0;for(size_t y=0;y<H;y++)for(size_t x=0;x<W;x++){float*px=h+4*(y*W+x);if(px[0]!=x||px[1]!=y||px[2]!=1||px[3]!=1)bad++;}
  printf("%zux%zu RGBA32F (%zu MB): ReadImage %.1f ms (%.2f GB/s)  WriteImage %.1f ms (%.2f GB/s)  bad pixels %zu\n",W,H,n>>20,dt*1e3,n/dt/1e9,dt2*1e3,n/dt2/1e9,bad);}
 /* sub-regions: odd origin and size, above and below the staging threshold */
 size_t regs[][4]={{13,7,333,211},{1,1,40,30},{W-100,H-50,100,50},{0,0,W,1}};
 CK(clEnqueueNDRangeKernel(q,k,2,0,g,0,0,0,0));clFinish(q);
 for(int i=0;i<4;i++){size_t o2[3]={regs[i][0],regs[i][1],0},r2[3]={regs[i][2],regs[i][3],1};float*b=malloc(r2[0]*r2[1]*16);
  CK(clEnqueueReadImage(q,img,CL_TRUE,o2,r2,0,0,b,0,0,0));size_t bad=0;
  for(size_t y=0;y<r2[1];y++)for(size_t x=0;x<r2[0];x++){float*px=b+4*(y*r2[0]+x);if(px[0]!=o2[0]+x||px[1]!=o2[1]+y)bad++;}
  printf("region %zu,%zu %zux%zu (%zu KB): bad pixels %zu\n",o2[0],o2[1],r2[0],r2[1],r2[0]*r2[1]*16>>10,bad);free(b);}
 return 0;}
