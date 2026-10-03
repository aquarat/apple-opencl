/* GPU memory leak check for image readback, the CTS basic imagedim pattern:
 * a src image written by the host, copied to a dst image by a kernel (or by
 * clEnqueueCopyImage with NOKERNEL=1), dst read back, both released, over
 * every power-of-two size up to 8192x4096, four times. Prints Shmem (where
 * Asahi's GPU buffer objects live) after each pass: it must stay flat.
 * Before Mesa 817bbf4 the kernel variant grew ~0.77 GB per pass.
 * Run it capped: ../capped.sh 6G out.log ../with-cl.sh ./imgleak */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static long shmem_mb(void){FILE*f=fopen("/proc/meminfo","r");char l[256];long v=-1;while(fgets(l,sizeof l,f))if(!strncmp(l,"Shmem:",6)){v=atol(l+6)/1024;break;}fclose(f);return v;}
static const char*src="kernel void cp(read_only image2d_t s, write_only image2d_t d){int2 p=(int2)(get_global_id(0),get_global_id(1));write_imagef(d,p,read_imagef(s,p));}";
int main(void){setvbuf(stdout,0,_IOLBF,0);
 cl_platform_id p;cl_device_id d;clGetPlatformIDs(1,&p,0);clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,0);
 cl_int e;cl_context ctx=clCreateContext(0,1,&d,0,0,&e);cl_command_queue q=clCreateCommandQueueWithProperties(ctx,d,0,&e);
 cl_program pr=clCreateProgramWithSource(ctx,1,&src,0,&e);clBuildProgram(pr,1,&d,"",0,0);cl_kernel k=clCreateKernel(pr,"cp",&e);
 cl_image_format f={CL_RGBA,CL_UNORM_INT8};size_t HB=(size_t)8192*8192*4;unsigned char*h=malloc(HB);memset(h,7,HB);
 printf("start %ld MB\n",shmem_mb());int n=0;for(int pass=0;pass<4;pass++){
 for(size_t H=1;H<=8192;H<<=1)for(size_t W=1;W<=8192;W<<=1){if(W*H>(size_t)8192*4096)continue;
  cl_image_desc dd={0};dd.image_type=CL_MEM_OBJECT_IMAGE2D;dd.image_width=W;dd.image_height=H;
  cl_mem a=clCreateImage(ctx,CL_MEM_READ_WRITE,&f,&dd,0,&e),b=clCreateImage(ctx,CL_MEM_READ_WRITE,&f,&dd,0,&e);
  size_t o[3]={0},r[3]={W,H,1},g[2]={W,H};clEnqueueWriteImage(q,a,CL_FALSE,o,r,0,0,h,0,0,0);
  if(getenv("NOKERNEL")){clEnqueueCopyImage(q,a,b,o,o,r,0,0,0);}else{clSetKernelArg(k,0,8,&a);clSetKernelArg(k,1,8,&b);clEnqueueNDRangeKernel(q,k,2,0,g,0,0,0,0);}
  clEnqueueReadImage(q,b,CL_TRUE,o,r,0,0,h,0,0,0);clReleaseMemObject(a);clReleaseMemObject(b);
  if(0)printf("%3d images (last %zux%zu): shmem %ld MB\n",n,W,H,shmem_mb());}
 clFinish(q);printf("pass %d end (%d) %ld MB\n",pass,n,shmem_mb());}return 0;}
