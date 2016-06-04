#include "kernels.cuh"

#define BLOCK_SIZE 1
#define divideup(x,y) (1 + (((x) - 1) / (y)))


__global__ void simple_kernel(float *img, const int width, const int height, const int depth){
  int x = (blockDim.x * blockIdx.x) + threadIdx.x;
  int y = (blockDim.y * blockIdx.y) + threadIdx.y;
  int z = (blockDim.z * blockIdx.z) + threadIdx.z;
  if( x >= width || y >= height || z >= depth ){
   return;
  }
  for(int k=0;k<depth;k++)
  for(int j=0;j<height;j++)
  for(int i=0;i<width;i++)
    img[x+y*width+x*height*width] = 0;
}

void wrapper_simple_kernel(const std::vector<float> &h_result, std::vector<float> &out, int width, int height, int depth){
  int v_size = width*height*depth;

  float *d_noisy_volume;
  // Allocate device memory
  checkCudaErrors(cudaMalloc((void**) &d_noisy_volume, sizeof(float)*v_size));
  //checkCudaErrors(cudaMemcpy((void*) d_noisy_volume, (void*)h_result.data(), sizeof(float)*v_size, cudaMemcpyHostToDevice));


  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Warp size:                                     %d\n", prop.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
    printf("  Is used by X11? : %d\n", prop.kernelExecTimeoutEnabled);

  }
  //dim3 block(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
  dim3 block(8,8,8);
  unsigned int xb = divideup(width,block.x);
  unsigned int yb = divideup(height,block.y);
  unsigned int zb = divideup(depth,block.z);
  std::cout<<xb<<std::endl;
  std::cout<<yb<<std::endl;
  std::cout<<zb<<std::endl;

  dim3 grid(xb,yb,zb);

  //simple_kernel<<<1,1>>>(d_noisy_volume,width,height,depth);
  cudaDeviceSynchronize();
  //checkCudaErrors(cudaGetLastError());

  // Copy denoised image back to host
  checkCudaErrors(cudaMemcpy((void*)out.data(), (void*) d_noisy_volume, sizeof(float)*v_size, cudaMemcpyDeviceToHost));
  // Cleanup
  checkCudaErrors(cudaFree(d_noisy_volume));
}