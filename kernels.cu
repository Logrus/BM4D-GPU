#include "kernels.cuh"

#define BLOCK_SIZE 1
#define divideup(x,y) (1 + (((x) - 1) / (y)))


__global__ void simple_kernel(float *img, const uint3 size){
  int x = (blockDim.x * blockIdx.x) + threadIdx.x;
  int y = (blockDim.y * blockIdx.y) + threadIdx.y;
  int z = (blockDim.z * blockIdx.z) + threadIdx.z;
  if( x >= size.x || y >= size.y || z >= size.x )
   return;

  img[x+y*size.x+x*size.y*size.x] = 0;
}

void run_block_matching(float *d_noisy_volume,
                      const uint3 size,
                      const Parameters params)
{
  dim3 block(size.x,1,1);
  dim3 grid(1,size.y,1);
  for (int i=0; i<size.z; i++)
    simple_kernel<<<grid,block>>>((float *)&d_noisy_volume[size.x*size.y*i], size);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
