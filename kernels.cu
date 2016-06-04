#include "kernels.cuh"


__global__ void simple_kernel(float *img, const int width, const int height, const int depth){
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  if( x >= width || y >= height || z >= depth )
    return;
  img[x + (y+height*z)*width] = 0;
}