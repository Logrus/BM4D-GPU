#include "bm4d.h"

__global__ void simple_kernel(float *img, const int width, const int height, const int depth){
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  if( x >= width || y >= height || z >= depth )
    return;
  img[x + (y+height*z)*width] += 10;
}

std::vector<unsigned char> BM4D::run_first_step(const std::vector<unsigned char> &noisy_volume, const int &width, int &height, int &depth){
  int v_size = width*height*depth;
  std::vector<unsigned char> result(v_size);
  std::vector<float> h_result(v_size);

  // Convert image to float
  for(int z=0;z<depth;++z)
    for(int y=0;y<height;++y)
      for(int x=0;x<width;++x)
        h_result[idx3(x,y,z,width,height)]=static_cast<float>(noisy_volume[idx3(x,y,z,width,height)]);

  // Allocate device memory
  cudaMalloc ((void**) &d_noisy_volume, sizeof(float)*v_size);
  cudaMemcpy ((void*) d_noisy_volume, (void*) h_result.data(), sizeof(float)*v_size, cudaMemcpyHostToDevice);

  block = {1,1, 1};
  grid = {16,16,16};

  simple_kernel<<<grid,block>>>(d_noisy_volume,width,height,depth);
  // Do block matching
  //run_block_matching();

  // Gather cubes together

  // Perform 3D DCT
  //run_dct3d();

  // Do WHT in 4th dim + Hard Thresholding + IWHT
  //run_wht_ht_iwht();

  // Perform inverse 3D DCT
  //run_idct3d();

  // Aggregate
  //run_aggregation();

  // Copy denoised image back to host
  cudaMemcpy ((void*) h_result.data(), (void*) d_noisy_volume, sizeof(float)*v_size, cudaMemcpyDeviceToHost);

  // Convert image to unsigned char
  for(int z=0;z<depth;++z)
    for(int y=0;y<height;++y)
      for(int x=0;x<width;++x)
        result[idx3(x,y,z,width,height)]=static_cast<unsigned char>(h_result[idx3(x,y,z,width,height)]);

  // Cleanup
  cudaFree(d_noisy_volume);

  return result;
}