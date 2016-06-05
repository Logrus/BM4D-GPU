#include "kernels.cuh"

#define BLOCK_SIZE 1
#define divideup(x,y) (1 + (((x) - 1) / (y)))


__global__ void k_simple_kernel(uchar *img, const uint3 size){
  int x = threadIdx.x;//(blockDim.x * blockIdx.x) + threadIdx.x;
  int y = blockIdx.y;//(blockDim.y * blockIdx.y) + threadIdx.y;
  int z = 1;//(blockDim.z * blockIdx.z) + threadIdx.z;
  if( x >= size.x || y >= size.y || z >= size.z )
   return;

  img[x+(y)*size.x] *= 2;

}

__global__ void k_block_matching(uchar *img,
                                const uint3 size,
                                int k,
                                int maxN,
                                int window_size,
                                int th
                                )
{
  int x = threadIdx.x;//(blockDim.x * blockIdx.x) + threadIdx.x;
  int y = blockIdx.y;//(blockDim.y * blockIdx.y) + threadIdx.y;
  int z = 1;//(blockDim.z * blockIdx.z) + threadIdx.z;
  if( x >= size.x || y >= size.y || z >= size.z )
   return;

  for(int k=-window_size;k<=window_size;++k)
  for(int j=-window_size;j<=window_size;++j)
  for(int i=-window_size;i<=window_size;++i){
    float w(0);
    for (int pz=0;pz<k;pz++)
    for (int py=0;py<k;py++)
    for (int px=0;px<k;px++){
          int tx = x + i;
          int ty = y + j;
          int tz = k;
          int fx = max(0, min(tx+px, size.x-1));
          int fy = max(0, min(ty+py, size.y-1));
          int fz = max(0, min(tz+py, size.z-1));
          float diff = img[x+(y+fz*size.y)*size.x]-img[fx+(fy+fz*size.y)*size.x];
          w += diff*diff;
    }
    printf("Diff %d\n", w);
  }
  //printf("%d", x+(y+z*size.y)*size.x);
}

void run_block_matching(uchar *d_noisy_volume,
                      const uint3 size,
                      const Parameters params)
{
  dim3 block(4,4,4);
  dim3 grid(1,1,1);
  for (int i=0; i<1; i++){
    k_block_matching<<<grid,block>>>(&d_noisy_volume[size.x*size.y*i],
                                    size,
                                    params.patch_size,
                                    params.maxN,
                                    params.window_size,
                                    params.sim_th
      );
    cudaDeviceSynchronize();
  }
  checkCudaErrors(cudaGetLastError());
}
