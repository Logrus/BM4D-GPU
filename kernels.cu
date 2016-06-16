#include "kernels.cuh"

#define BLOCK_SIZE 1
#define divideup(x,y) (1 + (((x) - 1) / (y)))

__device__
void add_stack(
	uint3float1 *d_stacks,
	uint *d_nstacks,
	const uint3float1 val,	
	const int maxN
	)
{
	int k;
	uint num = (*d_nstacks);
	if (num < maxN) //add new value
	{
		k = num++;
		while (k > 0 && val.val > d_stacks[k-1].val)
		{
			d_stacks[k] = d_stacks[k - 1];
			--k;
		}

		d_stacks[k] = val;
		*d_nstacks = num;
	}
	else if (val.val >= d_stacks[0].val) return;
	else //delete highest value and add new
	{
		k = 1;
		while (k < maxN && val.val < d_stacks[k].val)
		{
			d_stacks[k - 1] = d_stacks[k];
			k++;
		}
		d_stacks[k - 1] = val;
	}
}

__device__ float dist(uchar *img, uint3 size, uint3 ref, uint3 cmp, int k){
	float diff(0);
	for (int z = 0; z < k; ++z)
		for (int y = 0; y < k; ++y)
			for (int x = 0; x < k; ++x){
				//float w(0);
				//for (int pz = 0; pz<k; pz++)
				//	for (int py = 0; py<k; py++)
				//		for (int px = 0; px<k; px++){
				//			int tx = x + i;
				//			int ty = y + j;
				//			int tz = k;
				//			int fx = max(0, min(tx + px, size.x - 1));
				//			int fy = max(0, min(ty + py, size.y - 1));
				//			int fz = max(0, min(tz + py, size.z - 1));
				//			float diff = img[x + (y + fz*size.y)*size.x] - img[fx + (fy + fz*size.y)*size.x];
				float tmp = (img[(ref.x + x) + (ref.y + y)*size.x + (ref.z + z)*size.x*size.y] - img[(cmp.x + x) + (cmp.y + y)*size.x + (cmp.z + z)*size.x*size.y]);
				diff += tmp*tmp;
			}
	return diff;
}

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
                                int th,
								uint3float1 *d_stacks,
								uint *d_nstacks
                                )
{
  int x = (blockDim.x * blockIdx.x) + threadIdx.x;
  int y = (blockDim.y * blockIdx.y) + threadIdx.y;
  int z = (blockDim.z * blockIdx.z) + threadIdx.z;
  if( x >= size.x || y >= size.y || z >= size.z )
   return;
  uint3 ref = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);
  uint3 cmp = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);
  float w = dist(img, size, ref, cmp, k);
  if (w < th){
	  add_stack(
		  &d_stacks[maxN * idx3(cmp.x, cmp.y, cmp.z, size.x, size.y)],
		  &d_nstacks[idx3(cmp.x, cmp.y, cmp.z, size.x, size.y)],
		  uint3float1(cmp.x, cmp.y, cmp.z, w),
		  maxN
		  );
	  printf("Stack ref: %d %d %d, cmp: %d %d %d, diff %f\n", ref.x, ref.y, ref.z, cmp.x, cmp.y, cmp.z, w);
  }
  //printf("%d", x+(y+z*size.y)*size.x);
}

void run_block_matching(uchar *d_noisy_volume,
                      const uint3 size,
                      const Parameters params,
					  uint3float1 *d_stacks,
					  uint *d_nstacks
					  )
{
	unsigned int line = size.x / 8;
	dim3 grid(line, 1, 1);
	dim3 block(32,32,1);
	std::cout << "Grid x: " << grid.x << " y: " << grid.y << " z: " << grid.z << std::endl;
	std::cout << "Block x: " << block.x << " y: " << block.y << " z: " << block.z << std::endl;
	std::cout << "Warps: " << 8 * 8 * 8 / 32 << std::endl;
	std::cout << "Treads per block: " << 8 * 8 * 8 << std::endl;
	std::cout << "Total threads: " << 8 * 8 * 8 * line << std::endl;
	k_block_matching<<<grid,block>>>(d_noisy_volume,
									size,
									params.patch_size,
									params.maxN,
									params.window_size,
									params.sim_th,
									d_stacks,
									d_nstacks
		);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());
}
