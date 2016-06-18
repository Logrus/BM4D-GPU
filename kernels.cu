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

__global__ void k_block_matching(
								const uint3 center,
								uchar *img,
								uchar *out,
                                const uint3 size,
                                int k,
                                int maxN,
                                int window_size,
                                int th,
								uint3float1 *d_stacks,
								uint *d_nstacks
                                )
{
  int lx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int ly = (blockDim.y * blockIdx.y) + threadIdx.y;
  int lz = (blockDim.z * blockIdx.z) + threadIdx.z;
  uint halfx = window_size / 2;
  uint halfy = window_size / 2;
  uint halfz = window_size / 2;
  int gx = lx + center.x - halfx;
  int gy = ly + center.y - halfy;
  int gz = lz + center.z - halfz;
  if( gx >= size.x || gy >= size.y || gz >= size.z || gx < 0 || gy < 0 || gz < 0 )
   return;
  uint3 ref = make_uint3(center.x, center.y, center.z);
  uint3 cmp = make_uint3(gx, gy, gz);
  float w = dist(img, size, ref, cmp, k);
  img[(gx)+(gy)*size.x + (gz)*size.x*size.y] = w;
  /*if (w < th){
	  add_stack(
		  &d_stacks[maxN * idx3(cmp.x, cmp.y, cmp.z, size.x, size.y)],
		  &d_nstacks[idx3(cmp.x, cmp.y, cmp.z, size.x, size.y)],
		  uint3float1(cmp.x, cmp.y, cmp.z, w),
		  maxN
		  );
	  printf("Stack ref: %d %d %d, cmp: %d %d %d, diff %f\n", ref.x, ref.y, ref.z, cmp.x, cmp.y, cmp.z, w);
  }*/
  //printf("%d", x+(y+z*size.y)*size.x);
}

void run_block_matching(uchar *d_noisy_volume,
					                 uchar *out,
                      const uint3 size,
                      const Parameters params,
					                 uint3float1 *d_stacks,
					                 uint *d_nstacks
					  )
{
	dim3 block(16, 16, 1);
	dim3 grid(params.window_size / block.x, params.window_size / block.y, params.window_size);
	std::cout << "Grid x: " << grid.x << " y: " << grid.y << " z: " << grid.z << std::endl;
	std::cout << "Block x: " << block.x << " y: " << block.y << " z: " << block.z << std::endl;
	std::cout << "Warps: " << block.x * block.y * block.z / 32 << std::endl;
	std::cout << "Treads per block: " << block.x * block.y * block.z << std::endl;
 for (int y = 0; y < size.y-params.patch_size; ++y)
   for (int x = 0; x < size.x - params.patch_size; ++x){
     //std::cout << "Computing x: " << x << " y " << y << std::endl;
       k_block_matching << <grid, block >> >(
         make_uint3(x, y, 50),
         d_noisy_volume,
         out,
         size,
         params.patch_size,
         params.maxN,
         params.window_size,
         params.sim_th,
         d_stacks,
         d_nstacks
         );
   }

	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());
}
