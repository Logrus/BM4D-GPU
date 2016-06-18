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
        int rx = max(0, min(x + ref.x, size.x - 1));
        int ry = max(0, min(y + ref.y, size.y - 1));
        int rz = max(0, min(z + ref.z, size.z - 1));
        int cx = max(0, min(x + cmp.x, size.x - 1));
        int cy = max(0, min(y + cmp.y, size.y - 1));
        int cz = max(0, min(z + cmp.z, size.z - 1));
        //printf("rx: %d ry: %d rz: %d cx: %d cy: %d cz: %d\n", rx, ry, rz, cx, cy, cz);
		      float tmp = (img[(rx) + (ry)*size.x + (rz)*size.x*size.y] - img[(cx) + (cy)*size.x + (cz)*size.x*size.y]);
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
								const int in_z,
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
  int x = (blockDim.x * blockIdx.x) + threadIdx.x;
  int y = (blockDim.y * blockIdx.y) + threadIdx.y;
  int z = in_z; // (blockDim.z * blockIdx.z) + threadIdx.z;
  if( x >= size.x || y >= size.y || z >= size.z || x < 0 || y < 0 || z < 0 )
   return;

  int wxb = fmaxf(0, x - window_size); // window x begin
  int wyb = fmaxf(0, y - window_size); // window y begin
  int wzb = fmaxf(0, z - window_size); // window z begin
  int wxe = fminf(size.x - 1, x + window_size); // window x end
  int wye = fminf(size.y - 1, y + window_size); // window y end
  int wze = fminf(size.z - 1, z + window_size); // window z end
  //printf("Scores:");
  for (int wz = wzb; wz <= wze; wz++)
    for (int wy = wyb; wy <= wye; wy++)
      for (int wx = wyb; wx <= wye; wx++){
        float w = dist(img, size, make_uint3(x,y,z), make_uint3(wx,wy,wz), k);
  //      printf("%f\t", w);
      }
 //printf("\n");

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
	dim3 grid(size.x / block.x, size.y / block.y, 1);
	std::cout << "Grid x: " << grid.x << " y: " << grid.y << " z: " << grid.z << std::endl;
	std::cout << "Block x: " << block.x << " y: " << block.y << " z: " << block.z << std::endl;
	std::cout << "Warps per block: " << block.x * block.y * block.z / 32 << std::endl;
	std::cout << "Treads per block: " << block.x * block.y * block.z << std::endl;
 std::cout << "Total threads: " << block.x*block.y*block.z*grid.x*grid.y*grid.z << std::endl;
 for (int z = 0; z < size.z; z++){
   std::cout << "Computing z: " << z << std::endl;
   k_block_matching << <grid, block >> >(
     z,
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
