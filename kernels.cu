#include "kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <math.h>

#define PI 3.14159265359
#define BLOCK_SIZE 1
#define divideup(x,y) (1 + (((x) - 1) / (y)))

// Nearest lower power of 2
__device__ __inline__ uint flp2(uint x)
{
  x = x | (x >> 1);
  x = x | (x >> 2);
  x = x | (x >> 4);
  x = x | (x >> 8);
  x = x | (x >> 16);
  return x - (x >> 1);
}

__device__ void add_stack(uint3float1* d_stacks,
                          uint* d_nstacks,
                          const uint3float1 val,
                          const int maxN)
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

__device__ float dist(const uchar* __restrict img, const uint3 size, const uint3 ref, const uint3 cmp, const int k){
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

__global__ void k_block_matching(const uchar* __restrict img,
                                 const uint3 size,
                                 const uint3 tsize,
                                 const Parameters params,
								                         uint3float1* d_stacks,
								                         uint* d_nstacks,
                                 uchar* out)
{

  for (int Idz = blockDim.z * blockIdx.z + threadIdx.z; Idz < tsize.z; Idz += blockDim.z*gridDim.z)
    for (int Idy = blockDim.y * blockIdx.y + threadIdx.y; Idy < tsize.y; Idy += blockDim.y*gridDim.y)
      for (int Idx = blockDim.x * blockIdx.x + threadIdx.x; Idx < tsize.x; Idx += blockDim.x*gridDim.x)
  {

      int x = Idx * params.step_size;
      int y = Idy * params.step_size;
      int z = Idz * params.step_size;
      if (x >= size.x || y >= size.y || z >= size.z || x < 0 || y < 0 || z < 0)
        return;

      int wxb = fmaxf(0, x - params.window_size); // window x begin
      int wyb = fmaxf(0, y - params.window_size); // window y begin
      int wzb = fmaxf(0, z - params.window_size); // window z begin
      int wxe = fminf(size.x - 1, x + params.window_size); // window x end
      int wye = fminf(size.y - 1, y + params.window_size); // window y end
      int wze = fminf(size.z - 1, z + params.window_size); // window z end

      uint3 ref = make_uint3(x, y, z);

      for (int wz = wzb; wz <= wze; wz++)
        for (int wy = wyb; wy <= wye; wy++)
          for (int wx = wxb; wx <= wxe; wx++){
            float w = dist(img, size, ref, make_uint3(wx, wy, wz), params.patch_size);
            
            if (w < params.sim_th){
              add_stack(&d_stacks[(Idx + (Idy + Idz* tsize.y)*tsize.x)*params.maxN],
                &d_nstacks[Idx + (Idy + Idz* tsize.y)*tsize.x],
                uint3float1(wx, wy, wz, w),
                params.maxN);
            }
          }
    }
    
}


void run_block_matching(const uchar* __restrict d_noisy_volume,
                        const uint3 size,
                        const uint3 tsize,
                        const Parameters params,
					                   uint3float1 *d_stacks,
					                   uint *d_nstacks,
                        uchar* out)
{
	dim3 block(16, 16, 1);
 //dim3 grid(size.x / block.x / params.step_size, size.y / block.y / params.step_size, 1);
 dim3 grid(20, 20, 1);

 // Debug verification
 std::cout << "Total number of reference patches " << (tsize.x*tsize.y*tsize.z) << std::endl;

	std::cout << "Grid x: " << grid.x << " y: " << grid.y << " z: " << grid.z << std::endl;
	std::cout << "Block x: " << block.x << " y: " << block.y << " z: " << block.z << std::endl;
	std::cout << "Warps per block: " << block.x * block.y * block.z / 32 << std::endl;
	std::cout << "Treads per block: " << block.x * block.y * block.z << std::endl;
 std::cout << "Total threads: " << block.x*block.y*block.z*grid.x*grid.y*grid.z << std::endl;

 k_block_matching << <grid, block >> >(d_noisy_volume,
                                       size,
                                       tsize,
                                       params,
                                       d_stacks,
                                       d_nstacks,
                                       out);

 cudaDeviceSynchronize();
 checkCudaErrors(cudaGetLastError());
}

__global__ void k_nstack_to_pow(const uint* __restrict d_nstacks, uint* d_nstacks_pow, const int size){
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < size; i += blockDim.x*gridDim.x){
    if (i<size) 
      d_nstacks_pow[i] = flp2(d_nstacks[i]);
    //printf("Original: %d Stripped: %d\n", d_nstacks[i], flp2(d_nstacks[i]));
  }
}

__global__ void k_gather_cubes(const uchar* __restrict img,
                               const uint3 size,
                               const Parameters params,
                               const uint3float1* __restrict d_stacks,
                               const uint* __restrict d_nstacks,
                               const uint array_size,
                               float* d_gathered4dstack,
                               uint* d_nstacks_pow)
{
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < array_size; i += blockDim.x*gridDim.x){

    uint3float1 ref = d_stacks[i];
    int cube_size = params.patch_size*params.patch_size*params.patch_size;

    for (int z = 0; z < params.patch_size; ++z)
      for (int y = 0; y < params.patch_size; ++y)
        for (int x = 0; x < params.patch_size; ++x){

          int rx = max(0, min(x + ref.x, size.x - 1));
          int ry = max(0, min(y + ref.y, size.y - 1));
          int rz = max(0, min(z + ref.z, size.z - 1));

          int img_idx = (rx) + (ry)*size.x + (rz)*size.x*size.y;
          int stack_idx = i*cube_size + (x)+(y + z*params.patch_size)*params.patch_size;
          
          d_gathered4dstack[stack_idx] = img[img_idx];
        }

  }
}

struct is_not_empty
{
  __host__ __device__
    bool operator()(const uint3float1 x)
  {
    return (x.val != -1);
  }
};

__global__ void k_debug_lookup_stacks(uint3float1 * d_stacks, int total_elements){
  int a=345;
  for (int i = 0; i < 15; ++i){
    a += i;
    printf("%i: %d %d %d %f\n", i, d_stacks[i].x, d_stacks[i].y, d_stacks[i].z, d_stacks[i].val);
  }

}

void __global__ k_debug_lookup_4dgathered_stack(float* gathered_stack4d){
  for (int i = 0; i < 64*1; ++i){
    
    if (!(i % 4)) printf("\n");
    if (!(i % 16)) printf("------------\n");
    printf("%f ", gathered_stack4d[i]);
  }
}

void gather_cubes(const uchar* __restrict img,
                  const uint3 size,
                  const uint3 tsize,
                  const Parameters params,
                  uint3float1* d_stacks,
                  const uint* __restrict d_nstacks,
                  float* &d_gathered4dstack,
                  uint* d_nstacks_pow,
                  int &gather_stack_sum) 
{
  // Convert all the numbers to the lowest power of two
  uint array_size = (tsize.x*tsize.y*tsize.z);
  k_nstack_to_pow << <20, 1024 >> >(d_nstacks, d_nstacks_pow, array_size);
  checkCudaErrors(cudaGetLastError());
  thrust::device_ptr<uint> dt_nstacks_pow = thrust::device_pointer_cast(d_nstacks_pow);
  uint sum = thrust::reduce(dt_nstacks_pow, dt_nstacks_pow + array_size);
  std::cout << "Sum of pathces: "<< sum << std::endl;

  gather_stack_sum = sum; 
   
  k_debug_lookup_stacks << <1, 1 >> >(d_stacks, tsize.x*tsize.y*tsize.z);

  // Make a compaction
  uint3float1 * d_stacks_compacted;
  checkCudaErrors(cudaMalloc((void**)&d_stacks_compacted, sizeof(uint3float1)*(params.maxN *tsize.x*tsize.y*tsize.z)));
  thrust::device_ptr<uint3float1> dt_stacks = thrust::device_pointer_cast(d_stacks);
  thrust::device_ptr<uint3float1> dt_stacks_compacted = thrust::device_pointer_cast(d_stacks_compacted);

  thrust::copy_if(dt_stacks, dt_stacks + params.maxN *tsize.x*tsize.y*tsize.z, dt_stacks_compacted, is_not_empty());
  d_stacks_compacted = thrust::raw_pointer_cast(dt_stacks_compacted);
  std::cout << "+++++++++++++++++++++++" << std::endl;

  uint3float1* tmp = d_stacks;
  d_stacks = d_stacks_compacted;
  checkCudaErrors(cudaFree(tmp));
  k_debug_lookup_stacks << <1, 1 >> >(d_stacks, tsize.x*tsize.y*tsize.z);
  cudaDeviceSynchronize();

  // Allocate memory for gathered stacks uchar
  checkCudaErrors(cudaMalloc((void**)&d_gathered4dstack, sizeof(float)*(sum*params.patch_size*params.patch_size*params.patch_size)));
  std::cout << "Allocated " << sizeof(float)*(sum*params.patch_size*params.patch_size*params.patch_size) << " bytes for gathered4dstack" << std::endl;

  k_gather_cubes << < 20, 256 >> > (img, size, params, d_stacks, d_nstacks, sum, d_gathered4dstack, d_nstacks_pow);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  
}

void debug_kernel(float* tmp){
  k_debug_lookup_4dgathered_stack << <1, 1 >> >(tmp);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

#define BLOCK_SIZE2 16
#define BLOCK_SIZE 4
__device__ __host__ float alpha(int i){
  if (i == 0) return 0.5f;
  else return 0.70710678118f;
}

__device__ __host__ void mult4x4(float *in_out, int stride, const float *coeff, int coeff_stride)
{
  for (int i = 0; i < BLOCK_SIZE; i++)
    for (int j = 0; j < BLOCK_SIZE; j++)
    {
      float accumul = 0;
      for (int k = 0; k < BLOCK_SIZE; k++)
      {
        accumul += in_out[i*stride + k] * coeff[k*coeff_stride + j];
      }
      in_out[i*stride + j] = accumul;
    }

}

__global__ void dct3d(float* d_gathered4dstack, int patch_size){
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;
  int cuIdx = blockIdx.x;
  int stride = patch_size*patch_size*patch_size;
  // DCT 4x4 matrix
  const float dct_coeff[4][4] =
  {
    { 0.500000000000000f,  0.500000000000000f,  0.500000000000000f,  0.500000000000000f },
    { 0.653281482438188f,  0.270598050073099f, -0.270598050073099f, -0.653281482438188f },
    { 0.500000000000000f, -0.500000000000000f, -0.500000000000000f,  0.500000000000000f },
    { 0.270598050073099f, -0.653281482438188f,  0.653281482438188f, -0.270598050073099f }
  };
  // Load corresponding cube to the shared memory
  __shared__ float cube[4][4][4];
  int idx = (cuIdx*stride)+(x + y*patch_size + z*patch_size*patch_size);
  cube[z][y][x] = d_gathered4dstack[idx];
  __syncthreads();
  // Do 2d dct for rows (by taking slices along z direction)
  //float tmp = dct_coeff[y][0] * cube[x][0][z] + dct_coeff[y][1] * cube[x][1][z] + dct_coeff[y][2] * cube[x][2][z] + dct_coeff[y][3] * cube[x][3][z];
  float tmp = dct_coeff[x][0] * cube[0][y][z] + dct_coeff[x][1] * cube[1][y][z] + dct_coeff[x][2] * cube[2][y][z] + dct_coeff[x][3] * cube[3][y][z];
  __syncthreads();
  cube[z][y][x] = tmp;
  __syncthreads();
  //tmp = dct_coeff[0][x] * cube[0][x][z] + dct_coeff[1][x] * cube[1][x][z] + dct_coeff[2][x] * cube[2][x][z] + dct_coeff[3][x] * cube[3][x][z];
  //__syncthreads();
  //cube[x][y][z] = tmp;
 // cube[x][y][z] = dct_coeff[0][x] * cube[0][x][z] + dct_coeff[1][x] * cube[1][x][z] + dct_coeff[2][x] * cube[2][x][z] + dct_coeff[3][x] * cube[3][x][z];
  //cube[x][y][z] = dct_coeff[y][0] * cube[0][y][z] + dct_coeff[y][1] * cube[1][y][z] + dct_coeff[y][2] * cube[2][y][z] + dct_coeff[y][3] * cube[3][y][z];
  //cube[x][y][z] = dct_coeff[0][x] * cube[0][y][z] + dct_coeff[1][x] * cube[1][y][z] + dct_coeff[2][x] * cube[2][y][z] + dct_coeff[3][x] * cube[3][y][z];
  __syncthreads();

  // Part of inverse, maybe..
  //cube[x][y][z] = dct_coeff[0][y] * cube[x][0][z] + dct_coeff[1][y] * cube[x][1][z] + dct_coeff[2][y] * cube[x][2][z] + dct_coeff[3][y] * cube[x][3][z];


  d_gathered4dstack[idx] = cube[z][y][x];
}

void run_dct3d(float* d_gathered4dstack, uint gathered_size, int patch_size){
  int size = gathered_size*patch_size*patch_size*patch_size;
  float* golden_dct = new float[size];
  checkCudaErrors(cudaMemcpy(golden_dct, d_gathered4dstack, sizeof(float)*size, cudaMemcpyDeviceToHost));
  dct3d << <gathered_size, dim3(4, 4, 4) >> > (d_gathered4dstack, patch_size);
  debug_kernel(d_gathered4dstack);
  delete[] golden_dct;
}