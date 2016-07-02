#include "kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <math.h>

__global__ void k_debug_lookup_stacks(uint3float1 * d_stacks, int total_elements){
  int a = 345;
  for (int i = 0; i < 15; ++i){
    a += i;
    printf("%i: %d %d %d %f\n", i, d_stacks[i].x, d_stacks[i].y, d_stacks[i].z, d_stacks[i].val);
  }

}

void __global__ k_debug_lookup_4dgathered_stack(float* gathered_stack4d){
  for (int i = 0; i < 64 * 3; ++i){

    if (!(i % 4)) printf("\n");
    if (!(i % 16)) printf("------------\n");
    if (!(i % 64)) printf("------------\n");
    printf("%f ", gathered_stack4d[i]);
  }
}
__global__ void k_debug_lookup_int(int* gathered_stack4d){
  for (int i = 0; i < 64 * 3; ++i){
    if (!(i % 4)) printf("\n");
    if (!(i % 16)) printf("------------\n");
    if (!(i % 64)) printf("------------\n");
    printf("%d ", gathered_stack4d[i]);
  }
}
void debug_kernel_int(int* tmp){
  k_debug_lookup_int << <1, 1 >> >(tmp);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
void debug_kernel(float* tmp){
  k_debug_lookup_4dgathered_stack << <1, 1 >> >(tmp);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

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
								                         uint* d_nstacks)
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
					                   uint *d_nstacks)
{
	dim3 block(32, 32, 1);
 //dim3 grid(size.x / block.x / params.step_size, size.y / block.y / params.step_size, 1);
 dim3 grid(20, 20, 1);

 // Debug verification
 std::cout << "Total number of reference patches " << (tsize.x*tsize.y*tsize.z) << std::endl;

 k_block_matching << <grid, block >> >(d_noisy_volume,
                                       size,
                                       tsize,
                                       params,
                                       d_stacks,
                                       d_nstacks);

 cudaDeviceSynchronize();
 checkCudaErrors(cudaGetLastError());
}

__global__ void k_nstack_to_pow(const uint* __restrict d_nstacks, uint* d_nstacks_pow, const int size){
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < size; i += blockDim.x*gridDim.x){
    if (i<size) 
      d_nstacks_pow[i] = flp2(d_nstacks[i]);
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
  const float dct_coeff_T[4][4] =
  {
    { 0.500000000000000f, 0.653281482438188f, 0.500000000000000f, 0.270598050073099f },
    { 0.500000000000000f, 0.270598050073099f, -0.500000000000000f, -0.653281482438188f },
    { 0.500000000000000f, -0.270598050073099f, -0.500000000000000f, 0.653281482438188f },
    { 0.500000000000000f, -0.653281482438188f, 0.500000000000000f, -0.270598050073099f }
  };
  // Load corresponding cube to the shared memory
  __shared__ float cube[4][4][4];
  int idx = (cuIdx*stride)+(x + y*patch_size + z*patch_size*patch_size);
  cube[z][y][x] = d_gathered4dstack[idx];
  __syncthreads();
  // Do 2d dct for rows (by taking slices along z direction)
  float tmp = dct_coeff[y][0] * cube[z][0][x] + dct_coeff[y][1] * cube[z][1][x] + dct_coeff[y][2] * cube[z][2][x] + dct_coeff[y][3] * cube[z][3][x];
  __syncthreads();
  cube[z][y][x] = tmp;
  __syncthreads();
  tmp = dct_coeff_T[0][x] * cube[z][y][0] + dct_coeff_T[1][x] * cube[z][y][1] + dct_coeff_T[2][x] * cube[z][y][2] + dct_coeff_T[3][x] * cube[z][y][3];
  __syncthreads();
  cube[z][y][x] = tmp;
  __syncthreads();
  // Grab Z vector
  float z_vec[4];
  for (int i = 0; i < 4; ++i){
    z_vec[i] = cube[i][y][x];
  }
  __syncthreads();
  cube[z][y][x] = z_vec[0] * dct_coeff[z][0] + z_vec[1] * dct_coeff[z][1] + z_vec[2] * dct_coeff[z][2] + z_vec[3] * dct_coeff[z][3];
  __syncthreads();
  d_gathered4dstack[idx] = cube[z][y][x];
}

void run_dct3d(float* d_gathered4dstack, uint gathered_size, int patch_size){
  dct3d << <gathered_size, dim3(4, 4, 4) >> > (d_gathered4dstack, patch_size);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

__global__ void idct3d(float* d_gathered4dstack, int patch_size){
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;
  int cuIdx = blockIdx.x;
  int stride = patch_size*patch_size*patch_size;
  // DCT 4x4 matrix
  const float dct_coeff[4][4] =
  {
    { 0.500000000000000f, 0.500000000000000f, 0.500000000000000f, 0.500000000000000f },
    { 0.653281482438188f, 0.270598050073099f, -0.270598050073099f, -0.653281482438188f },
    { 0.500000000000000f, -0.500000000000000f, -0.500000000000000f, 0.500000000000000f },
    { 0.270598050073099f, -0.653281482438188f, 0.653281482438188f, -0.270598050073099f }
  };
  const float dct_coeff_T[4][4] =
  {
    { 0.500000000000000f, 0.653281482438188f, 0.500000000000000f, 0.270598050073099f },
    { 0.500000000000000f, 0.270598050073099f, -0.500000000000000f, -0.653281482438188f },
    { 0.500000000000000f, -0.270598050073099f, -0.500000000000000f, 0.653281482438188f },
    { 0.500000000000000f, -0.653281482438188f, 0.500000000000000f, -0.270598050073099f }
  };
  // Load corresponding cube to the shared memory
  __shared__ float cube[4][4][4];
  int idx = (cuIdx*stride) + (x + y*patch_size + z*patch_size*patch_size);
  cube[z][y][x] = d_gathered4dstack[idx];
  __syncthreads();
  float z_vec[4];
  for (int i = 0; i < 4; ++i){
    z_vec[i] = cube[i][y][x];
  }
  __syncthreads();
  cube[z][y][x] = z_vec[0] * dct_coeff_T[z][0] + z_vec[1] * dct_coeff_T[z][1] + z_vec[2] * dct_coeff_T[z][2] + z_vec[3] * dct_coeff_T[z][3];
  __syncthreads();
  float tmp = dct_coeff_T[y][0] * cube[z][0][x] + dct_coeff_T[y][1] * cube[z][1][x] + dct_coeff_T[y][2] * cube[z][2][x] + dct_coeff_T[y][3] * cube[z][3][x];
  __syncthreads();
  cube[z][y][x] = tmp;
  tmp = dct_coeff[0][x] * cube[z][y][0] + dct_coeff[1][x] * cube[z][y][1] + dct_coeff[2][x] * cube[z][y][2] + dct_coeff[3][x] * cube[z][y][3];
  __syncthreads();
  cube[z][y][x] = tmp;
  __syncthreads();
  d_gathered4dstack[idx] = cube[z][y][x];
}

void run_idct3d(float* d_gathered4dstack, uint gathered_size, int patch_size){
  idct3d << <gathered_size, dim3(4, 4, 4) >> > (d_gathered4dstack, patch_size);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

// (a,b) -> (a+b,a-b) without overflow
__device__ __host__ void whrotate(float& a, float& b)
{
  float t;
  t = a;
  a = a + b;
  b = t - b;
}

// Integer log2
__device__ __host__ long ilog2(long x)
{
  long l2 = 0;
  for (; x; x >>= 1) ++l2;
  return l2;
}

/**
* Fast Walsh-Hadamard transform
*/
__device__ __host__ void fwht(float* data, int size)
{
  const long l2 = ilog2(size) - 1;
  for (long i = 0; i < l2; ++i)
  {
    for (long j = 0; j < (1 << l2); j += 1 << (i + 1))
      for (long k = 0; k < (1 << i); ++k)
        whrotate(data[j + k], data[j + k + (1 << i)]);
  }
}

__global__ void k_run_wht_ht_iwht(float* d_gathered4dstack, 
                                  uint gathered_size, 
                                  int patch_size, 
                                  uint* d_nstacks_pow, 
                                  uint* accumulated_nstacks, 
                                  float* group_weights,
                                  int* group_keys){
  int x = threadIdx.x;
  int y = threadIdx.y;
  int z = threadIdx.z;
  int cuIdx = blockIdx.x;
  int stride = patch_size*patch_size*patch_size;
  float group_vector[16];
  int size = d_nstacks_pow[cuIdx];
  int group_start = accumulated_nstacks[cuIdx];
  //printf("\nSize: %d Group start: %d \n", size, group_start);

  for (int i = 0; i < size; i++){
    int gl_idx = (group_start*stride) + (x + y*patch_size + z*patch_size*patch_size + i*stride);
    group_vector[i] = d_gathered4dstack[gl_idx]; 
  }
  fwht(group_vector, size);
  //// Threshold
  float threshold = 2.7 * sqrtf((float)size);
  group_weights[cuIdx*stride + x + y*patch_size + z*patch_size*patch_size] = 0;
  group_keys[cuIdx*stride + x + y*patch_size + z*patch_size*patch_size] = cuIdx+1;
  for (int i = 0; i < size; i++){
    group_vector[i] /= size; // normalize
    if (fabs(group_vector[i]) > threshold)
    {
      group_weights[cuIdx*stride + x + y*patch_size + z*patch_size*patch_size] += 1;
    }
    else {
      group_vector[i] = 0;
    }
  }
  //// Inverse fwht
  fwht(group_vector, size);
  for (int i = 0; i < size; i++){
    int gl_idx = (group_start*stride) + (x + y*patch_size + z*patch_size*patch_size + i*stride);
    d_gathered4dstack[gl_idx] = group_vector[i];
  }
}

void run_wht_ht_iwht(float* d_gathered4dstack, uint gathered_size, int patch_size, uint* d_nstacks_pow, const uint3 tsize){
  uint* accumulated_nstacks;
  checkCudaErrors(cudaMalloc((void **)&accumulated_nstacks, sizeof(uint)*gathered_size));
  thrust::device_ptr<uint> dt_accumulated_nstacks = thrust::device_pointer_cast(accumulated_nstacks);
  thrust::device_ptr<uint> dt_nstacks = thrust::device_pointer_cast(d_nstacks_pow);
  thrust::exclusive_scan(dt_nstacks, dt_nstacks + gathered_size, dt_accumulated_nstacks);
  accumulated_nstacks = thrust::raw_pointer_cast(dt_accumulated_nstacks);
  int groups = tsize.x*tsize.y*tsize.z;
  

  float* group_weights;
  int *group_keys, *dummy;
  checkCudaErrors(cudaMalloc((void **)&group_weights, sizeof(float)*groups*patch_size*patch_size*patch_size)); // Cubes with weights for each group
  checkCudaErrors(cudaMalloc((void **)&group_keys, sizeof(int)*groups*patch_size*patch_size*patch_size));
  checkCudaErrors(cudaMalloc((void **)&dummy, sizeof(int)*groups*patch_size*patch_size*patch_size));

  k_run_wht_ht_iwht << <groups, dim3(4, 4, 4) >> > (d_gathered4dstack, gathered_size, patch_size, d_nstacks_pow, accumulated_nstacks, group_weights, group_keys);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());


  debug_kernel(group_weights);
  debug_kernel_int(group_keys);
  float* out_weights;
  checkCudaErrors(cudaMalloc((void **)&out_weights, sizeof(float)*groups*patch_size*patch_size*patch_size));

  // Keys
  thrust::device_ptr<int> dt_dummy = thrust::device_pointer_cast(dummy);
  thrust::device_ptr<int> dt_group_keys = thrust::device_pointer_cast(group_keys);
  // Data
  thrust::device_ptr<float> dt_out_weights = thrust::device_pointer_cast(out_weights);
  thrust::device_ptr<float> dt_group_weights = thrust::device_pointer_cast(group_weights);

  //thrust::reduce_by_key(dt_group_keys, dt_group_keys + 64, dt_group_weights, dt_dummy, dt_out_weights);
  out_weights = thrust::raw_pointer_cast(dt_out_weights);
  debug_kernel(out_weights);
  checkCudaErrors(cudaFree(accumulated_nstacks));
  checkCudaErrors(cudaFree(group_weights));
  checkCudaErrors(cudaFree(dummy));
  checkCudaErrors(cudaFree(group_keys));

}
__global__ void k_aggregation(float* d_denoised_volume, 
                              float* d_weights_volume,
                            const uint3 size,
                            const uint3 tsize,
                            float* d_gathered4dstack, 
                            uint3float1* d_stacks, 
                            uint* d_nstacks, 
                            float* group_weights, 
                            const Parameters params){

  uint array_size = (tsize.x*tsize.y*tsize.z);
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < array_size; i += blockDim.x*gridDim.x){

    uint3float1 ref = d_stacks[i];
    float weight = group_weights[i];
    int cube_size = params.patch_size*params.patch_size*params.patch_size;

    for (int z = 0; z < params.patch_size; ++z)
      for (int y = 0; y < params.patch_size; ++y)
        for (int x = 0; x < params.patch_size; ++x){
          int rx = x + ref.x;
          int ry = y + ref.y;
          int rz = z + ref.z;

          if (rx < 0 || rx >= size.x) continue;
          if (ry < 0 || ry >= size.y) continue;
          if (rz < 0 || rz >= size.z) continue;

          int img_idx = (rx)+(ry)*size.x + (rz)*size.x*size.y;
          int stack_idx = i*cube_size + (x)+(y + z*params.patch_size)*params.patch_size;
          //d_denoised_volume[img_idx] = d_gathered4dstack[stack_idx];
          atomicAdd(d_denoised_volume + img_idx, d_gathered4dstack[stack_idx]);
          atomicAdd(d_weights_volume + img_idx, weight);
        }
  }
}

__global__ void k_normalizer(float* d_denoised_volume,
                             const float* __restrict d_weights_volume,
                             const uint3 size)
{
  for (int Idz = blockDim.z * blockIdx.z + threadIdx.z; Idz < size.z; Idz += blockDim.z*gridDim.z)
    for (int Idy = blockDim.y * blockIdx.y + threadIdx.y; Idy < size.y; Idy += blockDim.y*gridDim.y)
      for (int Idx = blockDim.x * blockIdx.x + threadIdx.x; Idx < size.x; Idx += blockDim.x*gridDim.x)
      {
        int idx = Idx + Idy*size.x + Idx*size.x*size.y;
        float tmp = d_denoised_volume[idx];
        __syncthreads();
        d_denoised_volume[idx] = d_denoised_volume[idx] / d_weights_volume[idx];
      }
}

void run_aggregation(float* final_image,
                     const uint3 size, 
                     const uint3 tsize, 
                     float* d_gathered4dstack, 
                     uint3float1* d_stacks, 
                     uint* d_nstacks, 
                     float* group_weights,
                     const Parameters params)
{
  int im_size = size.x*size.y*size.z;
  int groups = tsize.x*tsize.y*tsize.z;
  float* d_junk_weights;
  float* junk_weights = new float[groups];
  for (int i = 0; i < groups; ++i) { junk_weights[i] = 1.0; }
  checkCudaErrors(cudaMalloc((void **)&d_junk_weights, sizeof(float)*groups));
  checkCudaErrors(cudaMemcpy(d_junk_weights, junk_weights, sizeof(float)*groups, cudaMemcpyHostToDevice));

  float* d_denoised_volume, *d_weights_volume;
  checkCudaErrors(cudaMalloc((void **)&d_denoised_volume, sizeof(float)*size.x*size.y*size.z));
  checkCudaErrors(cudaMalloc((void **)&d_weights_volume, sizeof(float)*size.x*size.y*size.z));

  k_aggregation << <20, 1024 >> >(d_denoised_volume, d_weights_volume, size, tsize, d_gathered4dstack, d_stacks, d_nstacks, d_junk_weights, params);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  k_normalizer << <20, dim3(32, 32, 1) >> >(d_denoised_volume, d_weights_volume, size);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(final_image, d_denoised_volume, sizeof(float)*im_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_junk_weights));
  delete[] junk_weights;
}