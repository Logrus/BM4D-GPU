// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#include <bm4d-gpu/kernels.cuh>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <math.h>

__global__ void k_debug_lookup_stacks(uint3float1 *d_stacks, int total_elements)
{
  int a = 345;
  for (int i = 0; i < 150; ++i)
  {
    a += i;
    printf("%i: %d %d %d %f\n", i, d_stacks[i].x, d_stacks[i].y, d_stacks[i].z, d_stacks[i].val);
  }
}
__global__ void k_debug_lookup_4dgathered_stack(float *gathered_stack4d)
{
  for (int i = 0; i < 64 * 3; ++i)
  {
    if (!(i % 4))
      printf("\n");
    if (!(i % 16))
      printf("------------\n");
    if (!(i % 64))
      printf("------------\n");
    printf("%f ", gathered_stack4d[i]);
  }
}
__global__ void k_debug_lookup_int(int *gathered_stack4d)
{
  for (int i = 0; i < 64 * 3; ++i)
  {
    if (!(i % 4))
      printf("\n");
    if (!(i % 16))
      printf("------------\n");
    if (!(i % 64))
      printf("------------\n");
    printf("%d ", gathered_stack4d[i]);
  }
}
void debug_kernel_int(int *tmp)
{
  k_debug_lookup_int<<<1, 1>>>(tmp);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
void debug_kernel(float *tmp)
{
  k_debug_lookup_4dgathered_stack<<<1, 1>>>(tmp);
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

__device__ void add_stack(uint3float1 *d_stacks,
                          uint *d_nstacks,
                          const uint3float1 val,
                          const int maxN)
{
  int k;
  uint num = (*d_nstacks);
  if (num < maxN) // add new value
  {
    k = num++;
    while (k > 0 && val.val > d_stacks[k - 1].val)
    {
      d_stacks[k] = d_stacks[k - 1];
      --k;
    }

    d_stacks[k] = val;
    *d_nstacks = num;
  }
  else if (val.val >= d_stacks[0].val)
    return;
  else // delete highest value and add new
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

__device__ __host__ float dist(const uchar *__restrict img, const uint3 size, const int3 ref, const int3 cmp, const int k)
{
  const float normalizer = 1.0f / (k * k * k);
  float diff{0.f};
  for (int z = 0; z < k; ++z)
    for (int y = 0; y < k; ++y)
      for (int x = 0; x < k; ++x)
      {
        const int rx = max(0, min(x + ref.x, size.x - 1));
        const int ry = max(0, min(y + ref.y, size.y - 1));
        const int rz = max(0, min(z + ref.z, size.z - 1));
        const int cx = max(0, min(x + cmp.x, size.x - 1));
        const int cy = max(0, min(y + cmp.y, size.y - 1));
        const int cz = max(0, min(z + cmp.z, size.z - 1));
        // printf("rx: %d ry: %d rz: %d cx: %d cy: %d cz: %d\n", rx, ry, rz, cx, cy, cz);
        float tmp = (img[(rx) + (ry)*size.x + (rz)*size.x * size.y] - img[(cx) + (cy)*size.x + (cz)*size.x * size.y]);
        diff += tmp * tmp * normalizer;
      }
  return diff ;
}

__global__ void k_block_matching(const uchar *__restrict img,
                                 const uint3 size,
                                 const uint3 tsize,
                                 const bm4d_gpu::Parameters params,
                                 uint3float1 *d_stacks,
                                 uint *d_nstacks)
{

  for (int Idz = blockDim.z * blockIdx.z + threadIdx.z; Idz < tsize.z; Idz += blockDim.z * gridDim.z)
    for (int Idy = blockDim.y * blockIdx.y + threadIdx.y; Idy < tsize.y; Idy += blockDim.y * gridDim.y)
      for (int Idx = blockDim.x * blockIdx.x + threadIdx.x; Idx < tsize.x; Idx += blockDim.x * gridDim.x)
      {

        int x = Idx * params.step_size;
        int y = Idy * params.step_size;
        int z = Idz * params.step_size;
        if (x >= size.x || y >= size.y || z >= size.z || x < 0 || y < 0 || z < 0)
          return;

        int wxb = fmaxf(0, x - params.window_size);          // window x begin
        int wyb = fmaxf(0, y - params.window_size);          // window y begin
        int wzb = fmaxf(0, z - params.window_size);          // window z begin
        int wxe = fminf(size.x - 1, x + params.window_size); // window x end
        int wye = fminf(size.y - 1, y + params.window_size); // window y end
        int wze = fminf(size.z - 1, z + params.window_size); // window z end

        int3 ref = make_int3(x, y, z);

        for (int wz = wzb; wz <= wze; wz++)
          for (int wy = wyb; wy <= wye; wy++)
            for (int wx = wxb; wx <= wxe; wx++)
            {
              float w = dist(img, size, ref, make_int3(wx, wy, wz), params.patch_size);
              // printf("Dist %f\n", w);

              if (w < params.sim_th)
              {
                add_stack(&d_stacks[(Idx + (Idy + Idz * tsize.y) * tsize.x) * params.maxN],
                          &d_nstacks[Idx + (Idy + Idz * tsize.y) * tsize.x],
                          uint3float1(wx, wy, wz, w),
                          params.maxN);
              }
            }
      }
}

void run_block_matching(const uchar *__restrict d_noisy_volume,
                        const uint3 size,
                        const uint3 tsize,
                        const bm4d_gpu::Parameters params,
                        uint3float1 *d_stacks,
                        uint *d_nstacks,
                        const cudaDeviceProp &d_prop)
{
  int threads = std::floor(sqrt(d_prop.maxThreadsPerBlock));
  dim3 block(threads, threads, 1);
  int bs_x = d_prop.maxGridSize[1] < tsize.x ? d_prop.maxGridSize[1] : tsize.x;
  int bs_y = d_prop.maxGridSize[1] < tsize.y ? d_prop.maxGridSize[1] : tsize.y;
  dim3 grid(bs_x, bs_y, 1);

  // Debug verification
  std::cout << "Total number of reference patches " << (tsize.x * tsize.y * tsize.z) << std::endl;

  k_block_matching<<<grid, block>>>(d_noisy_volume,
                                    size,
                                    tsize,
                                    params,
                                    d_stacks,
                                    d_nstacks);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

__global__ void k_nstack_to_pow(uint3float1 *d_stacks, uint *d_nstacks, const int elements, const uint maxN)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
  {
    if (i >= elements)
      return;

    uint inGroupId = i % maxN;
    uint groupId = i / maxN;

    uint n = d_nstacks[groupId];
    uint tmp = flp2(n);
    uint diff = d_nstacks[groupId] - tmp;

    __syncthreads();
    d_nstacks[groupId] = tmp;

    if (inGroupId < diff || inGroupId >= n)
      d_stacks[i].val = -1;
  }
}

__global__ void k_gather_cubes(const uchar *__restrict img,
                               const uint3 size,
                               const bm4d_gpu::Parameters params,
                               const uint3float1 *__restrict d_stacks,
                               const uint array_size,
                               float *d_gathered4dstack)
{
  int cube_size = params.patch_size * params.patch_size * params.patch_size;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < array_size; i += blockDim.x * gridDim.x)
  {
    if (i >= array_size)
      return;
    uint3float1 ref = d_stacks[i];

    for (int z = 0; z < params.patch_size; ++z)
      for (int y = 0; y < params.patch_size; ++y)
        for (int x = 0; x < params.patch_size; ++x)
        {

          int rx = max(0, min(x + ref.x, size.x - 1));
          int ry = max(0, min(y + ref.y, size.y - 1));
          int rz = max(0, min(z + ref.z, size.z - 1));

          int img_idx = (rx) + (ry)*size.x + (rz)*size.x * size.y;
          int stack_idx = i * cube_size + (x) + (y + z * params.patch_size) * params.patch_size;

          d_gathered4dstack[stack_idx] = img[img_idx];
        }
  }
}

struct is_not_empty
{
  __host__ __device__ bool operator()(const uint3float1 x)
  {
    return (x.val != -1);
  }
};

void gather_cubes(const uchar *__restrict img,
                  const uint3 size,
                  const uint3 tsize,
                  const bm4d_gpu::Parameters params,
                  uint3float1 *&d_stacks,
                  uint *d_nstacks,
                  float *&d_gathered4dstack,
                  uint &gather_stacks_sum,
                  const cudaDeviceProp &d_prop)
{
  // Convert all the numbers in d_nstacks to the lowest power of two
  uint array_size = (tsize.x * tsize.y * tsize.z);
  int threads = d_prop.maxThreadsPerBlock;
  int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(params.maxN * array_size / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(params.maxN * array_size / threads);
  k_nstack_to_pow<<<bs_x, threads>>>(d_stacks, d_nstacks, params.maxN * array_size, params.maxN);
  checkCudaErrors(cudaGetLastError());
  thrust::device_ptr<uint> dt_nstacks = thrust::device_pointer_cast(d_nstacks);
  gather_stacks_sum = thrust::reduce(dt_nstacks, dt_nstacks + array_size);
  // std::cout << "Sum of pathces: "<< gather_stacks_sum << std::endl;
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Make a compaction
  uint3float1 *d_stacks_compacted;
  checkCudaErrors(cudaMalloc((void **)&d_stacks_compacted, sizeof(uint3float1) * gather_stacks_sum));
  thrust::device_ptr<uint3float1> dt_stacks = thrust::device_pointer_cast(d_stacks);
  thrust::device_ptr<uint3float1> dt_stacks_compacted = thrust::device_pointer_cast(d_stacks_compacted);
  thrust::copy_if(dt_stacks, dt_stacks + params.maxN * tsize.x * tsize.y * tsize.z, dt_stacks_compacted, is_not_empty());
  d_stacks_compacted = thrust::raw_pointer_cast(dt_stacks_compacted);
  uint3float1 *tmp = d_stacks;
  d_stacks = d_stacks_compacted;
  checkCudaErrors(cudaFree(tmp));
  // k_debug_lookup_stacks << <1, 1 >> >(d_stacks, tsize.x*tsize.y*tsize.z);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Allocate memory for gathered stacks uchar
  checkCudaErrors(cudaMalloc((void **)&d_gathered4dstack, sizeof(float) * (gather_stacks_sum * params.patch_size * params.patch_size * params.patch_size)));
  // std::cout << "Allocated " << sizeof(float)*(gather_stacks_sum*params.patch_size*params.patch_size*params.patch_size) << " bytes for gathered4dstack" << std::endl;

  bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(gather_stacks_sum / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(gather_stacks_sum / threads);
  k_gather_cubes<<<bs_x, threads>>>(img, size, params, d_stacks, gather_stacks_sum, d_gathered4dstack);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

__global__ void dct3d(float *d_gathered4dstack, int patch_size, uint gather_stacks_sum)
{
  for (int cuIdx = blockIdx.x; cuIdx < gather_stacks_sum; cuIdx += blockDim.x * gridDim.x)
  {
    if (cuIdx >= gather_stacks_sum)
      return;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    // int cuIdx = blockIdx.x;
    int stride = patch_size * patch_size * patch_size;
    // DCT 4x4 matrix
    const float dct_coeff[4][4] =
        {
            {0.500000000000000f, 0.500000000000000f, 0.500000000000000f, 0.500000000000000f},
            {0.653281482438188f, 0.270598050073099f, -0.270598050073099f, -0.653281482438188f},
            {0.500000000000000f, -0.500000000000000f, -0.500000000000000f, 0.500000000000000f},
            {0.270598050073099f, -0.653281482438188f, 0.653281482438188f, -0.270598050073099f}};
    const float dct_coeff_T[4][4] =
        {
            {0.500000000000000f, 0.653281482438188f, 0.500000000000000f, 0.270598050073099f},
            {0.500000000000000f, 0.270598050073099f, -0.500000000000000f, -0.653281482438188f},
            {0.500000000000000f, -0.270598050073099f, -0.500000000000000f, 0.653281482438188f},
            {0.500000000000000f, -0.653281482438188f, 0.500000000000000f, -0.270598050073099f}};
    // Load corresponding cube to the shared memory
    __shared__ float cube[4][4][4];
    int idx = (cuIdx * stride) + (x + y * patch_size + z * patch_size * patch_size);
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
    for (int i = 0; i < 4; ++i)
    {
      z_vec[i] = cube[i][y][x];
    }
    __syncthreads();
    cube[z][y][x] = z_vec[0] * dct_coeff[z][0] + z_vec[1] * dct_coeff[z][1] + z_vec[2] * dct_coeff[z][2] + z_vec[3] * dct_coeff[z][3];
    __syncthreads();
    d_gathered4dstack[idx] = cube[z][y][x];
  }
}

void run_dct3d(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size, const cudaDeviceProp &d_prop)
{
  int threads = patch_size * patch_size * patch_size;
  int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(gather_stacks_sum / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(gather_stacks_sum / threads);
  dct3d<<<bs_x, dim3(patch_size, patch_size, patch_size)>>>(d_gathered4dstack, patch_size, gather_stacks_sum);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

__global__ void idct3d(float *d_gathered4dstack, int patch_size, uint gather_stacks_sum)
{
  for (int cuIdx = blockIdx.x; cuIdx < gather_stacks_sum; cuIdx += blockDim.x * gridDim.x)
  {
    if (cuIdx >= gather_stacks_sum)
      return;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    // int cuIdx = blockIdx.x;
    int stride = patch_size * patch_size * patch_size;
    // DCT 4x4 matrix
    const float dct_coeff[4][4] =
        {
            {0.500000000000000f, 0.500000000000000f, 0.500000000000000f, 0.500000000000000f},
            {0.653281482438188f, 0.270598050073099f, -0.270598050073099f, -0.653281482438188f},
            {0.500000000000000f, -0.500000000000000f, -0.500000000000000f, 0.500000000000000f},
            {0.270598050073099f, -0.653281482438188f, 0.653281482438188f, -0.270598050073099f}};
    const float dct_coeff_T[4][4] =
        {
            {0.500000000000000f, 0.653281482438188f, 0.500000000000000f, 0.270598050073099f},
            {0.500000000000000f, 0.270598050073099f, -0.500000000000000f, -0.653281482438188f},
            {0.500000000000000f, -0.270598050073099f, -0.500000000000000f, 0.653281482438188f},
            {0.500000000000000f, -0.653281482438188f, 0.500000000000000f, -0.270598050073099f}};
    // Load corresponding cube to the shared memory
    __shared__ float cube[4][4][4];
    int idx = (cuIdx * stride) + (x + y * patch_size + z * patch_size * patch_size);
    cube[z][y][x] = d_gathered4dstack[idx];
    __syncthreads();
    float z_vec[4];
    for (int i = 0; i < 4; ++i)
    {
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
}

void run_idct3d(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size, const cudaDeviceProp &d_prop)
{
  int threads = patch_size * patch_size * patch_size;
  int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(gather_stacks_sum / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(gather_stacks_sum / threads);
  idct3d<<<bs_x, dim3(patch_size, patch_size, patch_size)>>>(d_gathered4dstack, patch_size, gather_stacks_sum);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

// (a,b) -> (a+b,a-b) without overflow
__device__ __host__ void whrotate(float &a, float &b)
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
  for (; x; x >>= 1)
    ++l2;
  return l2;
}

/**
 * Fast Walsh-Hadamard transform
 */
__device__ __host__ void fwht(float *data, int size)
{
  const long l2 = ilog2(size) - 1;
  for (long i = 0; i < l2; ++i)
  {
    for (long j = 0; j < (1 << l2); j += 1 << (i + 1))
      for (long k = 0; k < (1 << i); ++k)
        whrotate(data[j + k], data[j + k + (1 << i)]);
  }
}

__global__ void k_run_wht_ht_iwht(float *d_gathered4dstack,
                                  uint groups,
                                  int patch_size,
                                  uint *d_nstacks,
                                  uint *accumulated_nstacks,
                                  float *d_group_weights,
                                  const float hard_th)
{

  for (uint cuIdx = blockIdx.x; cuIdx < groups; cuIdx += gridDim.x)
  {
    if (cuIdx >= groups)
      return;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    // int cuIdx = blockIdx.x;
    int stride = patch_size * patch_size * patch_size;
    float group_vector[16];
    int size = d_nstacks[cuIdx];
    int group_start = accumulated_nstacks[cuIdx];
    // printf("\nSize: %d Group start: %d \n", size, group_start);

    for (int i = 0; i < size; i++)
    {
      long long int gl_idx = (group_start * stride) + (x + y * patch_size + z * patch_size * patch_size + i * stride);
      group_vector[i] = d_gathered4dstack[gl_idx];
    }

    fwht(group_vector, size);
    // Threshold
    float threshold = hard_th * sqrtf((float)size);
    d_group_weights[cuIdx * stride + x + y * patch_size + z * patch_size * patch_size] = 0;
    for (int i = 0; i < size; i++)
    {
      group_vector[i] /= size; // normalize
      if (fabs(group_vector[i]) > threshold)
      {
        d_group_weights[cuIdx * stride + x + y * patch_size + z * patch_size * patch_size] += 1;
      }
      else
      {
        group_vector[i] = 0;
      }
    }
    // Inverse fwht
    fwht(group_vector, size);
    for (int i = 0; i < size; i++)
    {
      long long int gl_idx = (group_start * stride) + (x + y * patch_size + z * patch_size * patch_size + i * stride);
      d_gathered4dstack[gl_idx] = group_vector[i];
    }
  }
}
__global__ void k_sum_group_weights(float *d_group_weights, uint *d_accumulated_nstacks, uint *d_nstacks, uint groups, int patch_size)
{
  for (int cuIdx = blockIdx.x; cuIdx < groups; cuIdx += gridDim.x)
  {
    if (cuIdx >= groups)
      return;
    int stride = patch_size * patch_size * patch_size;
    float counter = 0;
    for (int i = 0; i < stride; ++i)
    {
      int idx = cuIdx * stride + i;
      counter += d_group_weights[idx];
    }
    __syncthreads();
    d_group_weights[cuIdx * stride] = counter > 0. ? 1.0 / (float)counter : 0.;
  }
}

struct is_computed_weight
{
  __host__ __device__ bool operator()(const float x)
  {
    return (x < 1.0);
  }
};

void run_wht_ht_iwht(float *d_gathered4dstack,
                     uint gather_stacks_sum,
                     int patch_size,
                     uint *d_nstacks,
                     const uint3 tsize,
                     float *&d_group_weights,
                     const bm4d_gpu::Parameters params,
                     const cudaDeviceProp &d_prop)
{
  int groups = tsize.x * tsize.y * tsize.z;
  // Accumulate nstacks through sum
  uint *d_accumulated_nstacks;
  checkCudaErrors(cudaMalloc((void **)&d_accumulated_nstacks, sizeof(uint) * groups));
  thrust::device_ptr<uint> dt_accumulated_nstacks = thrust::device_pointer_cast(d_accumulated_nstacks);
  thrust::device_ptr<uint> dt_nstacks = thrust::device_pointer_cast(d_nstacks);
  thrust::exclusive_scan(dt_nstacks, dt_nstacks + groups, dt_accumulated_nstacks);
  d_accumulated_nstacks = thrust::raw_pointer_cast(dt_accumulated_nstacks);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMalloc((void **)&d_group_weights, sizeof(float) * groups * patch_size * patch_size * patch_size)); // Cubes with weights for each group
  checkCudaErrors(cudaMemset(d_group_weights, 0.0, sizeof(float) * groups * patch_size * patch_size * patch_size));
  int threads = params.patch_size * params.patch_size * params.patch_size;
  int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(groups / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(groups / threads);
  k_run_wht_ht_iwht<<<bs_x, dim3(params.patch_size, params.patch_size, params.patch_size)>>>(d_gathered4dstack, groups, patch_size, d_nstacks, d_accumulated_nstacks, d_group_weights, params.hard_th);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  threads = 1;
  bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(groups / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(groups / threads);
  k_sum_group_weights<<<bs_x, threads>>>(d_group_weights, d_accumulated_nstacks, d_nstacks, groups, patch_size);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_accumulated_nstacks));
}

void aggregation_cpu(float *image_vol,
                     float *weights_vol,
                     float *group_weights,
                     uint3 size,
                     uint3 tsize,
                     int gather_stacks_sum,
                     uint3float1 *stacks,
                     uint *nstacks,
                     float *gathered_stacks,
                     int patch_size)
{

  int all_cubes = gather_stacks_sum;
  int stride = patch_size * patch_size * patch_size;

  int cubes_so_far = 0;
  int groupId = 0; // Iterator over group numbers

  for (int i = 0; i < all_cubes; ++i)
  {
    uint3float1 ref = stacks[i];

    uint cubes_in_group = nstacks[groupId];
    if ((i - cubes_so_far) == cubes_in_group)
    {
      cubes_so_far += cubes_in_group;
      // std::cout << "cubes in a grouo " << cubes_in_group << std::endl;
      groupId++;
    }

    float weight = group_weights[groupId * stride];
    // std::cout << "Weight: " << weight << std::endl;

    for (int z = 0; z < patch_size; ++z)
      for (int y = 0; y < patch_size; ++y)
        for (int x = 0; x < patch_size; ++x)
        {
          int rx = x + ref.x;
          int ry = y + ref.y;
          int rz = z + ref.z;
          if (rx < 0 || rx >= size.x)
            continue;
          if (ry < 0 || ry >= size.y)
            continue;
          if (rz < 0 || rz >= size.z)
            continue;
          // std::cout << image_vol[rx + ry*size.x + rz*size.x*size.y] << std::endl;
          float tmp = gathered_stacks[i * stride + x + y * patch_size + z * patch_size * patch_size];
          image_vol[rx + ry * size.x + rz * size.x * size.y] += tmp * weight;
          weights_vol[rx + ry * size.x + rz * size.x * size.y] += weight;
        }
  }

  for (int i = 0; i < size.x * size.y * size.z; ++i)
  {
    image_vol[i] = weights_vol[i] > 0. ? image_vol[i] / weights_vol[i] : 0.;
  }
}

__global__ void k_aggregation(float *d_denoised_volume,
                              float *d_weights_volume,
                              const uint3 size,
                              const uint3 tsize,
                              const float *d_gathered4dstack,
                              uint3float1 *d_stacks,
                              uint *d_nstacks,
                              float *group_weights,
                              const bm4d_gpu::Parameters params,
                              const uint *d_accumulated_nstacks)
{

  uint groups = (tsize.x * tsize.y * tsize.z);
  int stride = params.patch_size * params.patch_size * params.patch_size;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < groups; i += blockDim.x * gridDim.x)
  {

    if (i >= groups)
      return;

    float weight = group_weights[i * stride];
    int patches = d_nstacks[i];
    int group_beginning = d_accumulated_nstacks[i];
    // printf("Weight for the group %d is %f\n", i, weight);
    // printf("Num of patches %d\n", patches);
    // printf("Group beginning %d\n", group_beginning);
    // if (i > 15) return;
    for (int p = 0; p < patches; ++p)
    {
      uint3float1 ref = d_stacks[group_beginning + p];

      for (int z = 0; z < params.patch_size; ++z)
        for (int y = 0; y < params.patch_size; ++y)
          for (int x = 0; x < params.patch_size; ++x)
          {
            int rx = x + ref.x;
            int ry = y + ref.y;
            int rz = z + ref.z;

            if (rx < 0 || rx >= size.x)
              continue;
            if (ry < 0 || ry >= size.y)
              continue;
            if (rz < 0 || rz >= size.z)
              continue;

            int img_idx = (rx) + (ry)*size.x + (rz)*size.x * size.y;
            long long int stack_idx = group_beginning * stride + (x) + (y + z * params.patch_size) * params.patch_size + p * stride;
            float tmp = d_gathered4dstack[stack_idx];
            __syncthreads();
            atomicAdd(&d_denoised_volume[img_idx], tmp * weight);
            atomicAdd(&d_weights_volume[img_idx], weight);
          }
    }
  }
}

__global__ void k_normalizer(float *d_denoised_volume,
                             const float *__restrict d_weights_volume,
                             const uint3 size)
{
  int im_size = size.x * size.y * size.z;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < im_size; i += blockDim.x * gridDim.x)
  {
    if (i >= im_size)
      return;
    float tmp = d_denoised_volume[i];
    __syncthreads();
    d_denoised_volume[i] = tmp / d_weights_volume[i];
  }
}

void run_aggregation(float *final_image,
                     const uint3 size,
                     const uint3 tsize,
                     const float *d_gathered4dstack,
                     uint3float1 *d_stacks,
                     uint *d_nstacks,
                     float *d_group_weights,
                     const bm4d_gpu::Parameters params,
                     int gather_stacks_sum,
                     const cudaDeviceProp &d_prop)
{
  int im_size = size.x * size.y * size.z;
  int groups = tsize.x * tsize.y * tsize.z;

  // Accumulate nstacks through sum
  uint *d_accumulated_nstacks;
  checkCudaErrors(cudaMalloc((void **)&d_accumulated_nstacks, sizeof(uint) * groups));
  thrust::device_ptr<uint> dt_accumulated_nstacks = thrust::device_pointer_cast(d_accumulated_nstacks);
  thrust::device_ptr<uint> dt_nstacks = thrust::device_pointer_cast(d_nstacks);
  thrust::exclusive_scan(dt_nstacks, dt_nstacks + groups, dt_accumulated_nstacks);
  d_accumulated_nstacks = thrust::raw_pointer_cast(dt_accumulated_nstacks);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  float *d_denoised_volume, *d_weights_volume;
  checkCudaErrors(cudaMalloc((void **)&d_denoised_volume, sizeof(float) * size.x * size.y * size.z));
  checkCudaErrors(cudaMalloc((void **)&d_weights_volume, sizeof(float) * size.x * size.y * size.z));
  checkCudaErrors(cudaMemset(d_denoised_volume, 0.0, sizeof(float) * size.x * size.y * size.z));
  checkCudaErrors(cudaMemset(d_weights_volume, 0.0, sizeof(float) * size.x * size.y * size.z));
  int threads = d_prop.maxThreadsPerBlock;
  int bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(groups / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(groups / threads);
  k_aggregation<<<bs_x, threads>>>(d_denoised_volume, d_weights_volume, size, tsize, d_gathered4dstack, d_stacks, d_nstacks, d_group_weights, params, d_accumulated_nstacks);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  threads = d_prop.maxThreadsPerBlock;
  bs_x = std::ceil(d_prop.maxGridSize[1] / threads) < std::ceil(im_size / threads) ? std::ceil(d_prop.maxGridSize[1] / threads) : std::ceil(im_size / threads);
  k_normalizer<<<bs_x, threads>>>(d_denoised_volume, d_weights_volume, size);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(final_image, d_denoised_volume, sizeof(float) * im_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_denoised_volume));
  checkCudaErrors(cudaFree(d_weights_volume));
  checkCudaErrors(cudaFree(d_accumulated_nstacks));
}
