// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "helper_cuda.h"
#include "parameters.h"
#include "stdio.h"

#undef NDEBUG
#ifndef idx3
#define idx3(x, y, z, x_size, y_size) ((x) + ((y) + (y_size) * (z)) * (x_size))
#endif

typedef unsigned char uchar;
typedef unsigned int uint;

struct uint3float1 {
  uint x;
  uint y;
  uint z;
  float val;
  __host__ __device__ uint3float1() : x(0), y(0), z(0), val(-1) {};
  __host__ __device__ uint3float1(uint x, uint y, uint z, float val) : x(x), y(y), z(z), val(val) {}
};
inline uint3float1 make_uint3float1(uint x, uint y, uint z, float val) {
  return uint3float1(x, y, z, val);
}
inline uint3float1 make_uint3float1(uint3 c, float val) { return uint3float1(c.x, c.y, c.z, val); }

void run_block_matching(const uchar *__restrict d_noisy_volume, const uint3 size, const uint3 tsize,
                        const bm4d_gpu::Parameters params, uint3float1 *d_stacks, uint *d_nstacks,
                        const cudaDeviceProp &d_prop);
// Gather cubes together
void gather_cubes(const uchar *__restrict img, const uint3 size, const uint3 tsize,
                  const bm4d_gpu::Parameters params, uint3float1 *&d_stacks, uint *d_nstacks,
                  float *&d_gathered4dstack, uint &gather_stacks_sum, const cudaDeviceProp &d_prop);
// Perform 3D DCT
void run_dct3d(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size,
               const cudaDeviceProp &d_prop);
// Do WHT in 4th dim + Hard Thresholding + IWHT
void run_wht_ht_iwht(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size,
                     uint *d_nstacks, const uint3 tsize, float *&d_group_weights,
                     const bm4d_gpu::Parameters params, const cudaDeviceProp &d_prop);
// Perform inverse 3D DCT
void run_idct3d(float *d_gathered4dstack, uint gather_stacks_sum, int patch_size,
                const cudaDeviceProp &d_prop);
// Aggregate
void run_aggregation(float *final_image, const uint3 size, const uint3 tsize,
                     const float *d_gathered4dstack, uint3float1 *d_stacks, uint *d_nstacks,
                     float *group_weights, const bm4d_gpu::Parameters params, int gather_stacks_sum,
                     const cudaDeviceProp &d_prop);

// Helper functions
__device__ __host__ float dist(const uchar *__restrict img, const uint3 size, const int3 ref,
                               const int3 cmp, const int k);

void debug_kernel(float *tmp);
