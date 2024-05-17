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

/// @brief Compute the distance between two patches. include
/// The distance is computed as the sum of the squared differences between the
/// intensities of each pixel in patch cubes. The result is normalized by the
/// number of pixels in the patch.
/// $d(C^z_{x_i}, C^z_{x_j}) = \frac{1}{L^3} (C^z_{x_i} - C^z_{x_j})^2$
__device__ __host__ float dist(const uchar *__restrict img, const uint3 size, const int3 ref,
                               const int3 cmp, const int k);

/// @brief Round up to the lower power of 2.
/// This function only works for 32-bit unsigned integers.
/// Example lower_power_2(5) = 4, lower_power_2(8) = 8, lower_power_2(9) = 8.
///
/// @note It returns 0 for x = 0, which is not power of 2.
///
/// @param x input number
/// @return nearest lower power of 2 of x
///
/// @note Similar function that returns upper power:
///       https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
__device__ __host__ __inline__ uint lower_power_2(uint x) {
  x = x | (x >> 1);
  x = x | (x >> 2);
  x = x | (x >> 4);
  x = x | (x >> 8);
  x = x | (x >> 16);
  return x - (x >> 1);
}

void debug_kernel(float *tmp);
