#pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "helper_cuda.h"
#include "stdio.h"
#include "parameters.h"
#ifndef idx3
#define idx3(x,y,z,x_size,y_size) ((x) + ((y)+(y_size)*(z))*(x_size))
#endif

typedef unsigned char uchar;
typedef unsigned int uint;

struct uint3float1
{
	uint x;
	uint y;
	uint z;
	float val;

	__host__ __device__ uint3float1(uint x, uint y, uint z, float val) : x(x), y(y), z(z), val(val) { }
};

inline uint3float1 make_uint3float1(uint x, uint y, uint z, float val) { return uint3float1(x, y, z, val); }
inline uint3float1 make_uint3float1(uint3 c, float val) { return uint3float1(c.x, c.y, c.z, val); }

void run_block_matching(const uchar* __restrict d_noisy_volume,
                        const uint3 size,
                        const uint3 tsize,
                        const Parameters params,
                        uint3float1 *d_stacks,
                        uint *d_nstacks,
                        uchar* out);


// Gather cubes together
void gather_cubes(const uchar* __restrict img,
                  const uint3 size,
                  const Parameters params,
                  const uint3float1* __restrict d_stacks,
                  const uint* __restrict d_nstacks,
                  uchar* d_gathered4dstack,
                  uint* d_nstacks_pow,
                  int &gather_stack_sum); // TODO: remove

// Perform 3D DCT
void run_dct3d();
// Do WHT in 4th dim + Hard Thresholding + IWHT
void run_wht_ht_iwht();
// Perform inverse 3D DCT
void run_idct3d();
// Aggregate
void run_aggregation();