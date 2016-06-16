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

void run_block_matching(uchar *d_noisy_volume,
                      const uint3 im_size,
                      const Parameters params,
					  uint3float1 *d_stacks,
					  uint *d_nstacks
					  );


// Gather cubes together
// Perform 3D DCT
void run_dct3d();
// Do WHT in 4th dim + Hard Thresholding + IWHT
void run_wht_ht_iwht();
// Perform inverse 3D DCT
void run_idct3d();
// Aggregate
void run_aggregation();