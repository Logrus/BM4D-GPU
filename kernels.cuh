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


void run_block_matching(float *d_noisy_volume,
                      const uint3 im_size,
                      const Parameters params);


// Gather cubes together
// Perform 3D DCT
void run_dct3d();
// Do WHT in 4th dim + Hard Thresholding + IWHT
void run_wht_ht_iwht();
// Perform inverse 3D DCT
void run_idct3d();
// Aggregate
void run_aggregation();