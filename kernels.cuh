#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void simple_kernel(float *img, const int width, const int height, const int depth);

void inline run_simple_kernel(){

}