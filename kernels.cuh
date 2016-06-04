#pragma once
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "stdio.h"

void wrapper_simple_kernel(const std::vector<float> &h_result, std::vector<float> &out, int width, int height, int depth);