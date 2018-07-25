/*
 * 2016, Vladislav Tananaev
 * v.d.tananaev [at] gmail [dot] com
 */
#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <bm4d-gpu/parameters.h>
#include <bm4d-gpu/kernels.cuh>
#include <bm4d-gpu/stopwatch.hpp>

class BM4D {
 public:
  BM4D(bm4d_gpu::Parameters p, const std::vector<uchar>& in_noisy_volume, const int& width,
       const int& height, const int& depth)
      : params(p),
        width(width),
        height(height),
        depth(depth),
        d_gathered4dstack(NULL),
        d_stacks(NULL),
        d_nstacks(NULL) {
    noisy_volume = in_noisy_volume;
    size = width * height * depth;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&d_prop, device));

    twidth = std::floor((width - 1) / params.step_size + 1);
    theight = std::floor((height - 1) / params.step_size + 1);
    tdepth = std::floor((depth - 1) / params.step_size + 1);
    tsize = twidth * theight * tdepth;

    // uint3float1* tmp_arr = new uint3float1[params.maxN*tsize];
    checkCudaErrors(cudaMalloc((void**)&d_stacks, sizeof(uint3float1) * (params.maxN * tsize)));
    // std::cout << "Allocated " << sizeof(uint3float1)*(params.maxN*tsize) << "
    // bytes for d_stacks" << std::endl; checkCudaErrors(cudaMemcpy(d_stacks,
    // tmp_arr, sizeof(uint3float1)*params.maxN*tsize, cudaMemcpyHostToDevice));
    // delete[] tmp_arr;

    checkCudaErrors(cudaMalloc((void**)&d_nstacks, sizeof(uint) * (tsize)));
    checkCudaErrors(cudaMemset(d_nstacks, 0, sizeof(uint) * tsize));
    // std::cout << "Allocated " << (tsize) << " elements for d_nstacks" <<
    // std::endl;
  }

  inline ~BM4D() {
    // Cleanup
    if (d_stacks) {
      checkCudaErrors(cudaFree(d_stacks));
      // std::cout << "Cleaned up d_stacks" << std::endl;
    }
    if (d_nstacks) {
      checkCudaErrors(cudaFree(d_nstacks));
      // std::cout << "Cleaned up d_nstacks" << std::endl;
    }
    if (d_gathered4dstack) {
      checkCudaErrors(cudaFree(d_gathered4dstack));
      // std::cout << "Cleaned up bytes of d_gathered4dstack" << std::endl;
    }
    cudaDeviceReset();
  };

  std::vector<unsigned char> run_first_step();

 private:
  // Main variables
  std::vector<uchar> noisy_volume;

  // Device variables
  float* d_gathered4dstack;
  uint3float1* d_stacks;
  uint* d_nstacks;
  float* d_group_weights;
  int width, height, depth, size;
  int twidth, theight, tdepth, tsize;

  // Parameters for launching kernels
  dim3 block;
  dim3 grid;

  cudaDeviceProp d_prop;
  bm4d_gpu::Parameters params;
};
