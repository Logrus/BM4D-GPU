#pragma once

#include "CImg.h"
#include "parameters.h"
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include "kernels.cuh"
#include "stopwatch.hpp"
using namespace cimg_library;

class BM4D {
private:
  // Main variables
  std::vector<unsigned char> noisy_volume;
  std::vector<unsigned char> base_volume;

  std::vector<float> working_image;

  // Device variables
  float *d_noisy_volume;
  float *d_denoised_volume;
  int width, height, depth, size;

  // Parameters for launching kernels
  dim3 block;
  dim3 grid;

  cudaDeviceProp d_prop;
  Parameters params;

  void copy_image_to_device();
  void copy_image_to_host();


public:
  inline BM4D(Parameters p,
    const std::vector<unsigned char> &in_noisy_volume,
    const int &width,
    const int &height,
    const int &depth
    ):
      params(p),
      width(width),
      height(height),
      depth(depth)
  {
    noisy_volume = in_noisy_volume;
    size = width*height*depth;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&d_prop, device));

    // Memory allocation
    checkCudaErrors(cudaMalloc((void**) &d_noisy_volume, sizeof(float)*size));
    checkCudaErrors(cudaMalloc((void**) &d_denoised_volume, sizeof(float)*size));
  };
  inline ~BM4D(){
    // Cleanup
    checkCudaErrors(cudaFree(d_noisy_volume));
    checkCudaErrors(cudaFree(d_denoised_volume));
  };

  std::vector<unsigned char> run_first_step();

};
