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
  std::vector<uchar> noisy_volume;
  std::vector<uchar> base_volume;

  // Device variables
  uchar *d_noisy_volume;
  uchar *d_denoised_volume;
  uint3float1 *d_stacks;
  uint *d_nstacks;
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
    const std::vector<uchar> &in_noisy_volume,
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
    checkCudaErrors(cudaMalloc((void**) &d_noisy_volume, sizeof(uchar)*size));
    std::cout<<"Allocated "<<sizeof(uchar)*size<<" bytes for d_noisy_volume"<<std::endl;
    //checkCudaErrors(cudaMalloc((void**) &d_denoised_volume, sizeof(uchar)*size));
    //std::cout<<"Allocated "<<sizeof(uchar)*size<<" bytes for d_denoised_volume"<<std::endl;
	checkCudaErrors(cudaMalloc((void**)&d_stacks, sizeof(uint3float1)*(params.maxN * 1024)));
	std::cout << "Allocated " << sizeof(uint3float1)*(params.maxN*1024) << " bytes for d_stacks" << std::endl;
	checkCudaErrors(cudaMalloc((void**)&d_nstacks, sizeof(uint3float1)*(1024)));
	std::cout << "Allocated " << sizeof(uint3float1)*(1024) << " bytes for d_nstacks" << std::endl;
  };
  inline ~BM4D(){
    // Cleanup
    checkCudaErrors(cudaFree(d_noisy_volume));
    std::cout<<"Cleaned up "<<sizeof(uchar)*size<<" bytes of d_noisy_volume"<<std::endl;
    //checkCudaErrors(cudaFree(d_denoised_volume));
    //std::cout<<"Cleaned up "<<sizeof(uchar)*size<<" bytes of d_denoised_volume"<<std::endl;
	checkCudaErrors(cudaFree(d_stacks));
	//std::cout << "Cleaned up " << sizeof(uint3float1)*(params.maxN*size) << " bytes of d_stacks" << std::endl;
	checkCudaErrors(cudaFree(d_nstacks));
	//std::cout << "Cleaned up " << sizeof(uint3float1)*(size) << " bytes of d_nstacks" << std::endl;
  };

  std::vector<unsigned char> run_first_step();

};
