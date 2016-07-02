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
  uchar* d_noisy_volume;
  float* d_gathered4dstack;
  uint* d_nstacks_pow;
  uint3float1* d_stacks;
  uint* d_nstacks;
  float* d_group_weights;
  int width, height, depth, size;
  int twidth, theight, tdepth, tsize;
  uint gather_stack_sum;

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
      depth(depth),
      d_noisy_volume(NULL),
      d_gathered4dstack(NULL),
      d_stacks(NULL),
      d_nstacks_pow(NULL),
      d_nstacks(NULL)
  {
    noisy_volume = in_noisy_volume;
    size = width*height*depth;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&d_prop, device));

    // Memory allocation
    checkCudaErrors(cudaMalloc((void**)&d_noisy_volume, sizeof(uchar)*size));
    std::cout<<"Allocated "<<sizeof(uchar)*size<<" bytes for d_noisy_volume"<<std::endl;

    twidth  = std::floor((width - 1)  / params.step_size + 1);
    theight = std::floor((height - 1) / params.step_size + 1);
    tdepth  = std::floor((depth - 1)  / params.step_size + 1);
    tsize   = twidth * theight * tdepth;

    uint3float1* tmp_arr = new uint3float1[params.maxN*tsize];
    checkCudaErrors(cudaMalloc((void**)&d_stacks, sizeof(uint3float1)*(params.maxN *tsize)));
    std::cout << "Allocated " << sizeof(uint3float1)*(params.maxN*tsize) << " bytes for d_stacks" << std::endl;
    checkCudaErrors(cudaMemcpy(d_stacks, tmp_arr, sizeof(uint3float1)*params.maxN*tsize, cudaMemcpyHostToDevice));
    delete[] tmp_arr;

    checkCudaErrors(cudaMalloc((void**)&d_nstacks, sizeof(uint3float1)*(tsize)));
    std::cout << "Allocated " << (tsize) << " elements for d_nstacks" << std::endl;

    checkCudaErrors(cudaMalloc((void**)&d_nstacks_pow, sizeof(uint3float1)*(tsize)));
    std::cout << "Allocated " << (tsize) << " elements for d_nstacks_pow" << std::endl;
  };

  inline ~BM4D(){
    // Cleanup
    if (d_noisy_volume){
      checkCudaErrors(cudaFree(d_noisy_volume));
      std::cout << "Cleaned up d_noisy_volume" << std::endl;
    }
    //if (d_denoised_volume){
    //  checkCudaErrors(cudaFree(d_denoised_volume));
    //  std::cout << "Cleaned up d_denoised_volume" << std::endl;
    //}
    //if (d_stacks){
    //  checkCudaErrors(cudaFree(d_stacks));
    //  std::cout << "Cleaned up d_stacks" << std::endl;
    //}
    if (d_nstacks){
      checkCudaErrors(cudaFree(d_nstacks));
      std::cout << "Cleaned up d_nstacks" << std::endl;
    }
    if (d_gathered4dstack){
      checkCudaErrors(cudaFree(d_gathered4dstack));
      std::cout << "Cleaned up bytes of d_gathered4dstack" << std::endl;
    }
    cudaDeviceReset();
  };

  std::vector<unsigned char> run_first_step();

};
