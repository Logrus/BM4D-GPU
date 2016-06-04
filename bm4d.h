#pragma once
#include "CImg.h"
#include "parameters.h"
#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include "kernels.cuh"
using namespace cimg_library;
#ifndef idx3
#define idx3(x,y,z,x_size,y_size) ((x) + ((y)+(y_size)*(z))*(x_size))
#endif

class BM4D {
private:
  // Main variables
  CImg<unsigned char> noisy_volume;
  CImg<unsigned char> base_volume;

  // Devide variables
  float *d_noisy_volume;
  float *d_denoised_volume;
  int width, height, depth;

  // Parameters for launching kernels
  dim3 block;
  dim3 grid;

  // Main methods
  void run_block_matching();
  void run_dct3d();
  void run_wht_ht_iwht();
  void run_idct3d();
  void run_aggregation();

public:
  inline BM4D(Parameters p) {};
  CImg<unsigned char> run_first_step();
  std::vector<unsigned char> run_first_step(const std::vector<unsigned char> &noisy_volume, const int &width, int &height, int &depth);

};
