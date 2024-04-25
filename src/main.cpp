// SPDX-License-Identifier: MIT
// 2024, Vladislav Tananaev

#include <bm4d-gpu/allreader.h>
#include <bm4d-gpu/bm4d.h>
#include <bm4d-gpu/parameters.h>
#include <bm4d-gpu/utils.h>
#include <bm4d-gpu/stopwatch.hpp>

#include <iostream>
#include <string>

namespace
{
  namespace bg = bm4d_gpu;
}

int main(int argc, char *argv[])
{
  // Parse parameters
  bg::Parameters parameters;
  if (!parameters.parse(argc, argv))
  {
    std::cerr << "Unable to parse input arguments!" << std::endl;
    parameters.printHelp();
    exit(EXIT_FAILURE);
  }

  parameters.printParameters();

  // Define variables
  std::vector<unsigned char> gt;
  std::vector<unsigned char> noisy_image;
  int width, height, depth;

  // Load volume
  AllReader reader(false);      // true - show debug video on load
  Stopwatch loading_file(true); // true - start right away
  reader.read(parameters.input_filename, noisy_image, width, height, depth);
  loading_file.stop();
  std::cout << "Loading file took: " << loading_file.getSeconds() << std::endl;
  std::cout << "Volume size: (" << width << ", " << height << ", " << depth
            << ") total: " << width * height * depth << std::endl;
  reader.read("gt/t.txt", gt, width, height, depth);

  // Run first step of BM4D
  Stopwatch bm4d_timing(true); // true - start right away
  BM4D filter(parameters, noisy_image, width, height, depth);
  std::vector<unsigned char> denoised_image = filter.run_first_step();
  bm4d_timing.stop();
  std::cout << "BM4D total time: " << bm4d_timing.getSeconds() << std::endl;

  reader.saveTIFF(parameters.output_filename, denoised_image, width, height, depth);

  std::cout << "PSNR noisy: " << bg::psnr(gt, noisy_image) << std::endl;
  std::cout << "PSNR denoised: " << bg::psnr(gt, denoised_image) << std::endl;
  std::cout << "PSNR reconstructed: " << bg::psnr(noisy_image, denoised_image) << std::endl;

  return EXIT_SUCCESS;
}
