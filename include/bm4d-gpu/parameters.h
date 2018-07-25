/*
 * 2016, Vladislav Tananaev
 * tananaev@cs.uni-freiburg.de
 */
#pragma once
#include <cstdlib>  // EXIT_SUCESS, EXIT_FAILURE
#include <iomanip>
#include <iostream>
#include <string>

namespace bm4d_gpu {

struct Parameters {
  std::string input_filename;
  std::string output_filename;
  float sim_th{2500.0f};  // Similarity threshold for the first step
  float hard_th{2.7f};    // Hard schrinkage threshold

  // Can be changed but not advisable
  int window_size{5};  // Search window, barely affects the results [Lebrun M., 2013]
  int step_size{3};    // Reasonable values {1,2,3,4}
                       // Significantly (exponentially) affects speed,
                       // slightly affect results
  int gpu_device = -1;

  // Fixed in current implementation
  const int patch_size{4};  // Patch size
  const int maxN{16};       // Maximal number of the patches in one group

  bool parse(const int argc, const char const* const* argv) {
    // TODO: use boost?
    if (argc == 1) {
      return false;
    }

    if (argc >= 2) input_filename = argv[1];
    if (argc >= 3) output_filename = argv[2];
    if (argc >= 4) sim_th = std::atof(argv[3]);
    if (argc >= 5) hard_th = std::atof(argv[4]);

    if (argc >= 6) window_size = std::atoi(argv[5]);
    if (argc >= 7) step_size = std::atoi(argv[6]);

    if (argc >= 8) gpu_device = std::atoi(argv[7]);

    return true;
  }

  void printHelp() const {
    std::cout << "bm4d-gpu input_file[tiff,avi] [sim_th] [hard_th]" << std::endl;
  }

  void printParameters() const {
    std::cout << "Parameters:" << std::endl;
    std::cout << "            input file: " << input_filename << std::endl;
    std::cout << " similarity threshold: " << std::fixed << std::setprecision(3) << sim_th
              << std::endl;
    std::cout << "       hard threshold: " << std::fixed << std::setprecision(3) << hard_th
              << std::endl;
    std::cout << "          window size: " << window_size << std::endl;
    std::cout << "            step size: " << step_size << std::endl;
    std::cout << " max cubes in a group: " << maxN << std::endl;
    std::cout << "           patch size: " << patch_size << std::endl;
  }
};
}  // namespace bm4d_gpu
