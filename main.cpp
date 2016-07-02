#include <string>
#include <iostream>
#include "allreader.h"
#include "bm4d.h"
#include "parameters.h"
#include "CImg.h"
#include "stopwatch.hpp"
using namespace cimg_library;

int main(int argc, char *argv[]){
  // Take parameters
  Parameters p;
  read_parameters(argc, argv, p);

  // Define variables
  std::vector<unsigned char> noisy_image;
  int width, height, depth;

  // Load volume
  AllReader reader(false); // true - show debug video on load
  Stopwatch loading_file(true); // true - start right away
  reader.read(p.filename, noisy_image, width, height, depth);
  loading_file.stop(); std::cout<<"Loading file took: "<<loading_file.getSeconds()<<std::endl;
  std::cout << "Volume size: (" << width << ", " << height << ", " << depth<< ") total: " << width*height*depth << std::endl;

  // Run first step of BM4D
  Stopwatch bm4d_timing(true); // true - start right away
  BM4D filter(p, noisy_image, width, height, depth);
  std::vector<unsigned char> denoised_image = filter.run_first_step();
  bm4d_timing.stop(); std::cout<<"BM4D total time: "<<bm4d_timing.getSeconds()<<std::endl;
  
  // Save image
  CImg<unsigned char> test(denoised_image.data(), width, height, depth, 1, 1); test.display();
}
