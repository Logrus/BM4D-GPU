#include <string>
#include <iostream>
#include "allreader.h"
#include "bm4d.h"
#include "parameters.h"
#include "CImg.h"
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
  reader.read(p.filename, noisy_image, width, height, depth);

  // Run first step of BM4D
  BM4D filter(p);
  std::vector<unsigned char> denoised_image;
  denoised_image = filter.run_first_step(noisy_image, width, height, depth);

  // Save image
  //CImg<unsigned char> test(denoised_image.data(), width, height, depth, 1, 1);
  //test.display();
}
