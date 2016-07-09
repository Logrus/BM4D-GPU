#define cimg_use_tiff
#pragma comment( lib, "libtiff.lib" )
#pragma comment( lib, "libtiff-bcc.lib" )
#include <string>
#include <iostream>
#include "allreader.h"
#include "bm4d.h"
#include "parameters.h"
#include "stopwatch.hpp"
#include "CImg.h"
using namespace cimg_library;


float psnr(std::vector<unsigned char>& gt, std::vector<unsigned char>& noisy)
{
  float max_signal = 255;
  float sqr_err = 0;
  for (int i = 0; i<gt.size(); ++i)
  {
    float diff = gt[i] - noisy[i];
    sqr_err += diff*diff;
  }
  float mse = sqr_err / gt.size();
  float psnr = 10.f*log10(max_signal*max_signal / mse);
  return psnr;
}

int main(int argc, char *argv[]){
  // Take parameters
  Parameters p;
  read_parameters(argc, argv, p);

  // Define variables
  std::vector<unsigned char> gt;
  std::vector<unsigned char> noisy_image;
  int width, height, depth;

  // Load volume
  AllReader reader(false); // true - show debug video on load
  Stopwatch loading_file(true); // true - start right away
  reader.read(p.filename, noisy_image, width, height, depth);
  loading_file.stop(); std::cout<<"Loading file took: "<<loading_file.getSeconds()<<std::endl;
  std::cout << "Volume size: (" << width << ", " << height << ", " << depth<< ") total: " << width*height*depth << std::endl;
  //CImg<unsigned char> test2(noisy_image.data(), width, height, depth, 1); test2.display();
  //test2.save_tiff("noisy.tiff");
  //reader.read("gt/t.txt", gt, width, height, depth);


  // Run first step of BM4D
  Stopwatch bm4d_timing(true); // true - start right away
  BM4D filter(p, noisy_image, width, height, depth);
  std::vector<unsigned char> denoised_image = filter.run_first_step();
  bm4d_timing.stop(); std::cout<<"BM4D total time: "<<bm4d_timing.getSeconds()<<std::endl;
  
  // Save image
  CImg<unsigned char> test(denoised_image.data(), width, height, depth, 1, 1); //test.display();
  test.save_tiff(p.out_filename.c_str());
  //std::cout << "PSNR noisy: " << psnr(gt, noisy_image) << std::endl;
  //std::cout << "PSNR denoised: " << psnr(gt, denoised_image) << std::endl;
  //std::cout << "PSNR reconstructed: " << psnr(noisy_image, denoised_image) << std::endl;
}
