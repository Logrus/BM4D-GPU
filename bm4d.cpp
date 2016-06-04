#include "bm4d.h"

std::vector<unsigned char> BM4D::run_first_step(const std::vector<unsigned char> &noisy_volume, const int &width, int &height, int &depth){
  int v_size = width*height*depth;
  std::vector<unsigned char> result(v_size, 100);
  std::vector<float> h_result(v_size, 100);
  std::vector<float> out(v_size, 100);

  // Convert image to float
  // for(int z=0;z<depth;++z)
  //   for(int y=0;y<height;++y)
  //     for(int x=0;x<width;++x)
  //       h_result[idx3(x,y,z,width,height)]=static_cast<float>(noisy_volume[idx3(x,y,z,width,height)]);

  wrapper_simple_kernel(h_result, out, width, height, depth);

  CImg<float> test(out.data(), width, height, depth, 1); test.display();

  // Do block matching
  //run_block_matching();

  // Gather cubes together

  // Perform 3D DCT
  //run_dct3d();

  // Do WHT in 4th dim + Hard Thresholding + IWHT
  //run_wht_ht_iwht();

  // Perform inverse 3D DCT
  //run_idct3d();

  // Aggregate
  //run_aggregation();

  // Convert image to unsigned char
  for(int z=0;z<depth;++z)
    for(int y=0;y<height;++y)
      for(int x=0;x<width;++x)
        result[idx3(x,y,z,width,height)]=static_cast<unsigned char>(h_result[idx3(x,y,z,width,height)]);



  return result;
}