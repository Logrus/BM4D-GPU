#include "bm4d.h"

std::vector<unsigned char> BM4D::run_first_step()
{
  working_image.resize(size);
  working_image.assign(noisy_volume.begin(), noisy_volume.end());

  Stopwatch copyingtodevice(true);
  // working_image -> d_noisy_volume
  copy_image_to_device();
  copyingtodevice.stop(); std::cout<<"Copying to device took: "<<copyingtodevice.getSeconds()<<std::endl;

  //CImg<float> test(working_image.data(), width, height, depth, 1); test.display();

  Stopwatch blockmatching(true);
  // Do block matching
  run_block_matching(d_noisy_volume, make_uint3(width, height, depth), params);
  blockmatching.stop(); std::cout<<"Blockmatching took: "<<blockmatching.getSeconds()<<std::endl;

  // Gather cubes together

  // Perform 3D DCT
  //run_dct3d();

  // Do WHT in 4th dim + Hard Thresholding + IWHT
  //run_wht_ht_iwht();

  // Perform inverse 3D DCT
  //run_idct3d();

  // Aggregate
  //run_aggregation();

  // d_noisy_volume -> working_image
  Stopwatch copyingtohost(true);
  copy_image_to_host();
  copyingtohost.stop(); std::cout<<"Copying to host took: "<<copyingtohost.getSeconds()<<std::endl;

  noisy_volume.assign(working_image.begin(), working_image.end());
  return noisy_volume;
}


void BM4D::copy_image_to_device(){
  checkCudaErrors(cudaMemcpy((void*)d_noisy_volume, (void*)working_image.data(), sizeof(float)*size, cudaMemcpyHostToDevice));
}
void BM4D::copy_image_to_host(){
  checkCudaErrors(cudaMemcpy((void*)working_image.data(), (void*) d_denoised_volume, sizeof(float)*size, cudaMemcpyDeviceToHost));
}