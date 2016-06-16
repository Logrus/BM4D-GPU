#include "bm4d.h"

std::vector<unsigned char> BM4D::run_first_step()
{
  //working_image.resize(size);
  //working_image.assign(noisy_volume.begin(), noisy_volume.end());

  Stopwatch copyingtodevice(true);
  // working_image -> d_noisy_volume
  copy_image_to_device();
  copyingtodevice.stop(); std::cout<<"Copying to device took: "<<copyingtodevice.getSeconds()<<std::endl;

  Stopwatch blockmatching(true);
  // Do block matching
  run_block_matching(d_noisy_volume, make_uint3(width, height, depth), params, d_stacks, d_nstacks);
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
  CImg<float> test1(noisy_volume.data(), width, height, depth, 1); test1.display();

  //noisy_volume.assign(working_image.begin(), working_image.end());
  return noisy_volume;
}


void BM4D::copy_image_to_device(){
  checkCudaErrors(cudaMemcpy((void*)d_noisy_volume, (void*)noisy_volume.data(), sizeof(uchar)*size, cudaMemcpyHostToDevice));
  std::cout<<"Copied noisy_volume to d_noisy_volume "<<sizeof(uchar)*size<<" bytes"<<std::endl;
}
void BM4D::copy_image_to_host(){
  checkCudaErrors(cudaMemcpy((void*)noisy_volume.data(), (void*)d_noisy_volume, sizeof(uchar)*size, cudaMemcpyDeviceToHost));
  std::cout<<"Copied d_noisy_volume to noisy_volume "<<sizeof(uchar)*size<<" bytes "<<std::endl;
}