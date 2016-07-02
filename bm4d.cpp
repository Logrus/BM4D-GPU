#include "bm4d.h"

std::vector<unsigned char> BM4D::run_first_step()
{

  Stopwatch copyingtodevice(true);
  copy_image_to_device();
  copyingtodevice.stop(); std::cout<<"Copying to device took: "<<copyingtodevice.getSeconds()<<std::endl;
  
  std::cout << "Width " << width << " height " << height << " depth " << depth << std::endl;
  std::cout << "Size " <<size << std::endl;

  // Do block matching
  Stopwatch blockmatching(true);
  run_block_matching(d_noisy_volume, make_uint3(width, height, depth), make_uint3(twidth, theight, tdepth), params, d_stacks, d_nstacks);
  blockmatching.stop(); std::cout<<"Blockmatching took: "<<blockmatching.getSeconds()<<std::endl;

  // Gather cubes together
  int gathered_size; 
  Stopwatch gatheringcubes(true);
  gather_cubes(d_noisy_volume, make_uint3(width, height, depth), make_uint3(twidth, theight, tdepth), params, d_stacks, d_nstacks, d_gathered4dstack, d_nstacks_pow, gathered_size);
  std::cout << "Acquied size " << gathered_size << std::endl;
  gatheringcubes.stop(); std::cout << "Gathering cubes took: " << gatheringcubes.getSeconds() << std::endl;
  debug_kernel(d_gathered4dstack);

  // Perform 3D DCT
  run_dct3d(d_gathered4dstack, gathered_size, params.patch_size);
  debug_kernel(d_gathered4dstack);

  // Do WHT in 4th dim + Hard Thresholding + IWHT
  run_wht_ht_iwht(d_gathered4dstack, gathered_size, params.patch_size, d_nstacks_pow, make_uint3(twidth, theight, tdepth));

  // Perform inverse 3D DCT
  run_idct3d(d_gathered4dstack, gathered_size, params.patch_size);

  // Aggregate
  float* final_image = new float[size];
  run_aggregation(final_image, make_uint3(width, height, depth), make_uint3(twidth, theight, tdepth), d_gathered4dstack, d_stacks, d_nstacks_pow, d_group_weights, params);

  // d_noisy_volume -> working_image
  //Stopwatch copyingtohost(true);
  //copy_image_to_host();
  //copyingtohost.stop(); std::cout<<"Copying to host took: "<<copyingtohost.getSeconds()<<std::endl;
  //CImg<float> test2(noisy_volume.data(), width, height, depth, 1); test2.display();

  CImg<float> test2(final_image, width, height, depth, 1); test2.display();
  for (int i = 0; i < size; i++){ noisy_volume[i] = static_cast<uchar>(final_image[i]); }
  delete[] final_image;
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