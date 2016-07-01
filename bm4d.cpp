#include "bm4d.h"

std::vector<unsigned char> BM4D::run_first_step()
{

  Stopwatch copyingtodevice(true);
  copy_image_to_device();
  copyingtodevice.stop(); std::cout<<"Copying to device took: "<<copyingtodevice.getSeconds()<<std::endl;
  
  std::cout << "Width " << width << " height " << height << " depth " << depth << std::endl;
  std::cout << "Size " <<size << std::endl;

  uchar* d_denoised_volume;
  checkCudaErrors(cudaMalloc((void**)&d_denoised_volume, sizeof(uchar)*size));
  checkCudaErrors(cudaMemset((void*)d_denoised_volume,0, sizeof(uchar)*size));

  // Do block matching
  Stopwatch blockmatching(true);
  run_block_matching(d_noisy_volume, make_uint3(width, height, depth), make_uint3(twidth, theight, tdepth), params, d_stacks, d_nstacks, d_denoised_volume);
  blockmatching.stop(); std::cout<<"Blockmatching took: "<<blockmatching.getSeconds()<<std::endl;

  // Gather cubes together
  int gathered_size; // TODO: remove
  Stopwatch gatheringcubes(true);
  gather_cubes(d_noisy_volume, make_uint3(width, height, depth), make_uint3(twidth, theight, tdepth), params, d_stacks, d_nstacks, d_gathered4dstack, d_nstacks_pow, gathered_size);
  std::cout << "Acquied size " << gathered_size << std::endl;
  gatheringcubes.stop(); std::cout << "Gathering cubes took: " << gatheringcubes.getSeconds() << std::endl;

  debug_kernel(d_gathered4dstack);
  //base_volume.resize(deb_size*params.patch_size*params.patch_size*params.patch_size);
  //std::cout << base_volume.size() << std::endl;
  //checkCudaErrors(cudaMemcpy((void*)base_volume.data(), (void*)d_gathered4dstack, sizeof(uchar)*base_volume.size(), cudaMemcpyDeviceToHost));
  //CImg<uchar> test1(base_volume.data(), width, height, deb_size, params.patch_size, 1); test1.display();

  // Perform 3D DCT
  run_dct3d(d_gathered4dstack, gathered_size, params.patch_size);

  // Do WHT in 4th dim + Hard Thresholding + IWHT
  //run_wht_ht_iwht();

  // Perform inverse 3D DCT
  run_idct3d(d_gathered4dstack, gathered_size, params.patch_size);

  // Aggregate
  //run_aggregation();

  // d_noisy_volume -> working_image
  //Stopwatch copyingtohost(true);
  //copy_image_to_host();
  //copyingtohost.stop(); std::cout<<"Copying to host took: "<<copyingtohost.getSeconds()<<std::endl;
  //CImg<float> test2(noisy_volume.data(), width, height, depth, 1); test2.display();

  //uchar* patches = new uchar[deb_size*params.patch_size*params.patch_size*params.patch_size];
  //checkCudaErrors(cudaMemcpy((void*)patches, (void*)d_gathered4dstack, sizeof(uchar)*deb_size*params.patch_size*params.patch_size*params.patch_size, cudaMemcpyDeviceToHost));
  //CImg<uchar> test2(patches, params.patch_size*params.patch_size, params.patch_size, deb_size, 1); test2.display();
  //delete[] patches;
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