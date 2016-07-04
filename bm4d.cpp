#include "bm4d.h"

std::vector<uchar> BM4D::run_first_step()
{

  Stopwatch copyingtodevice(true);
  uchar* d_noisy_volume;
  assert(size == noisy_volume.size());
  checkCudaErrors(cudaMalloc((void**)&d_noisy_volume, sizeof(uchar)*size));
  checkCudaErrors(cudaMemcpy((void*)d_noisy_volume, (void*)noisy_volume.data(), sizeof(uchar)*size, cudaMemcpyHostToDevice));
  copyingtodevice.stop(); std::cout<<"Copying to device took: "<<copyingtodevice.getSeconds()<<std::endl;
  
  std::cout << "Width " << width << " height " << height << " depth " << depth << std::endl;
  std::cout << "Size " <<size << std::endl;

  uint3 im_size = make_uint3(width, height, depth);
  uint3 tr_size = make_uint3(twidth, theight, tdepth); // Truncated size, with some step for ref patches

  // Do block matching
  Stopwatch blockmatching(true);
  run_block_matching(d_noisy_volume, im_size, tr_size, params, d_stacks, d_nstacks, d_prop);
  blockmatching.stop(); std::cout<<"Blockmatching took: "<<blockmatching.getSeconds()<<std::endl;

  // Gather cubes together
  int gather_stacks_sum; 
  Stopwatch gatheringcubes(true);
  gather_cubes(d_noisy_volume, im_size, tr_size, params, d_stacks, d_nstacks, d_gathered4dstack, gather_stacks_sum, d_prop);
  std::cout << "Acquied size " << gather_stacks_sum << std::endl;
  gatheringcubes.stop(); std::cout << "Gathering cubes took: " << gatheringcubes.getSeconds() << std::endl;
  //debug_kernel(d_gathered4dstack);

  // Perform 3D DCT
  Stopwatch dct_forward(true);
  run_dct3d(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_prop);
  dct_forward.stop(); std::cout << "3D DCT forwars took: " << dct_forward.getSeconds() << std::endl;
  //debug_kernel(d_gathered4dstack);

  // Do WHT in 4th dim + Hard Thresholding + IWHT
  float* d_group_weights;
  Stopwatch wht_t(true);
  run_wht_ht_iwht(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_nstacks, tr_size, d_group_weights, params, d_prop);
  wht_t.stop(); std::cout << "WHT took: " << wht_t.getSeconds() << std::endl;

  // Perform inverse 3D DCT
  Stopwatch dct_backward(true);
  run_idct3d(d_gathered4dstack, gather_stacks_sum, params.patch_size, d_prop);
  dct_backward.stop(); std::cout << "3D DCT forwars took: " << dct_backward.getSeconds() << std::endl;
  //debug_kernel(d_gathered4dstack);

  // Aggregate
  float* final_image = new float[width*height*depth];
  memset(final_image, 0.0, sizeof(float)*width*height*depth);
  Stopwatch aggregation_t(true);
  run_aggregation(final_image, im_size, tr_size, d_gathered4dstack, d_stacks, d_nstacks, d_group_weights, params, gather_stacks_sum);
  aggregation_t.stop(); std::cout << "3D DCT forwars took: " << aggregation_t.getSeconds() << std::endl;
  for (int i = 0; i < size; i++){ noisy_volume[i] = static_cast<uchar>(final_image[i]); }
  delete[] final_image;
  return noisy_volume;
}
