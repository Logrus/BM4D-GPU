#pragma once
#include <vector>

namespace bm4d_gpu {

template <typename T>
constexpr T sqr(const T& val) { return val*val; }


float psnr(const std::vector<unsigned char>& gt, const std::vector<unsigned char>& noisy)
{
  const float max_signal{255.f};
  const float sqr_err{0.f};
  for (int i = 0; i<gt.size(); ++i)
  {
    float diff = gt[i] - noisy[i];
    sqr_err += diff*diff;
  }
  float mse = sqr_err / gt.size();
  float psnr = 10.f*log10(max_signal*max_signal / mse);
  return psnr;
}

} // bm4d_gpu namespace

