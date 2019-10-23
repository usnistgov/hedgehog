//
// Created by tjb3 on 10/21/19.
//

#ifndef HEDGEHOG_CUDA_LINK_2_EXAMPLE_H
#define HEDGEHOG_CUDA_LINK_2_EXAMPLE_H


#include <hedgehog/hedgehog.h>
#ifdef HH_USE_CUDA
class CudaLink2Example : public hh::AbstractCUDATask<int, float> {
 public:
  explicit CudaLink2Example();
  virtual ~CudaLink2Example() = default;
  void execute(std::shared_ptr<float> ptr) override;

  void initializeCuda() override;
  void shutdownCuda() override;

  std::shared_ptr<AbstractTask <int, float>> copy()
  override;
};
#endif


#endif //HEDGEHOG_CUDA_LINK_2_EXAMPLE_H
