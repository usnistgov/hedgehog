//
// Created by tjb3 on 10/18/19.
//

#ifndef HEDGEHOG_CUDA_LINK_EXAMPLE_H
#define HEDGEHOG_CUDA_LINK_EXAMPLE_H


#include <hedgehog/hedgehog.h>
#ifdef HH_USE_CUDA
class CudaLinkExample : public hh::AbstractCUDATask<float, int> {
 public:
  explicit CudaLinkExample();
  virtual ~CudaLinkExample() = default;
  void execute(std::shared_ptr<int> ptr) override;

  void initializeCuda() override;
  void shutdownCuda() override;

  std::shared_ptr<AbstractTask < float, int>> copy()
  override;
};
#endif

#endif //HEDGEHOG_CUDA_LINK_EXAMPLE_H
