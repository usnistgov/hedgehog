//
// Created by tjb3 on 10/21/19.
//

#include "cuda_link2_example.h"
#ifdef HH_USE_CUDA
void CudaLink2Example::execute(std::shared_ptr<float> ptr) {
  addResult(std::make_shared<int>(*ptr));
}

CudaLink2Example::CudaLink2Example() : AbstractCUDATask("CudaLink2Example", 2) {}

void CudaLink2Example::initializeCuda() {
  hh::checkCudaErrors(cudaSuccess);
  AbstractCUDATask::initializeCuda();
}

void CudaLink2Example::shutdownCuda() {
  AbstractCUDATask::shutdownCuda();
}


std::shared_ptr<hh::AbstractTask<int, float>> CudaLink2Example::copy() {
  return std::make_shared<CudaLink2Example>();
}
#endif