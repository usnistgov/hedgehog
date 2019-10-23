//
// Created by tjb3 on 10/18/19.
//

#include "cuda_link_example.h"
#ifdef HH_USE_CUDA
void CudaLinkExample::execute(std::shared_ptr<int> ptr) {
  addResult(std::make_shared<float>(*ptr));
}

CudaLinkExample::CudaLinkExample() : AbstractCUDATask("CudaLinkExample", 2) {}

void CudaLinkExample::initializeCuda() {
  hh::checkCudaErrors(cudaSuccess);
  AbstractCUDATask::initializeCuda();
}

void CudaLinkExample::shutdownCuda() {
  AbstractCUDATask::shutdownCuda();
}

std::shared_ptr<hh::AbstractTask<float, int>> CudaLinkExample::copy() {
  return std::make_shared<CudaLinkExample>();
}
#endif