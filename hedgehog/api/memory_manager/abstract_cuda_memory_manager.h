//
// Created by anb22 on 6/21/19.
//

#ifndef HEDGEHOG_ABSTRACT_CUDA_MEMORY_MANAGER_H
#define HEDGEHOG_ABSTRACT_CUDA_MEMORY_MANAGER_H
#ifdef HH_USE_CUDA
#include <cuda_runtime.h>
#endif //HH_USE_CUDA

#include "abstract_static_memory_manager.h"

#ifdef HH_USE_CUDA

template<class MANAGEDDATA>
class AbstractCUDAMemoryManager : public AbstractStaticMemoryManager<MANAGEDDATA> {
 public:
  AbstractCUDAMemoryManager() = delete;
  explicit AbstractCUDAMemoryManager(size_t const &poolSize) : AbstractStaticMemoryManager<MANAGEDDATA>(poolSize) {}
  virtual ~AbstractCUDAMemoryManager() = default;

  void initializeCUDAMemoryManager() {}

 private:
  void initializeStaticMemoryManager() final {
    cudaSetDevice(this->deviceId());
    initializeCUDAMemoryManager();
  }
};

#endif //HH_USE_CUDA
#endif //HEDGEHOG_ABSTRACT_CUDA_MEMORY_MANAGER_H
