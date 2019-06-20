//
// Created by anb22 on 5/24/19.
//

#ifndef HEDGEHOG_ABSTRACT_CUDA_ALLOCATOR_H
#define HEDGEHOG_ABSTRACT_CUDA_ALLOCATOR_H

#include <cuda_runtime.h>
#include <cstdio>
#include "../../behaviour/memory_manager/abstract_allocator.h"

template<class Data>
class AbstractCudaAllocator : public AbstractAllocator<Data> {
 private:
  int cudaId_ = 0;
 public:
  AbstractCudaAllocator() = delete;
  explicit AbstractCudaAllocator(int cudaId) : cudaId_(cudaId) {}

  void initialize() override {
    cudaSetDevice(cudaId_);
  }

};

#endif //HEDGEHOG_ABSTRACT_CUDA_ALLOCATOR_H
