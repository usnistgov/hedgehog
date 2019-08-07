//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_CUDA_MM_H
#define HEDGEHOG_CUDA_MM_H

#include "../../../../hedgehog/api/memory_manager/abstract_static_memory_manager.h"
#include "../data/matrix_data.h"
#ifdef HH_USE_CUDA
template<class T>
class CudaMM : public AbstractStaticMemoryManager<MatrixData<T>> {
 public:
  CudaMM(size_t const &poolSize) : AbstractStaticMemoryManager<MatrixData<T>>(poolSize) {}

  bool canRecycle([[maybe_unused]]std::shared_ptr<MatrixData<T>> const &ptr) override {
    return true;
  }

  std::shared_ptr<AbstractMemoryManager<MatrixData<T>>> copy() override {
    return std::make_shared<CudaMM<T>>(this->poolSize());
  }

  void allocate(std::shared_ptr<MatrixData<T>> ptr) override {
    ptr->data(new T[ptr->matrixSize()]);
  }
};
#endif //HH_USE_CUDA
#endif //HEDGEHOG_CUDA_MM_H
