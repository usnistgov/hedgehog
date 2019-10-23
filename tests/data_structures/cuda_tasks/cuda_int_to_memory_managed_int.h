//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_CUDA_INT_TO_MEMORY_MANAGED_INT_H
#define HEDGEHOG_TESTS_CUDA_INT_TO_MEMORY_MANAGED_INT_H

#include "../../../hedgehog/hedgehog.h"
#include "../datas/static_memory_manage_data.h"
#ifdef HH_USE_CUDA
class CudaIntToStaticMemoryManagedInt : public hh::AbstractCUDATask<StaticMemoryManageData<int>, int> {
 public:
  CudaIntToStaticMemoryManagedInt() : AbstractCUDATask("CUDA Task") {}
  virtual ~CudaIntToStaticMemoryManagedInt() = default;

  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    addResult(this->getManagedMemory());
  }

  std::shared_ptr<AbstractTask<StaticMemoryManageData<int>, int>> copy() override {
    return std::make_shared<CudaIntToStaticMemoryManagedInt>();
  }
};
#endif
#endif //HEDGEHOG_TESTS_CUDA_INT_TO_MEMORY_MANAGED_INT_H
