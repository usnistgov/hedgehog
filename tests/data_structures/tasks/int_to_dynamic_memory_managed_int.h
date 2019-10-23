//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_INT_TO_DYNAMIC_MEMORY_MANAGED_INT_H
#define HEDGEHOG_TESTS_INT_TO_DYNAMIC_MEMORY_MANAGED_INT_H

#include "../datas/dynamic_memory_manage_data.h"

class IntToDynamicMemoryManagedInt : public hh::AbstractTask<DynamicMemoryManageData<int>, int> {
 public:
  explicit IntToDynamicMemoryManagedInt(size_t numberThreads) : AbstractTask("Dynamic Task", numberThreads) {}
  virtual ~IntToDynamicMemoryManagedInt() = default;

  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    auto mem = this->getManagedMemory();
    mem->data(new int[30]());
    addResult(mem);
  }

  std::shared_ptr<AbstractTask<DynamicMemoryManageData<int>, int>> copy() override {
    return std::make_shared<IntToDynamicMemoryManagedInt>(this->numberThreads());
  }
};

#endif //HEDGEHOG_TESTS_INT_TO_DYNAMIC_MEMORY_MANAGED_INT_H
