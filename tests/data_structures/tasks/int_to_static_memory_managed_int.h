//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_INT_TO_STATIC_MEMORY_MANAGED_INT_H
#define HEDGEHOG_TESTS_INT_TO_STATIC_MEMORY_MANAGED_INT_H

#include "../datas/static_memory_manage_data.h"

class IntToStaticMemoryManagedInt : public hh::AbstractTask<StaticMemoryManageData<int>, int> {
 public:
  explicit IntToStaticMemoryManagedInt(size_t numberThreads) : AbstractTask("Static cast", numberThreads) {}
  virtual ~IntToStaticMemoryManagedInt() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override { addResult(this->getManagedMemory()); }
  std::shared_ptr<AbstractTask<StaticMemoryManageData<int>, int>> copy() override {
    return std::make_shared<IntToStaticMemoryManagedInt>(this->numberThreads());
  }

};

#endif //HEDGEHOG_TESTS_INT_TO_STATIC_MEMORY_MANAGED_INT_H
