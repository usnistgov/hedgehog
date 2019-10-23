//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_STATIC_MEMORY_MANAGED_INT_DYNAMIC_MEMORY_MANAGED_INT_TO_STATIC_MEMORY_MANAGED_INT_H
#define HEDGEHOG_TESTS_STATIC_MEMORY_MANAGED_INT_DYNAMIC_MEMORY_MANAGED_INT_TO_STATIC_MEMORY_MANAGED_INT_H

#include "../datas/static_memory_manage_data.h"
#include "../datas/dynamic_memory_manage_data.h"

class StaticMemoryManagedIntDynamicMemoryManagedIntToStaticMemoryManagedInt :
    public hh::AbstractTask<StaticMemoryManageData<int>, StaticMemoryManageData<int>, DynamicMemoryManageData<int>> {
 public:
  StaticMemoryManagedIntDynamicMemoryManagedIntToStaticMemoryManagedInt() : AbstractTask("Output") {}
  virtual ~StaticMemoryManagedIntDynamicMemoryManagedIntToStaticMemoryManagedInt() = default;

  void execute(std::shared_ptr<StaticMemoryManageData<int>> ptr) override {
    addResult(ptr);
  }

  void execute(std::shared_ptr<DynamicMemoryManageData<int>> ptr) override {
    ptr->returnToMemoryManager();
  }

  std::shared_ptr<
      AbstractTask<StaticMemoryManageData<int>,
                   StaticMemoryManageData<int>, DynamicMemoryManageData<int>>> copy() override {
    return std::make_shared<StaticMemoryManagedIntDynamicMemoryManagedIntToStaticMemoryManagedInt>();
  }

};

#endif //HEDGEHOG_TESTS_STATIC_MEMORY_MANAGED_INT_DYNAMIC_MEMORY_MANAGED_INT_TO_STATIC_MEMORY_MANAGED_INT_H
