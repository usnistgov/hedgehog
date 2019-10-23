//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_DYNAMIC_MEMORY_MANAGER_H
#define HEDGEHOG_TESTS_DYNAMIC_MEMORY_MANAGER_H

#include "../datas/dynamic_memory_manage_data.h"

template<class T>
class DynamicMemoryManager : public hh::AbstractMemoryManager<DynamicMemoryManageData<T>> {
 public:
  explicit DynamicMemoryManager(size_t const &poolSize)
      : hh::AbstractMemoryManager<DynamicMemoryManageData<T>>(poolSize) {}

  std::shared_ptr<hh::AbstractMemoryManager<DynamicMemoryManageData<T>>> copy() override {
    return std::make_shared<DynamicMemoryManager<T>>(this->capacity());
  }
};

#endif //HEDGEHOG_TESTS_DYNAMIC_MEMORY_MANAGER_H
