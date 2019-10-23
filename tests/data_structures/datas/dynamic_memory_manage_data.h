//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_DYNAMIC_MEMORY_MANAGE_DATA_H
#define HEDGEHOG_TESTS_DYNAMIC_MEMORY_MANAGE_DATA_H

#include "../../../hedgehog/hedgehog.h"

template<class T>
class DynamicMemoryManageData : public hh::MemoryData<DynamicMemoryManageData<T>> {
 private:
  T *data_ = nullptr;

 public:
  DynamicMemoryManageData() = default;
  virtual ~DynamicMemoryManageData() = default;
  void data(T *data) { data_ = data; }
  void recycle() override { delete[] data_; }
};

#endif //HEDGEHOG_TESTS_DYNAMIC_MEMORY_MANAGE_DATA_H
