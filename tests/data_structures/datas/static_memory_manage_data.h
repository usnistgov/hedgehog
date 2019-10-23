//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_STATIC_MEMORY_MANAGE_DATA_H
#define HEDGEHOG_TESTS_STATIC_MEMORY_MANAGE_DATA_H

#include "../../../hedgehog/hedgehog.h"


template<class T>
class StaticMemoryManageData : public hh::MemoryData<StaticMemoryManageData<T>> {
 private:
  size_t matrixSize_ = 1024 * 1024;
  T *data_ = nullptr;

 public:
  StaticMemoryManageData() { data_ = new T[matrixSize_](); }
  ~StaticMemoryManageData() { delete[] data_; }

  void recycle() override { std::fill_n(data_, matrixSize_, 0); }
};

#endif //HEDGEHOG_TESTS_STATIC_MEMORY_MANAGE_DATA_H
