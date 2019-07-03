//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_DYNAMIC_MATRIX_DATA_H
#define HEDGEHOG_DYNAMIC_MATRIX_DATA_H

#include "../../../../hedgehog/hedgehog.h"

template<class T>
class DynamicMatrixData : public MemoryData<DynamicMatrixData<T>> {
  T *data_ = nullptr;

 public:
  DynamicMatrixData() {}
  virtual ~DynamicMatrixData() = default;
  void data(T *data) { data_ = data; }
  void recycle() override { delete[] data_; }
};

#endif //HEDGEHOG_DYNAMIC_MATRIX_DATA_H
