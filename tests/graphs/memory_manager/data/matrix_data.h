//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_MATRIX_DATA_H
#define HEDGEHOG_MATRIX_DATA_H

#include "../../../../hedgehog/hedgehog.h"

template<class T>
class MatrixData : public MemoryData<MatrixData<T>> {
  T *data_ = nullptr;
  size_t
      matrixSize_ = 1024 * 1024;

 public:
  MatrixData() {}
  void data(T *data) {
    data_ = data;
  }
  size_t matrixSize() const { return matrixSize_; }
  virtual ~MatrixData() { delete[] data_; }
  void recycle() override { std::fill_n(data_, matrixSize_, 0); }
};

#endif //HEDGEHOG_MATRIX_DATA_H
