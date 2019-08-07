//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_STATIC_MM_H
#define HEDGEHOG_STATIC_MM_H

#include "../../../../hedgehog/hedgehog.h"
#include "../data/matrix_data.h"

template<class T>
class StaticMM : public AbstractStaticMemoryManager<MatrixData<T>> {
 public:
  explicit StaticMM(size_t const &poolSize) : AbstractStaticMemoryManager<MatrixData<T>>(poolSize) {}
  ~StaticMM() override = default;

  std::shared_ptr<AbstractMemoryManager<MatrixData<T>>> copy() override {
    return std::make_shared<StaticMM>(this->poolSize());
  }

  void allocate(std::shared_ptr<MatrixData<T>> ptr) override { ptr->data(new T[ptr->matrixSize()]); }
  bool canRecycle([[maybe_unused]]std::shared_ptr<MatrixData<T>> const &ptr) override { return true; }
};

#endif //HEDGEHOG_STATIC_MM_H
