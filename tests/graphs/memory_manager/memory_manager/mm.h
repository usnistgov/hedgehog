//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_DYNAMIC_MM_H
#define HEDGEHOG_DYNAMIC_MM_H

#include "../data/dynamic_matrix_data.h"

template<class T>
class MM : public AbstractMemoryManager<DynamicMatrixData<T>> {
 public:
  MM(size_t const &poolSize) : AbstractMemoryManager<DynamicMatrixData<T>>(poolSize) {}

  std::shared_ptr<AbstractMemoryManager<DynamicMatrixData<T>>> copy() override {
    return std::make_shared<MM<T>>(this->poolSize());
  }

  bool canRecycle([[maybe_unused]]std::shared_ptr<DynamicMatrixData<T>> const &ptr) override { return true; }
};

#endif //HEDGEHOG_DYNAMIC_MM_H
