//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_OUTPUT_STATE_H
#define HEDGEHOG_TESTS_OUTPUT_STATE_H

#include <ostream>

#include "../../../hedgehog/hedgehog.h"
#include "../datas/data_type.h"
#include "../datas/matrix_block_data.h"

template<class Type, Order Ord = Order::Row>
class OutputState : public hh::AbstractState<MatrixBlockData<Type, 'c', Ord>, MatrixBlockData<Type, 'c', Ord>> {
 private:
  std::vector<size_t>
      ttl_ = {};

  size_t
      gridHeightTTL_ = 0,
      gridWidthTTL_ = 0;

 public:
  OutputState(size_t gridHeightTtl, size_t gridWidthTtl, size_t const &ttl)
      : ttl_(std::vector<size_t>(gridHeightTtl * gridWidthTtl, ttl)),
        gridHeightTTL_(gridHeightTtl), gridWidthTTL_(gridWidthTtl) {}

  virtual ~OutputState() = default;

  void execute(std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> ptr) override {
    auto i = ptr->rowIdx(), j = ptr->colIdx();
    --ttl_[i * gridWidthTTL_ + j];
    if (ttl_[i * gridWidthTTL_ + j] == 0) {
      this->push(ptr);
    }
  }

  friend std::ostream &operator<<(std::ostream &os, OutputState const &state) {
    for (size_t i = 0; i < state.gridHeightTTL_; ++i) {
      for (size_t j = 0; j < state.gridWidthTTL_; ++j) {
        os << state.ttl_[i * state.gridWidthTTL_ + j] << " ";
      }
      os << std::endl;
    }
    return os;
  }
};

#endif //HEDGEHOG_TESTS_OUTPUT_STATE_H
