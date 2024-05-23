// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.

#ifndef HEDGEHOG_INPUT_BLOCK_STATE_H
#define HEDGEHOG_INPUT_BLOCK_STATE_H
#include <ostream>

#include "../../../../hedgehog/hedgehog.h"
#include "../data/matrix_block_data.h"

template<class Type, Order Ord = Order::Row>
class InputBlockState : public hh::AbstractState<
    2,
    MatrixBlockData<Type, 'a', Ord>, MatrixBlockData<Type, 'b', Ord>,
    std::pair<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>, std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>
> {
 private:
  size_t
      gridHeightLeft_ = 0,
      gridSharedDimension_ = 0,
      gridWidthRight_ = 0;

  std::vector<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>>
      gridMatrixA_ = {};

  std::vector<std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>
      gridMatrixB_ = {};

  std::vector<size_t>
      ttlA_ = {},
      ttlB_ = {};

 public:
  InputBlockState(size_t gridHeightLeft, size_t gridSharedDimension, size_t gridWidthRight) :
      gridHeightLeft_(gridHeightLeft), gridSharedDimension_(gridSharedDimension), gridWidthRight_(gridWidthRight) {
    gridMatrixA_ = std::vector<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>>
        (gridHeightLeft_ * gridSharedDimension_, nullptr);
    gridMatrixB_ = std::vector<std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>
        (gridWidthRight_ * gridSharedDimension_, nullptr);

    ttlA_ = std::vector<size_t>(gridHeightLeft_ * gridSharedDimension_, gridWidthRight_);
    ttlB_ = std::vector<size_t>(gridWidthRight_ * gridSharedDimension_, gridHeightLeft_);
  }

  virtual ~InputBlockState() = default;

  void execute(std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> ptr) override {
    matrixA(ptr);
    for (size_t jB = 0; jB < gridWidthRight_; ++jB) {
      if (auto bB = matrixB(ptr->colIdx(), jB)) {
        ttlA_[ptr->rowIdx() * gridSharedDimension_ + ptr->colIdx()]
            = ttlA_[ptr->rowIdx() * gridSharedDimension_ + ptr->colIdx()] - 1;
        if (ttlA_[ptr->rowIdx() * gridSharedDimension_ + ptr->colIdx()] == 0) {
          gridMatrixA_[ptr->rowIdx() * gridSharedDimension_ + ptr->colIdx()] = nullptr;
        }
        auto res = std::make_shared<std::pair<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>,
                                              std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>>();
        res->first = ptr;
        res->second = bB;
        this->addResult(res);
      }
    }
  }

  void execute(std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> ptr) override {
    matrixB(ptr);
    for (size_t iA = 0; iA < gridHeightLeft_; ++iA) {
      if (auto bA = matrixA(iA, ptr->rowIdx())) {
        ttlB_[ptr->rowIdx() * gridWidthRight_ + ptr->colIdx()]
            = ttlB_[ptr->rowIdx() * gridWidthRight_ + ptr->colIdx()] - 1;
        if (ttlB_[ptr->rowIdx() * gridWidthRight_ + ptr->colIdx()] == 0) {
          gridMatrixB_[ptr->rowIdx() * gridWidthRight_ + ptr->colIdx()] = nullptr;
        }
        auto res = std::make_shared<std::pair<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>,
                                              std::shared_ptr<MatrixBlockData<Type, 'b', Ord>>>>();
        res->first = bA;
        res->second = ptr;
        this->addResult(res);
      }
    }
  }

 private:
  std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> matrixA(size_t i, size_t j) {
    std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> res = nullptr;
    if ((res = gridMatrixA_[i * gridSharedDimension_ + j])) {
      ttlA_[i * gridSharedDimension_ + j] = ttlA_[i * gridSharedDimension_ + j] - 1;
      if (ttlA_[i * gridSharedDimension_ + j] == 0) { gridMatrixA_[i * gridSharedDimension_ + j] = nullptr; }
    }
    return res;
  }

  std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> matrixB(size_t i, size_t j) {
    std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> res = nullptr;
    if ((res = gridMatrixB_[i * gridWidthRight_ + j])) {
      ttlB_[i * gridWidthRight_ + j] = ttlB_[i * gridWidthRight_ + j] - 1;
      if (ttlB_[i * gridWidthRight_ + j] == 0) { gridMatrixB_[i * gridWidthRight_ + j] = nullptr; }
    }
    return res;
  }

  void matrixA(std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> blockA) {
    gridMatrixA_[blockA->rowIdx() * gridSharedDimension_ + blockA->colIdx()] = blockA;
  }

  void matrixB(std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> blockB) {
    gridMatrixB_[blockB->rowIdx() * gridWidthRight_ + blockB->colIdx()] = blockB;
  }

 public:
  friend std::ostream &operator<<(std::ostream &os, const InputBlockState &state) {
    os << "State Input Block: " << std::endl;
    os << "Grid Matrix A" << std::endl;
    for (size_t i = 0; i < state.gridHeightLeft_; ++i) {
      for (size_t j = 0; j < state.gridSharedDimension_; ++j) {
        os << std::setw(14) << state.gridMatrixA_[i * state.gridSharedDimension_ + j] << ", ";
      }
      os << std::endl;
    }
    os << "TTL Matrix A" << std::endl;
    for (size_t i = 0; i < state.gridHeightLeft_; ++i) {
      for (size_t j = 0; j < state.gridSharedDimension_; ++j) {
        os << state.ttlA_[i * state.gridSharedDimension_ + j] << ", ";
      }
      os << std::endl;
    }
    os << "Grid Matrix B" << std::endl;
    for (size_t i = 0; i < state.gridSharedDimension_; ++i) {
      for (size_t j = 0; j < state.gridWidthRight_; ++j) {
        os << std::setw(14) << state.gridMatrixB_[i * state.gridWidthRight_ + j] << ", ";
      }
      os << std::endl;
    }
    os << "TTL Matrix B" << std::endl;
    for (size_t i = 0; i < state.gridSharedDimension_; ++i) {
      for (size_t j = 0; j < state.gridWidthRight_; ++j) {
        os << state.ttlB_[i * state.gridWidthRight_ + j] << ", ";
      }
      os << std::endl;
    }
    return os;
  }
};

#endif //HEDGEHOG_INPUT_BLOCK_STATE_H
