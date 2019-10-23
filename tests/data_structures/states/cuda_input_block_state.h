//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_CUDA_INPUT_BLOCK_STATE_H
#define HEDGEHOG_TESTS_CUDA_INPUT_BLOCK_STATE_H

#include <hedgehog/hedgehog.h>
#include <ostream>

#include "../datas/cuda_matrix_block_data.h"

template<class Type>
class CudaInputBlockState : public hh::AbstractState<
    std::pair<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>, std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>,
    CudaMatrixBlockData<Type, 'a'>, CudaMatrixBlockData<Type, 'b'>
> {
 private:
  size_t
      gridHeightLeft_ = 0,
      gridSharedDimension_ = 0,
      gridWidthRight_ = 0;

  std::vector<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>>
      gridMatrixA_ = {};

  std::vector<std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>
      gridMatrixB_ = {};

  std::vector<size_t>
      ttlA_ = {},
      ttlB_ = {};

 public:
  CudaInputBlockState(size_t gridHeightLeft, size_t gridSharedDimension, size_t gridWidthRight) :
      gridHeightLeft_(gridHeightLeft), gridSharedDimension_(gridSharedDimension), gridWidthRight_(gridWidthRight) {
    gridMatrixA_ = std::vector<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>>
        (gridHeightLeft_ * gridSharedDimension_, nullptr);
    gridMatrixB_ = std::vector<std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>
        (gridWidthRight_ * gridSharedDimension_, nullptr);

    ttlA_ = std::vector<size_t>(gridHeightLeft_ * gridSharedDimension_, gridWidthRight_);
    ttlB_ = std::vector<size_t>(gridWidthRight_ * gridSharedDimension_, gridHeightLeft_);
  }

  virtual ~CudaInputBlockState() = default;

  void execute([[maybe_unused]]std::shared_ptr<CudaMatrixBlockData<Type, 'a'>> ptr) override {
    matrixA(ptr);
    for (size_t jB = 0; jB < gridWidthRight_; ++jB) {
      if (auto bB = matrixB(ptr->colIdx(), jB)) {
        ttlA_[ptr->rowIdx() * gridSharedDimension_ + ptr->colIdx()]
            = ttlA_[ptr->rowIdx() * gridSharedDimension_ + ptr->colIdx()] - 1;
        if (ttlA_[ptr->rowIdx() * gridSharedDimension_ + ptr->colIdx()] == 0) {
          gridMatrixA_[ptr->rowIdx() * gridSharedDimension_ + ptr->colIdx()] = nullptr;
        }
        auto res = std::make_shared<std::pair<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>,
                                              std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>>();
        res->first = ptr;
        res->second = bB;
        this->push(res);
      }
    }
  }

  void execute([[maybe_unused]]std::shared_ptr<CudaMatrixBlockData<Type, 'b'>> ptr) override {
    matrixB(ptr);
    for (size_t iA = 0; iA < gridHeightLeft_; ++iA) {
      if (auto bA = matrixA(iA, ptr->rowIdx())) {
        ttlB_[ptr->rowIdx() * gridWidthRight_ + ptr->colIdx()]
            = ttlB_[ptr->rowIdx() * gridWidthRight_ + ptr->colIdx()] - 1;
        if (ttlB_[ptr->rowIdx() * gridWidthRight_ + ptr->colIdx()] == 0) {
          gridMatrixB_[ptr->rowIdx() * gridWidthRight_ + ptr->colIdx()] = nullptr;
        }
        auto res = std::make_shared<std::pair<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>,
                                              std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>>();
        res->first = bA;
        res->second = ptr;
        this->push(res);
      }
    }
  }

 private:
  std::shared_ptr<CudaMatrixBlockData<Type, 'a'>> matrixA(size_t i, size_t j) {
    std::shared_ptr<CudaMatrixBlockData<Type, 'a'>> res = nullptr;
    if ((res = gridMatrixA_[i * gridSharedDimension_ + j])) {
      ttlA_[i * gridSharedDimension_ + j] = ttlA_[i * gridSharedDimension_ + j] - 1;
      if (ttlA_[i * gridSharedDimension_ + j] == 0) {
        gridMatrixA_[i * gridSharedDimension_ + j] = nullptr;

      }
    }
    return res;
  }

  std::shared_ptr<CudaMatrixBlockData<Type, 'b'>> matrixB(size_t i, size_t j) {
    std::shared_ptr<CudaMatrixBlockData<Type, 'b'>> res = nullptr;
    if ((res = gridMatrixB_[i * gridWidthRight_ + j])) {
      ttlB_[i * gridWidthRight_ + j] = ttlB_[i * gridWidthRight_ + j] - 1;
      if (ttlB_[i * gridWidthRight_ + j] == 0) { gridMatrixB_[i * gridWidthRight_ + j] = nullptr; }
    }
    return res;
  }

  void matrixA(std::shared_ptr<CudaMatrixBlockData<Type, 'a'>> blockA) {
    gridMatrixA_[blockA->rowIdx() * gridSharedDimension_ + blockA->colIdx()] = blockA;
  }

  void matrixB(std::shared_ptr<CudaMatrixBlockData<Type, 'b'>> blockB) {
    gridMatrixB_[blockB->rowIdx() * gridWidthRight_ + blockB->colIdx()] = blockB;
  }

 public:
  friend std::ostream &operator<<(std::ostream &os, const CudaInputBlockState &state) {
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

#endif //HEDGEHOG_TESTS_CUDA_INPUT_BLOCK_STATE_H
