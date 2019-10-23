//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_MATRIX_DATA_H
#define HEDGEHOG_TESTS_MATRIX_DATA_H

#include <iostream>
#include <memory>
#include <cmath>
#include <iomanip>
#include "data_type.h"

template<class Type, char Id = '0', Order Ord = Order::Row>
class MatrixData {
 private:
  size_t matrixHeight_ = 0;
  size_t matrixWidth_ = 0;
  size_t blockSize_ = 0;
  size_t numBlocksRows_ = 0;
  size_t numBlocksCols_ = 0;
  size_t leadingDimension_ = 0;
  Type *matrixData_ = nullptr;

 public:
  MatrixData(size_t matrixHeight, size_t matrixWidth, size_t blockSize, Type *matrixData) :
      matrixHeight_(matrixHeight), matrixWidth_(matrixWidth), blockSize_(blockSize),
      numBlocksRows_(std::ceil(matrixHeight_ / blockSize_) + (matrixHeight_ % blockSize_ == 0 ? 0 : 1)),
      numBlocksCols_(std::ceil(matrixWidth_ / blockSize_) + (matrixWidth_ % blockSize_ == 0 ? 0 : 1)),
      matrixData_(matrixData) {
    if (blockSize_ == 0) { blockSize_ = 1; }
    if (matrixHeight_ == 0 || matrixWidth_ == 0) { std::cout << "Can't compute an empty matrix" << std::endl; }
    if (Ord == Order::Row) { this->leadingDimension_ = matrixWidth_; }
    else { this->leadingDimension_ = matrixHeight_; }
  }

  [[nodiscard]] size_t matrixHeight() const { return matrixHeight_; }
  [[nodiscard]] size_t matrixWidth() const { return matrixWidth_; }
  [[nodiscard]] size_t blockSize() const { return blockSize_; }
  [[nodiscard]] size_t numBlocksRows() const { return numBlocksRows_; }
  [[nodiscard]] size_t numBlocksCols() const { return numBlocksCols_; }
  [[nodiscard]] size_t leadingDimension() const { return leadingDimension_; }
  Type *matrixData() const { return matrixData_; }

  friend std::ostream &operator<<(std::ostream &os, const MatrixData &data) {
    os
        << "MatrixData " << Id
        << " size: (" << data.matrixHeight() << ", " << data.matrixWidth() << ")"
        << " size Grid: (" << data.numBlocksRows() << ", " << data.numBlocksCols() << ")"
        << " blockSize: " << data.blockSize() << " leading Dimension: " << data.leadingDimension()
        << std::endl;
    if (Ord == Order::Row) {
      for (size_t i = 0; i < data.matrixHeight(); ++i) {
        for (size_t j = 0; j < data.matrixWidth(); ++j) {
          os << std::setprecision(std::numeric_limits<Type>::digits10 + 1) << data
              .matrixData_[i * data.leadingDimension() + j] << " ";
        }
        os << std::endl;
      }
    } else {
      for (size_t i = 0; i < data.matrixHeight(); ++i) {
        for (size_t j = 0; j < data.matrixWidth(); ++j) {
          os << std::setprecision(std::numeric_limits<Type>::digits10 + 1) << data
              .matrixData_[j * data.leadingDimension() + i] << " ";
        }
        os << std::endl;
      }
    }
    os << std::endl;
    return os;
  }

};

#endif //HEDGEHOG_TESTS_MATRIX_DATA_H
