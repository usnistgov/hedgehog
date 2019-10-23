//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_MATRIX_BLOCK_DATA_H
#define HEDGEHOG_TESTS_MATRIX_BLOCK_DATA_H

#include <iostream>
#include <memory>

#include "matrix_data.h"

template<class Type, char Id, Order Ord>
class MatrixBlockData {
 protected:
  size_t rowIdx_ = 0;
  size_t colIdx_ = 0;
  size_t blockSizeHeight_ = 0;
  size_t blockSizeWidth_ = 0;
  size_t leadingDimension_ = 0;
  Type *fullMatrixData_ = nullptr;
  Type *blockData_ = nullptr;

 public:
  MatrixBlockData() = default;
  MatrixBlockData(size_t rowIdx,
                  size_t colIdx,
                  size_t blockSizeHeight,
                  size_t blockSizeWidth,
                  size_t leadingDimension,
                  Type *fullMatrixData,
                  Type *blockData) : rowIdx_(rowIdx),
                                     colIdx_(colIdx),
                                     blockSizeHeight_(blockSizeHeight),
                                     blockSizeWidth_(blockSizeWidth),
                                     leadingDimension_(leadingDimension),
                                     fullMatrixData_(fullMatrixData),
                                     blockData_(blockData) {
    if (blockSizeHeight_ == 0 || blockSizeWidth_ == 0) {
      std::cout << "Can't compute an empty matrix block" << std::endl;
    }
  }

  MatrixBlockData(size_t rowIdx,
                  size_t colIdx,
                  size_t blockSizeHeight,
                  size_t blockSizeWidth,
                  Type *fullMatrixData,
                  Type *blockData) : rowIdx_(rowIdx),
                                     colIdx_(colIdx),
                                     blockSizeHeight_(blockSizeHeight),
                                     blockSizeWidth_(blockSizeWidth),
                                     fullMatrixData_(fullMatrixData),
                                     blockData_(blockData) {
    if (blockSizeHeight_ == 0 || blockSizeWidth_ == 0) {
      std::cout << "Can't compute an empty matrix block" << std::endl;
    }
    if (Ord == Order::Row) {
      this->leadingDimension(this->blockSizeWidth());
    } else {
      this->leadingDimension(this->blockSizeHeight());
    }
  }

  MatrixBlockData(size_t rowIdx,
                  size_t colIdx,
                  MatrixData<Type, Id, Ord> &matrix) {
    rowIdx_ = rowIdx;
    colIdx_ = colIdx;
    blockSizeHeight_ = std::min(matrix.blockSize(), matrix.matrixHeight() - (rowIdx * matrix.blockSize()));
    blockSizeWidth_ = std::min(matrix.blockSize(), matrix.matrixWidth() - (colIdx * matrix.blockSize()));

    if (blockSizeHeight_ == 0 || blockSizeWidth_ == 0) {
      std::cout << "Can't compute an empty matrix block" << std::endl;
    }

    leadingDimension_ = matrix.leadingDimension();
    fullMatrixData_ = matrix.matrixData();
    if (Ord == Order::Row) {
      blockData_ =
          matrix.matrixData() + (rowIdx * matrix.blockSize()) * matrix.leadingDimension() + colIdx * matrix.blockSize();
    } else {
      blockData_ =
          matrix.matrixData() + (colIdx * matrix.blockSize()) * matrix.leadingDimension() + rowIdx * matrix.blockSize();
    }
  }

  template<char OldId>
  explicit MatrixBlockData(MatrixBlockData<Type, OldId, Ord> &o) {
    this->rowIdx_ = o.rowIdx_;
    this->colIdx_ = o.colIdx_;
    this->blockSizeHeight_ = o.blockSizeHeight_;
    this->blockSizeWidth_ = o.blockSizeWidth_;
    this->leadingDimension_ = o.leadingDimension_;
    this->fullMatrixData_ = o.fullMatrixData_;
    this->blockData_ = o.blockData_;
  }

  template<char OldId>
  explicit MatrixBlockData(std::shared_ptr<MatrixBlockData<Type, OldId, Ord>> &o) {
    this->rowIdx_ = o->getRowIdx();
    this->colIdx_ = o->getColIdx();
    this->blockSizeHeight_ = o->getBlockSizeHeight();
    this->blockSizeWidth_ = o->getBlockSizeWidth();
    this->leadingDimension_ = o->getLeadingDimension();
    this->fullMatrixData_ = o->getFullMatrixData();
    this->blockData_ = o->getBlockData();
  }

  [[nodiscard]] size_t rowIdx() const { return rowIdx_; }
  [[nodiscard]] size_t colIdx() const { return colIdx_; }
  [[nodiscard]] size_t blockSizeHeight() const { return blockSizeHeight_; }
  [[nodiscard]] size_t blockSizeWidth() const { return blockSizeWidth_; }
  [[nodiscard]] size_t leadingDimension() const { return leadingDimension_; }
  Type *fullMatrixData() const { return fullMatrixData_; }
  Type *blockData() const { return blockData_; }

  void rowIdx(size_t rowIdx) { rowIdx_ = rowIdx; }
  void colIdx(size_t colIdx) { colIdx_ = colIdx; }
  void blockSizeHeight(size_t blockSizeHeight) { blockSizeHeight_ = blockSizeHeight; }
  void blockSizeWidth(size_t blockSizeWidth) { blockSizeWidth_ = blockSizeWidth; }
  void leadingDimension(size_t leadingDimension) { leadingDimension_ = leadingDimension; }
  void fullMatrixData(Type *fullMatrixData) { fullMatrixData_ = fullMatrixData; }
  void blockData(Type *blockData) { blockData_ = blockData; }

  friend std::ostream &operator<<(std::ostream &os, MatrixBlockData const &data) {
    os << "MatrixBlockData " << Id << " position Grid: (" << data.rowIdx_ << ", " << data.colIdx_ << ")" << std::endl;
    os << "Block: (" << data.blockSizeHeight() << ", " << data.blockSizeWidth() << ") leadingDimension="
       << data.leadingDimension_
       << std::endl;
    if (Ord == Order::Row) {
      for (size_t i = 0; i < data.blockSizeHeight(); ++i) {
        for (size_t j = 0; j < data.blockSizeWidth(); ++j) {
          os << data.blockData_[i * data.leadingDimension_ + j] << " ";
        }
        os << std::endl;
      }
    } else {
      for (size_t i = 0; i < data.blockSizeHeight(); ++i) {
        for (size_t j = 0; j < data.blockSizeWidth(); ++j) {
          os << data.blockData_[j * data.leadingDimension_ + i] << " ";
        }
        os << std::endl;
      }
    }

    return os;
  }
};

#endif //HEDGEHOG_TESTS_MATRIX_BLOCK_DATA_H
