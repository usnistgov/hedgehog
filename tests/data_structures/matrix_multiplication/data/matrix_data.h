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

#ifndef HEDGEHOG_MATRIXDATA_H
#define HEDGEHOG_MATRIXDATA_H

#include <iostream>
#include <memory>
#include <cmath>
#include <iomanip>
#include "data_type.h"

template <class Type, char Id = '0', Order Ord = Order::Row>
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
  MatrixData(size_t matrixHeight, size_t matrixWidth, size_t blockSize, Type *matrixData):
      matrixHeight_(matrixHeight), matrixWidth_(matrixWidth), blockSize_(blockSize),
      numBlocksRows_((size_t)std::ceil(matrixHeight_ / blockSize_) + (matrixHeight_ % blockSize_ == 0 ? 0: 1)),
      numBlocksCols_((size_t)std::ceil(matrixWidth_ / blockSize_) + (matrixWidth_ % blockSize_ == 0 ? 0: 1)),
      matrixData_(matrixData) {
    if(blockSize_ == 0) { blockSize_ = 1; }
    if(matrixHeight_ == 0 || matrixWidth_ == 0){ std::cout << "Can't compute an empty matrix" << std::endl; }
    if(Ord == Order::Row){ this->leadingDimension_ = matrixWidth_;}
    else{ this->leadingDimension_ = matrixHeight_; }
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
    if(Ord == Order::Row){
      for(size_t i = 0; i < data.matrixHeight(); ++i){
        for(size_t j = 0; j < data.matrixWidth(); ++j) {
          os << std::setprecision(std::numeric_limits<Type >::digits10 + 1) << data
          .matrixData_[i * data.leadingDimension() + j] << " ";
        }
        os << std::endl;
      }
    }else{
      for(size_t i = 0; i < data.matrixHeight(); ++i){
        for(size_t j = 0; j < data.matrixWidth(); ++j) {
          os << std::setprecision(std::numeric_limits<Type >::digits10 + 1) << data
              .matrixData_[j * data.leadingDimension() + i] << " ";
        }
        os << std::endl;
      }
    }
    os << std::endl;
    return os;
  }

};
#endif //HEDGEHOG_MATRIXDATA_H
