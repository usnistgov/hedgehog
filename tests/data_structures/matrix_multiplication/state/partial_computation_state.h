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


#ifndef HEDGEHOG_TEST_PARTIAL_COMPUTATION_STATE_H
#define HEDGEHOG_TEST_PARTIAL_COMPUTATION_STATE_H
#include <ostream>
#include "../../../../hedgehog/hedgehog.h"
#include "../data/matrix_block_data.h"

template<class Type, Order Ord = Order::Row>
class PartialComputationState
    : public hh::AbstractState<
        2,
        MatrixBlockData<Type, 'c', Ord>, MatrixBlockData<Type, 'p', Ord>,
        std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>, std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>,
        MatrixBlockData<Type, 'c', Ord>
    > {
 private:
  size_t
      gridHeightResults_ = 0,
      gridWidthResults_ = 0;

  std::vector<std::vector<std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>>
      gridPartialProduct_ = {};

  std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>>
      gridMatrixC_ = {};

  size_t
      ttlState_ = 0;

 public:
  PartialComputationState(size_t gridHeightResults, size_t gridWidthResults, size_t ttl)
      : gridHeightResults_(gridHeightResults), gridWidthResults_(gridWidthResults), ttlState_(ttl) {
    gridPartialProduct_ =
        std::vector<std::vector<std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>>(
            gridHeightResults_ * gridWidthResults_);
    gridMatrixC_ =
        std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>>(
            gridHeightResults_ * gridWidthResults_,
            nullptr
        );
  }

  void execute(std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> ptr) override {
    auto i = ptr->rowIdx(), j = ptr->colIdx();

    if (ttlState_ == 1) {
      this->addResult(ptr);
      --ttlState_;
    } else {
      if (isPAvailable(i, j)) {
        auto res = std::make_shared<
            std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>,
                      std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>
        >();
        res->first = ptr;
        res->second = partialProduct(i, j);
        this->addResult(res);
        --ttlState_;
      } else {
        blockMatrixC(ptr);
      }
    }
  }

  void execute(std::shared_ptr<MatrixBlockData<Type, 'p', Ord>> ptr) override {
    auto i = ptr->rowIdx(), j = ptr->colIdx();
    if (isCAvailable(i, j)) {
      auto res = std::make_shared<
          std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>, std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>
      >();
      res->first = blockMatrixC(i, j);
      res->second = ptr;
      this->addResult(res);
      --ttlState_;
    } else {
      partialProduct(ptr);
    }
  }

  bool isDone() { return ttlState_ == 0; };

 private:
  bool isPAvailable(size_t i, size_t j) { return gridPartialProduct_[i * gridWidthResults_ + j].size() != 0; }
  bool isCAvailable(size_t i, size_t j) { return gridMatrixC_[i * gridWidthResults_ + j] != nullptr; }

  std::shared_ptr<MatrixBlockData<Type, 'p', Ord>> partialProduct(size_t i, size_t j) {
    assert(isPAvailable(i, j));
    std::shared_ptr<MatrixBlockData<Type, 'p', Ord>> p = gridPartialProduct_[i * gridWidthResults_ + j].back();
    gridPartialProduct_[i * gridWidthResults_ + j].pop_back();
    return p;
  }

  void partialProduct(std::shared_ptr<MatrixBlockData<Type, 'p', Ord>> p) {
    gridPartialProduct_[p->rowIdx() * gridWidthResults_ + p->colIdx()].push_back(p);
  }

  std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> blockMatrixC(size_t i, size_t j) {
    assert(isCAvailable(i, j));
    auto c = gridMatrixC_[i * gridWidthResults_ + j];
    gridMatrixC_[i * gridWidthResults_ + j] = nullptr;
    return c;
  }

  void blockMatrixC(std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> c) {
    assert(!isCAvailable(c->rowIdx(), c->colIdx()));
    gridMatrixC_[c->rowIdx() * gridWidthResults_ + c->colIdx()] = c;
  }

};

#endif //HEDGEHOG_TEST_PARTIAL_COMPUTATION_STATE_H
