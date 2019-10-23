//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_PARTIAL_COMPUTATION_STATE_MANAGER_H
#define HEDGEHOG_TESTS_PARTIAL_COMPUTATION_STATE_MANAGER_H

#include "hedgehog/hedgehog.h"

#include "../datas/data_type.h"
#include "../datas/matrix_block_data.h"
#include "../states/partial_computation_state.h"

template<class Type, Order Ord = Order::Row>
class PartialComputationStateManager
    : public hh::StateManager<
        std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>, std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>,
        MatrixBlockData<Type, 'c', Ord>,
        MatrixBlockData<Type, 'p', Ord>
    > {
 public:
  explicit PartialComputationStateManager(std::shared_ptr<PartialComputationState<Type, Ord>> const &state) :
      hh::StateManager<std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>,
                                 std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>,
                       MatrixBlockData<Type, 'c', Ord>,
                       MatrixBlockData<Type, 'p', Ord>>("Partial Computation State Manager", state, false){}

  bool canTerminate() override {
    this->state()->lock();
    auto ret = std::dynamic_pointer_cast<PartialComputationState<Type, Ord>>(this->state())->isDone();
    this->state()->unlock();
    return ret;
  }
};

#endif //HEDGEHOG_TESTS_PARTIAL_COMPUTATION_STATE_MANAGER_H
