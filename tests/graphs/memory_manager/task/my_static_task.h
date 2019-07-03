//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_MY_STATIC_TASK_H
#define HEDGEHOG_MY_STATIC_TASK_H

#include "../data/matrix_data.h"

class MyStaticTask : public AbstractTask<MatrixData<int>, int> {
 public:
  MyStaticTask() : AbstractTask("Static Task", 2) {}
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    addResult(this->getManagedMemory());
  }
  std::shared_ptr<AbstractTask<MatrixData<int>, int>> copy() override {
    return std::make_shared<MyStaticTask>();
  }
};

#endif //HEDGEHOG_MY_STATIC_TASK_H
