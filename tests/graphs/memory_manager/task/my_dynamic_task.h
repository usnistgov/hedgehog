//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_MY_DYNAMIC_TASK_H
#define HEDGEHOG_MY_DYNAMIC_TASK_H

#include "../data/dynamic_matrix_data.h"

class MyDynamicTask : public AbstractTask<DynamicMatrixData<int>, int> {
 public:
  MyDynamicTask() : AbstractTask("Dynamic Task", 2) {}
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    auto mem = this->getManagedMemory();
    mem->data(new int[30]());
    addResult(mem);
  }
  std::shared_ptr<AbstractTask<DynamicMatrixData<int>, int>> copy() override {
    return std::make_shared<MyDynamicTask>();
  }
};

#endif //HEDGEHOG_MY_DYNAMIC_TASK_H
