//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_MY_CUDA_TASK_H
#define HEDGEHOG_MY_CUDA_TASK_H
#ifdef HH_USE_CUDA
#include "../data/matrix_data.h"

class MyCUDATask : public AbstractCUDATask<MatrixData<int>, int> {
 private:
  int count = 0;
 public:
  MyCUDATask() : AbstractCUDATask("CUDA Task") {}
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    ++count;
    addResult(this->getManagedMemory());
  }
  std::string extraPrintingInformation() const override {
    return "Count " + std::to_string(count);
  }
  std::shared_ptr<AbstractTask<MatrixData<int>, int>> copy() override {
    return std::make_shared<MyCUDATask>();
  }
};
#endif //HH_USE_CUDA
#endif //HEDGEHOG_MY_CUDA_TASK_H
