//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_OUTPUT_MD_TASK_H
#define HEDGEHOG_OUTPUT_MD_TASK_H

#include "../data/matrix_data.h"
#include "../data/dynamic_matrix_data.h"

class OutputMDTask : public AbstractTask<MatrixData<int>, MatrixData<int>, DynamicMatrixData<int>> {
 public:
  OutputMDTask() : AbstractTask("output") {}

  void execute(std::shared_ptr<MatrixData<int>> ptr) override {
    addResult(ptr);
  }

  void execute(std::shared_ptr<DynamicMatrixData<int>> ptr) override {
    ptr->returnToMemoryManager();
  }

  std::shared_ptr<AbstractTask<MatrixData<int>, MatrixData<int>, DynamicMatrixData<int>>> copy() override {
    return std::make_shared<OutputMDTask>();
  }
};

#endif //HEDGEHOG_OUTPUT_MD_TASK_H
