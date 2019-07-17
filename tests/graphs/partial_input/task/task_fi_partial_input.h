//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-07-17.
//

#ifndef HEDGEHOG_TASK_FI_PARTIAL_INPUT_H
#define HEDGEHOG_TASK_FI_PARTIAL_INPUT_H

#include "../../../../hedgehog/hedgehog.h"

class TaskFIPartialInput : public AbstractTask<int, float> {
 public:
  TaskFIPartialInput() : AbstractTask("task", 10) {}
  void execute(std::shared_ptr<float> ptr) override { addResult(std::make_shared<int>(*ptr)); }
  std::shared_ptr<AbstractTask<int, float>> copy() override { return std::make_shared<TaskFIPartialInput>(); }
};

#endif //HEDGEHOG_TASK_FI_PARTIAL_INPUT_H
