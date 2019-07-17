//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-07-17.
//

#ifndef HEDGEHOG_TASK_II_PARTIAL_INPUT_H
#define HEDGEHOG_TASK_II_PARTIAL_INPUT_H

#include "../../../../hedgehog/hedgehog.h"

class TaskIIPartialInput : public AbstractTask<int, int> {
 public:
  TaskIIPartialInput() : AbstractTask() {}
  void execute(std::shared_ptr<int> ptr) override { addResult(ptr); }
  std::shared_ptr<AbstractTask<int, int>> copy() override { return std::make_shared<TaskIIPartialInput>(); }
};

#endif //HEDGEHOG_TASK_II_PARTIAL_INPUT_H
