//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_TASK_IC_H
#define HEDGEHOG_TASK_IC_H

#include "../../../hedgehog/hedgehog.h"
#include "../types/b.h"
#include "../types/c.h"

class TaskIC : public AbstractTask<int, C, B> {
 public:
  TaskIC() : AbstractTask("TaskIC", 10) {}
  void execute([[maybe_unused]]std::shared_ptr<C> input) override {
    addResult(std::make_shared<int>(input->taskCount() + 1));
  }
  void execute([[maybe_unused]]std::shared_ptr<B> input) override {
    addResult(std::make_shared<int>(input->taskCount() + 1));
  }

  std::shared_ptr<AbstractTask<int, C, B>> copy() override {
    return std::make_shared<TaskIC>();
  }
};

#endif //HEDGEHOG_TASK_IC_H
