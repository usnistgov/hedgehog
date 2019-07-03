//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_TASK_BDA_H
#define HEDGEHOG_TASK_BDA_H

#include "../../../hedgehog/hedgehog.h"
#include "../types/a.h"
#include "../types/b.h"

class TaskBDA : public AbstractTask<B, double, A> {
 public:
  TaskBDA() : AbstractTask("TaskBDA", 10) {}
  void execute([[maybe_unused]]std::shared_ptr<double> input) override {
    addResult(std::make_shared<B>((*input) + 1));
  }
  void execute([[maybe_unused]]std::shared_ptr<A> input) override {
    addResult(std::make_shared<B>(input->taskCount() + 1));
  }
  std::shared_ptr<AbstractTask<B, double, A>> copy() override {
    return std::make_shared<TaskBDA>();
  }
};

#endif //HEDGEHOG_TASK_BDA_H
