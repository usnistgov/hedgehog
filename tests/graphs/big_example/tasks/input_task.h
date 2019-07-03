//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_INPUT_TASK_H
#define HEDGEHOG_INPUT_TASK_H

#include "../../../hedgehog/hedgehog.h"

class InputTask : public AbstractTask<double, int, float> {
 public:
  InputTask() : AbstractTask("InputTask", 10) {}

  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
    addResult(std::make_shared<double>((*input) + 1));
  }
  void execute([[maybe_unused]]std::shared_ptr<float> input) override {
    addResult(std::make_shared<double>((*input) + 1));
  }

  std::shared_ptr<AbstractTask<double, int, float>> copy() override {
    return std::make_shared<InputTask>();
  }

};

#endif //HEDGEHOG_INPUT_TASK_H
