//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_INPUT_TASK_2_H
#define HEDGEHOG_INPUT_TASK_2_H

#include "../../../hedgehog/hedgehog.h"
#include "../types/a.h"

class InputTask2 : public AbstractTask<A, int, float> {
 public:
  InputTask2() : AbstractTask("InputTask2", 10) {}

  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
    addResult(std::make_shared<A>((*input) + 1));
  }

  void execute([[maybe_unused]]std::shared_ptr<float> input) override {
    addResult(std::make_shared<A>((*input) + 1));
  }

  std::shared_ptr<AbstractTask<A, int, float>> copy() override {
    return std::make_shared<InputTask2>();
  }
};
#endif //HEDGEHOG_INPUT_TASK_2_H
