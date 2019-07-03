//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_OUTPUT_TASK_2_H
#define HEDGEHOG_OUTPUT_TASK_2_H

#include "../../../hedgehog/hedgehog.h"
#include "../types/a.h"

class OutputTask2 : public AbstractTask<A, int> {
 public:
  OutputTask2() : AbstractTask("OutputTask2", 10) {}
  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
    addResult(std::make_shared<A>((*input) + 1));
  }
  std::shared_ptr<AbstractTask<A, int>> copy() override {
    return std::make_shared<OutputTask2>();
  }
};

#endif //HEDGEHOG_OUTPUT_TASK_2_H
