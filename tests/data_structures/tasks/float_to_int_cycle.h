//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_FLOAT_TO_INT_CYCLE_H
#define HEDGEHOG_TESTS_FLOAT_TO_INT_CYCLE_H

#include "../../../hedgehog/hedgehog.h"

class FloatToIntCycle : public hh::AbstractTask<int, float> {
 private:
  int count_ = 0;

 public:
  explicit FloatToIntCycle(size_t numberThreads) : AbstractTask("Task Cycle", numberThreads) {}
  virtual ~FloatToIntCycle() = default;

  void execute(std::shared_ptr<float> ptr) override {
    if (count_ != 3 * 100) {
      addResult(std::make_shared<int>(*ptr));
      count_++;
    }
  }

  std::shared_ptr<AbstractTask<int, float>> copy() override {
    return std::make_shared<FloatToIntCycle>(this->numberThreads());
  }

  bool canTerminate() override { return count_ == 3 * 100; }
};

#endif //HEDGEHOG_TESTS_FLOAT_TO_INT_CYCLE_H
