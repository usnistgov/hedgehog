//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_INT_FLOAT_TO_INT_H
#define HEDGEHOG_TESTS_INT_FLOAT_TO_INT_H

#include "../../../hedgehog/hedgehog.h"

class IntFloatToInt : public hh::AbstractTask<int, int, float> {
 public:
  explicit IntFloatToInt(size_t numberThreads = 1) : AbstractTask("IntFloatToInt", numberThreads) {}
  virtual ~IntFloatToInt() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override { this->addResult(std::make_shared<int>()); }
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override { this->addResult(std::make_shared<int>()); }
  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
    return std::make_shared<IntFloatToInt>(this->numberThreads());
  }
};

#endif //HEDGEHOG_TESTS_INT_FLOAT_TO_INT_H
