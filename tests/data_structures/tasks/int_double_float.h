//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_INT_DOUBLE_FLOAT_H
#define HEDGEHOG_TESTS_INT_DOUBLE_FLOAT_H

#include "../../../hedgehog/hedgehog.h"

class IntDoubleFloat : public hh::AbstractTask<float, int, double> {
 public:
  explicit IntDoubleFloat(size_t numberThreads = 1) : AbstractTask("IntDoubleFloat", numberThreads) {}
  virtual ~IntDoubleFloat() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override { this->addResult(std::make_shared<float>()); }
  void execute([[maybe_unused]]std::shared_ptr<double> ptr) override { this->addResult(std::make_shared<float>()); }
  std::shared_ptr<AbstractTask<float, int, double>> copy() override {
    return std::make_shared<IntDoubleFloat>(this->numberThreads());
  }
};

#endif //HEDGEHOG_TESTS_INT_DOUBLE_FLOAT_H
