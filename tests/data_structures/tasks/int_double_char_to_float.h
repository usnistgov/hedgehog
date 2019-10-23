//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_INT_DOUBLE_CHAR_TO_FLOAT_H
#define HEDGEHOG_TESTS_INT_DOUBLE_CHAR_TO_FLOAT_H

#include "../../../hedgehog/hedgehog.h"

class IntDoubleCharToFloat : public hh::AbstractTask<float, int, double, char> {
 public:
  explicit IntDoubleCharToFloat(size_t numberThreads = 1) : AbstractTask("IntDoubleCharToFloat", numberThreads) {}
  virtual ~IntDoubleCharToFloat() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override { this->addResult(std::make_shared<float>()); }
  void execute([[maybe_unused]]std::shared_ptr<double> ptr) override { this->addResult(std::make_shared<float>()); }
  void execute([[maybe_unused]]std::shared_ptr<char> ptr) override { this->addResult(std::make_shared<float>()); }

  std::shared_ptr<AbstractTask<float, int, double, char>> copy() override {
    return std::make_shared<IntDoubleCharToFloat>(this->numberThreads());
  }
};

#endif //HEDGEHOG_TESTS_INT_DOUBLE_CHAR_TO_FLOAT_H
