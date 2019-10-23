//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_INT_FLOAT_TO_DOUBLE_H
#define HEDGEHOG_TESTS_INT_FLOAT_TO_DOUBLE_H

#include "../../../hedgehog/hedgehog.h"

class IntFloatToDouble : public hh::AbstractTask<double, int, float> {
 public:
  explicit IntFloatToDouble(size_t numberThreads = 1) : AbstractTask("IntFloatToDouble", numberThreads) {}
  virtual ~IntFloatToDouble() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override { this->addResult(std::make_shared<double>()); }
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override { this->addResult(std::make_shared<double>()); }
};

#endif //HEDGEHOG_TESTS_INT_FLOAT_TO_DOUBLE_H
