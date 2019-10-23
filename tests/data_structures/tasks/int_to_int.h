//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_INT_TO_INT_H
#define HEDGEHOG_TESTS_INT_TO_INT_H

#include "../../../hedgehog/hedgehog.h"

class IntToInt : public hh::AbstractTask<int, int> {
 public:
  explicit IntToInt(size_t numberThreads = 1) : AbstractTask("IntToInt", numberThreads) {}
  virtual ~IntToInt() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override { this->addResult(ptr); }
  std::shared_ptr<AbstractTask<int, int>> copy() override {
    return std::make_shared<IntToInt>(this->numberThreads());
  }
};

#endif //HEDGEHOG_TESTS_INT_TO_INT_H
