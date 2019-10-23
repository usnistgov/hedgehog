//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_STATE_INT_FLOAT_TO_INT_H
#define HEDGEHOG_TESTS_STATE_INT_FLOAT_TO_INT_H

#include "hedgehog/hedgehog.h"

class StateIntFloatToInt : public hh::AbstractState<int, int, float> {
 public:
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override { this->push(std::make_shared<int>()); }
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override { this->push(std::make_shared<int>()); }
};

#endif //HEDGEHOG_TESTS_STATE_INT_FLOAT_TO_INT_H
