//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_MY_STATE_H
#define HEDGEHOG_MY_STATE_H

#include "../../../../hedgehog/hedgehog.h"

class MyState : public AbstractState<float, float> {
 public:

  MyState() = default;
  void execute(std::shared_ptr<float> ptr) override { this->push(ptr); }
};

#endif //HEDGEHOG_MY_STATE_H
