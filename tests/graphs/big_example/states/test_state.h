//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_TEST_STATE_H
#define HEDGEHOG_TEST_STATE_H

#include "../../../hedgehog/hedgehog.h"
#include "../types/a.h"
#include "../types/b.h"
#include "../types/c.h"

class TestState : public AbstractState<C, B, A> {
 public:
  void execute(std::shared_ptr<B> ptr) override {
    this->push(std::make_shared<C>(ptr->taskCount() + 1));
  }
  void execute(std::shared_ptr<A> ptr) override {
    this->push(std::make_shared<C>(ptr->taskCount() + 1));
  }
};

#endif //HEDGEHOG_TEST_STATE_H
