//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_MY_TASK_H
#define HEDGEHOG_MY_TASK_H

#include "../../../../hedgehog/hedgehog.h"

class MyTask : public AbstractTask<float, int, double, float> {
 public:
  MyTask(const std::string_view &name, size_t numberThreads) : AbstractTask(name, numberThreads) {}

  void execute(std::shared_ptr<int> ptr) override { addResult(std::make_shared<float>(*ptr)); }
  void execute(std::shared_ptr<double> ptr) override { addResult(std::make_shared<float>(*ptr)); }
  void execute(std::shared_ptr<float> ptr) override { addResult(std::make_shared<float>(*ptr)); }

  std::shared_ptr<AbstractTask<float, int, double, float>> copy() override {
    return std::make_shared<MyTask>(this->name(), this->numberThreads());
  }
};

#endif //HEDGEHOG_MY_TASK_H
