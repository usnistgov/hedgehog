//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_MY_TASK_2_H
#define HEDGEHOG_MY_TASK_2_H

#include "../../../../hedgehog/hedgehog.h"

class MyTask2 : public AbstractTask<int, float> {
 private:
  int count_ = 0;

 public:
  MyTask2(const std::string_view &name, size_t numberThreads) : AbstractTask(name, numberThreads), count_(0) {}

  void execute(std::shared_ptr<float> ptr) override {
    if (count_ != 3 * 100) {
      addResult(std::make_shared<int>(*ptr));
      count_++;
    }
  }

  std::shared_ptr<AbstractTask<int, float>> copy() override {
    return std::make_shared<MyTask2>(this->name(), this->numberThreads());
  }

  bool canTerminate() override { return count_ == 3 * 100; }
};

#endif //HEDGEHOG_MY_TASK_2_H
