//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_A_H
#define HEDGEHOG_A_H

class A {
 private:
  int taskCount_ = 0;
 public:
  explicit A(int taskCount) : taskCount_(taskCount) {}
  int taskCount() const { return taskCount_; }
};

#endif //HEDGEHOG_A_H
