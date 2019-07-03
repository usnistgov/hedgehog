//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_B_H
#define HEDGEHOG_B_H

class B {
 private:
  int taskCount_ = 0;
 public:
  explicit B(int taskCount) : taskCount_(taskCount) {}
  int taskCount() const { return taskCount_; }
};

#endif //HEDGEHOG_B_H
