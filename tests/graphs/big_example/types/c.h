//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_C_H
#define HEDGEHOG_C_H

class C {
 private:
  int taskCount_ = 0;
 public:
  C(int taskCount) : taskCount_(taskCount) {}
  int taskCount() const { return taskCount_; }
};

#endif //HEDGEHOG_C_H
