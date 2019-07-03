//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_II_TASK_H
#define HEDGEHOG_II_TASK_H

#include "../../../../hedgehog/hedgehog.h"

class IITask : public AbstractTask<int, int> {
 public:
  IITask() : AbstractTask("IITask", 4, false) {}
  void execute(std::shared_ptr<int> ptr) override { addResult(ptr); }
  std::shared_ptr<AbstractTask<int, int>> copy() override { return std::make_shared<IITask>(); }
};

#endif //HEDGEHOG_II_TASK_H
