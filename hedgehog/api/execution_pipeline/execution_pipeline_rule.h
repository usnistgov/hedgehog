//
// Created by anb22 on 5/29/19.
//

#ifndef HEDGEHOG_EXECUTION_PIPELINE_RULE_H
#define HEDGEHOG_EXECUTION_PIPELINE_RULE_H

#include <cstdio>



#include "../../core/io/task/receiver/core_task_receiver.h"

template <class GraphInput>
class ExecutionPipelineRule {
 public:
  virtual bool sendToGraph(int GraphId) = 0;
};

#endif //HEDGEHOG_EXECUTION_PIPELINE_RULE_H
