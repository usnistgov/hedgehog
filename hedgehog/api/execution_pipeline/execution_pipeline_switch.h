//
// Created by anb22 on 5/29/19.
//

#ifndef HEDGEHOG_EXECUTION_PIPELINE_SWITCH_H
#define HEDGEHOG_EXECUTION_PIPELINE_SWITCH_H

#include "execution_pipeline_rule.h"


#include "../../core/io/task/receiver/core_task_multi_receivers.h"

template<class ...GraphInputs>
 class ExecutionPipelineSwitch : public CoreTaskMultiReceivers<GraphInputs...>, public ExecutionPipelineRule<GraphInputs>...{

 };

#endif //HEDGEHOG_EXECUTION_PIPELINE_SWITCH_H
