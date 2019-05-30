//
// Created by anb22 on 5/29/19.
//

#ifndef HEDGEHOG_EXECUTION_PIPELINE_H
#define HEDGEHOG_EXECUTION_PIPELINE_H


#include "../../core/io/base/sender/core_sender.h"

#include "execution_pipeline_switch.h"

template<size_t numberDuplications, class GraphOutput, class ...GraphInputs>
class ExecutionPipeline : public ExecutionPipelineSwitch<GraphInputs...>{
 public:
  void waitForNotification() override {

  }

  Node *getNode() override {
    return nullptr;
  }

  void copyWholeNode(std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>> &insideNodesGraph) override {

  }
  void visit(AbstractPrinter *printer) override {

  }

  void wakeUp() override final {
    CoreTaskSlot::wakeUp();
  }

};

#endif //HEDGEHOG_EXECUTION_PIPELINE_H
