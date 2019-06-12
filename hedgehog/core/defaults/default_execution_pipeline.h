//
// Created by anb22 on 6/10/19.
//

#ifndef HEDGEHOG_DEFAULT_EXECUTION_PIPELINE_H
#define HEDGEHOG_DEFAULT_EXECUTION_PIPELINE_H

#include "../node/execution_pipeline/core_execution_pipeline.h"
#include "../../behaviour/switch_rule.h"

#if defined (__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif //__clang//
template<class GraphInput, class GraphOutput, class ...GraphInputs>
class DefaultExecutionPipelineExecute : public virtual CoreExecutionPipeline<GraphOutput, GraphInputs...> {
 public:
  DefaultExecutionPipelineExecute(std::string_view const &name,
                                  AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline,
                                  std::shared_ptr<Graph<GraphOutput, GraphInputs...>> baseGraph,
                                  size_t numberGraphs,
                                  std::vector<int> const &deviceIds,
                                  bool automaticStart = false
  ) : CoreExecutionPipeline<GraphOutput, GraphInputs...>(name,
                                                         executionPipeline,
                                                         baseGraph,
                                                         numberGraphs,
                                                         deviceIds,
                                                         automaticStart) {}
  void callExecute([[maybe_unused]]std::shared_ptr<GraphInput> data) override {
    if (this->callSendToGraph<GraphInput>(data, this->baseCoreGraph_->graphId())) {
      static_cast<CoreGraphReceiver<GraphInput> *>(this->baseCoreGraph_)->receive(data);
      this->baseCoreGraph_->wakeUp();
    }

    for (auto graph : this->epGraphs_) {
      if (this->callSendToGraph<GraphInput>(data, graph->graphId())) {
        std::static_pointer_cast<CoreGraphReceiver<GraphInput>>(graph)->receive(data);
        graph->wakeUp();
      }
    }
  }

 private:
  template<class Input>
  bool callSendToGraph(std::shared_ptr<Input> &data, size_t const &graphId) {
    return static_cast<SwitchRule<Input> *>(this->executionPipeline())->sendToGraph(data, graphId);
  }

};
#if defined (__clang__)
#pragma clang diagnostic pop
#endif //__clang//

template<class GraphOutput, class ...GraphInputs>
class DefaultExecutionPipeline : public DefaultExecutionPipelineExecute<GraphInputs, GraphOutput, GraphInputs...> ... {
 public:
  using DefaultExecutionPipelineExecute<GraphInputs, GraphOutput, GraphInputs...>::callExecute...;
  DefaultExecutionPipeline(
      std::string_view const &name,
      AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline,
      std::shared_ptr<Graph<GraphOutput, GraphInputs...>> baseGraph,
      size_t numberGraphs,
      std::vector<int> const &deviceIds,
      bool automaticStart = false
  ) :
      CoreNode(name, NodeType::ExecutionPipeline, 1),
      CoreNotifier(name, NodeType::ExecutionPipeline, 1),
      CoreQueueSender<GraphOutput>(name, NodeType::ExecutionPipeline, 1),
      CoreSlot(name, NodeType::ExecutionPipeline, 1),
      CoreReceiver<GraphInputs>(name, NodeType::ExecutionPipeline, 1)...,
      CoreExecutionPipeline<GraphOutput,
                            GraphInputs...>(name,
                                            executionPipeline,
                                            baseGraph,
                                            numberGraphs,
                                            deviceIds,
                                            automaticStart),
      DefaultExecutionPipelineExecute<GraphInputs,
                                      GraphOutput,
                                      GraphInputs...>(name,
                                                      executionPipeline,
                                                      baseGraph,
                                                      numberGraphs,
                                                      deviceIds,
                                                      automaticStart)
  ... {}
  void postRun() override {
    (removeSwitchReceiver<GraphInputs>(this->baseCoreGraph_), ...);
    (this->baseCoreGraph_->removeNotifier(static_cast<CoreQueueSender<GraphInputs> *>(this->coreSwitch_)), ...);

    for (std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> graph : this->epGraphs_) {
      (removeSwitchReceiver<GraphInputs>(graph.get()), ...);
      (graph->removeNotifier(static_cast<CoreQueueSender<GraphInputs> *>(this->coreSwitch_)), ...);
    }

    this->coreSwitch_->notifyAllTerminated();
  }

 private:
  template<class GraphInput>
  void removeSwitchReceiver(CoreGraphReceiver<GraphInput> *coreGraphReceiver) {
    for (auto r : coreGraphReceiver->receivers()) {
      static_cast<CoreQueueSender<GraphInput> *>(this->coreSwitch_)->removeReceiver(r);

    }

    coreGraphReceiver->removeSender(static_cast<CoreQueueSender<GraphInput> *>(this->coreSwitch_));
  }
};

#endif //HEDGEHOG_DEFAULT_EXECUTION_PIPELINE_H
