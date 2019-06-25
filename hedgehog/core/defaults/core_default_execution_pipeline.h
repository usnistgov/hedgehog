//
// Created by anb22 on 6/10/19.
//

#ifndef HEDGEHOG_CORE_DEFAULT_EXECUTION_PIPELINE_H
#define HEDGEHOG_CORE_DEFAULT_EXECUTION_PIPELINE_H

#include "../node/execution_pipeline/core_execution_pipeline.h"
#include "../../behaviour/switch_rule.h"

#if defined (__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif //__clang//
template<class GraphInput, class GraphOutput, class ...GraphInputs>
class CoreDefaultExecutionPipelineExecute : public virtual CoreExecutionPipeline<GraphOutput, GraphInputs...> {
 public:
  CoreDefaultExecutionPipelineExecute(std::string_view const &name,
                                      AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline,
                                      std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> coreBaseGraph,
                                      size_t numberGraphs,
                                      std::vector<int> const &deviceIds,
                                      bool automaticStart = false
  ) : CoreExecutionPipeline<GraphOutput, GraphInputs...>(name,
                                                         executionPipeline,
                                                         coreBaseGraph,
                                                         numberGraphs,
                                                         deviceIds,
                                                         automaticStart) {}
  void callExecute([[maybe_unused]]std::shared_ptr<GraphInput> data) override {
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
class CoreDefaultExecutionPipeline : public CoreDefaultExecutionPipelineExecute<GraphInputs,
                                                                                GraphOutput,
                                                                                GraphInputs...> ... {
 public:
  using CoreDefaultExecutionPipelineExecute<GraphInputs, GraphOutput, GraphInputs...>::callExecute...;
  CoreDefaultExecutionPipeline(
      std::string_view const &name,
      AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline,
      std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> coreBaseGraph,
      size_t numberGraphs,
      std::vector<int> const &deviceIds,
      bool automaticStart = false) :
      CoreNode(name, NodeType::ExecutionPipeline, 1),
      CoreNotifier(name, NodeType::ExecutionPipeline, 1),
      CoreQueueNotifier(name, NodeType::ExecutionPipeline, 1),
      CoreQueueSender<GraphOutput>(name, NodeType::ExecutionPipeline, 1),
      CoreSlot(name, NodeType::ExecutionPipeline, 1),
      CoreReceiver<GraphInputs>(name, NodeType::ExecutionPipeline, 1)...,
      CoreExecutionPipeline<GraphOutput,
                            GraphInputs...>(name,
                                            executionPipeline,
                                            coreBaseGraph,
                                            numberGraphs,
                                            deviceIds,
                                            automaticStart),
      CoreDefaultExecutionPipelineExecute<GraphInputs,
                                          GraphOutput,
                                          GraphInputs...>(name,
                                                          executionPipeline,
                                                          coreBaseGraph,
                                                          numberGraphs,
                                                          deviceIds,
                                                          automaticStart)
  ... {}

  CoreDefaultExecutionPipeline(CoreDefaultExecutionPipeline<GraphOutput, GraphInputs...> const &rhs,
                               std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> baseGraph) :
      CoreNode(rhs.name(), NodeType::ExecutionPipeline, 1),
      CoreNotifier(rhs.name(), NodeType::ExecutionPipeline, 1),
      CoreQueueNotifier(rhs.name(), NodeType::ExecutionPipeline, 1),
      CoreQueueSender<GraphOutput>(rhs.name(), NodeType::ExecutionPipeline, 1),
      CoreSlot(rhs.name(), NodeType::ExecutionPipeline, 1),
      CoreReceiver<GraphInputs>(rhs.name(), NodeType::ExecutionPipeline, 1)...,
      CoreExecutionPipeline<GraphOutput, GraphInputs...>(
      rhs
  .
  name(), rhs
  .
  executionPipeline(), baseGraph, rhs
  .
  numberGraphs(), rhs
  .
  deviceIds(), rhs
  .
  automaticStart()
  ),
  CoreDefaultExecutionPipelineExecute<GraphInputs, GraphOutput, GraphInputs...>(
      rhs
  .
  name(), rhs
  .
  executionPipeline(), baseGraph, rhs
  .
  numberGraphs(), rhs
  .
  deviceIds(), rhs
  .
  automaticStart()
  )...
  {
  }

  virtual ~CoreDefaultExecutionPipeline() = default;

  std::shared_ptr<CoreNode> clone() override {
    return std::make_shared<CoreDefaultExecutionPipeline<GraphOutput, GraphInputs...>>(
        *this,
        std::dynamic_pointer_cast<CoreGraph<GraphOutput, GraphInputs...>>(
            this->baseCoreGraph()->clone()
        )
    );
  }

  void postRun() override {
    this->isActive(false);
    for (std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> graph : this->epGraphs_) {
      (removeSwitchReceiver<GraphInputs>(graph.get()), ...);
      (graph->removeNotifier(static_cast<CoreQueueSender<GraphInputs> *>(this->coreSwitch_.get())), ...);
    }
    this->coreSwitch_->notifyAllTerminated();

    for (std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> graph : this->epGraphs_) {
      graph->waitForTermination();
    }

  }

 private:
  template<class GraphInput>
  void removeSwitchReceiver(CoreGraphReceiver<GraphInput> *coreGraphReceiver) {
    for (auto r : coreGraphReceiver->receivers()) {
      static_cast<CoreQueueSender<GraphInput> *>(this->coreSwitch_.get())->removeReceiver(r);

    }
    coreGraphReceiver->removeSender(static_cast<CoreQueueSender<GraphInput> *>(this->coreSwitch_.get()));
  }
};

#endif //HEDGEHOG_CORE_DEFAULT_EXECUTION_PIPELINE_H
