//
// Created by anb22 on 5/29/19.
//

#ifndef HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H
#define HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H

#include "../graph.h"
#include "../../core/defaults/core_default_execution_pipeline.h"

template<class GraphOutput, class ...GraphInputs>
class AbstractExecutionPipeline
    : public MultiReceivers<GraphInputs...>, public Sender<GraphOutput>, public SwitchRule<GraphInputs> ... {
 private:
  std::shared_ptr<Graph<GraphOutput, GraphInputs...>>
      graph_ = nullptr;

  std::shared_ptr<CoreDefaultExecutionPipeline<GraphOutput, GraphInputs...>>
      coreExecutionPipeline_ = nullptr;

 public:

  AbstractExecutionPipeline() = delete;

  AbstractExecutionPipeline(std::shared_ptr<Graph<GraphOutput, GraphInputs...>> graph,
                            size_t const &numberGraphDuplications,
                            std::vector<int> const deviceIds, bool automaticStart = false)
      : graph_(graph),
        coreExecutionPipeline_(std::make_shared<CoreDefaultExecutionPipeline<GraphOutput, GraphInputs...>>(
            "AbstractExecutionPipeline",
            this,
            std::dynamic_pointer_cast<CoreGraph<GraphOutput, GraphInputs...>>(graph->core()),
            numberGraphDuplications,
            deviceIds, automaticStart)) {}

  AbstractExecutionPipeline(std::string_view const &name,
                            std::shared_ptr<Graph<GraphOutput, GraphInputs...>> const &graph,
                            size_t const &numberGraphDuplications,
                            std::vector<int> const &deviceIds,
                            bool automaticStart = false)
      : graph_(graph),
        coreExecutionPipeline_(std::make_shared<CoreDefaultExecutionPipeline<GraphOutput, GraphInputs...>>(
            name,
            this,
            std::dynamic_pointer_cast<CoreGraph<GraphOutput, GraphInputs...>>(graph->core()),
            numberGraphDuplications,
            deviceIds,
            automaticStart)
        ) {}

  ~AbstractExecutionPipeline() override = default;

  std::shared_ptr<CoreNode> core() override { return this->coreExecutionPipeline_; }

  std::shared_ptr<Graph<GraphOutput, GraphInputs...>> const &graph() const {
    return graph_;
  }
};

#endif //HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H
