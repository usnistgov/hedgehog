//
// Created by anb22 on 5/29/19.
//

#ifndef HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H
#define HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H

#include "../graph.h"
#include "../../core/defaults/default_execution_pipeline.h"

template<class GraphOutput, class ...GraphInputs>
class AbstractExecutionPipeline
    : public MultiReceivers<GraphInputs...>, public Sender<GraphOutput>, public SwitchRule<GraphInputs> ... {
 private:
  std::shared_ptr<Graph<GraphOutput, GraphInputs...>>
      graph_ = nullptr;

  DefaultExecutionPipeline<GraphOutput, GraphInputs...> *
      coreExecutionPipeline_ = nullptr;

 public:

  AbstractExecutionPipeline() = delete;

  AbstractExecutionPipeline(std::shared_ptr<Graph<GraphOutput, GraphInputs...>> graph,
                            size_t const &numberGraphDuplications,
                            std::vector<int> const &deviceIds, bool automaticStart = false)
      : graph_(graph),
        coreExecutionPipeline_(new DefaultExecutionPipeline<GraphOutput, GraphInputs...>(
            "AbstractExecutionPipeline",
            this,
            graph,
            numberGraphDuplications,
            deviceIds, automaticStart)) {}

  AbstractExecutionPipeline(std::string_view const &name,
                            std::shared_ptr<Graph<GraphOutput, GraphInputs...>> const &graph,
                            size_t const &numberGraphDuplications,
                            std::vector<int> const &deviceIds,
                            bool automaticStart = false)
      : graph_(graph),
        coreExecutionPipeline_(new DefaultExecutionPipeline<GraphOutput, GraphInputs...>(
            name,
            this,
            graph,
            numberGraphDuplications,
            deviceIds,
            automaticStart)
        ) {}

  ~AbstractExecutionPipeline() override {
    delete this->coreExecutionPipeline_;
  };

  bool canTerminate() override {
    return !this->coreExecutionPipeline_->hasNotifierConnected();
  }

  CoreNode *core() override { return this->coreExecutionPipeline_; }

};

#endif //HEDGEHOG_ABSTRACT_EXECUTION_PIPELINE_H
