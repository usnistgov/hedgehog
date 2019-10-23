//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_TO_INT_H
#define HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_TO_INT_H

#include "../../../hedgehog/hedgehog.h"

class ExecutionPipelineIntToInt : public hh::AbstractExecutionPipeline<int, int> {
 public:
  explicit ExecutionPipelineIntToInt(std::shared_ptr<hh::Graph<int, int>> const &graph,
                                     size_t const &numberGraphDuplications,
                                     std::vector<int> const &deviceIds) : AbstractExecutionPipeline(graph,
                                                                                                    numberGraphDuplications,
                                                                                                    deviceIds) {}
  virtual ~ExecutionPipelineIntToInt() = default;
  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
};

#endif //HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_TO_INT_H
