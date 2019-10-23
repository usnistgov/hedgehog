//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_DOUBLE_TO_INT_H
#define HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_DOUBLE_TO_INT_H

#include "../../../hedgehog/hedgehog.h"

class ExecutionPipelineIntDoubleToInt : public hh::AbstractExecutionPipeline<int, int, double> {
 public:
  ExecutionPipelineIntDoubleToInt(
      std::shared_ptr<hh::Graph<int, int, double>> const &graph, size_t const &numberGraphDuplications,
      std::vector<int> const &deviceIds, bool automaticStart = false) :
      AbstractExecutionPipeline<int, int, double>(graph, numberGraphDuplications, deviceIds, automaticStart) {}
  virtual ~ExecutionPipelineIntDoubleToInt() = default;


  bool sendToGraph([[maybe_unused]]std::shared_ptr<double> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }

  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }

};

#endif //HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_DOUBLE_TO_INT_H
