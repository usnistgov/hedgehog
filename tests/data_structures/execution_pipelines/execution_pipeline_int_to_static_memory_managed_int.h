//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_TO_STATIC_MEMORY_MANAGED_INT_H
#define HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_TO_STATIC_MEMORY_MANAGED_INT_H

#include "../../../hedgehog/hedgehog.h"

#include "../datas/static_memory_manage_data.h"

class ExecutionPipelineIntToStaticMemoryManagedInt :
    public hh::AbstractExecutionPipeline<StaticMemoryManageData<int>, int> {
 public:
  ExecutionPipelineIntToStaticMemoryManagedInt(
      std::shared_ptr<hh::Graph<StaticMemoryManageData<int>, int>> const &graph,
      size_t const &numberGraphDuplications, std::vector<int> const &deviceIds)
      : AbstractExecutionPipeline(graph, numberGraphDuplications, deviceIds, false) {}
  virtual ~ExecutionPipelineIntToStaticMemoryManagedInt() = default;
  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
};

#endif //HEDGEHOG_TESTS_EXECUTION_PIPELINE_INT_TO_STATIC_MEMORY_MANAGED_INT_H
