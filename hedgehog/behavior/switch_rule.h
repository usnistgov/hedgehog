//
// Created by anb22 on 6/3/19.
//

#ifndef HEDGEHOG_SWITCH_RULE_H
#define HEDGEHOG_SWITCH_RULE_H

#include <cstdio>

/// @brief Hedgehog behavior namespace
namespace hh::behavior {
/// @brief Behavior definition for dispatching data to a Graph managed by an AbstractExecutionPipeline
/// @tparam GraphInput Input data type
template<class GraphInput>
class SwitchRule {
 public:
  /// @brief Switch rule to determine if data should be sent to the graph graphId managed by an
  /// AbstractExecutionPipeline
  /// @param data Data to be sent to Graph graphId
  /// @param graphId Graph Id
  /// @return True if the data should be sent to the Graph graphId, else False
  virtual bool sendToGraph(std::shared_ptr<GraphInput> &data, size_t const &graphId) = 0;
};
}

#endif //HEDGEHOG_SWITCH_RULE_H
