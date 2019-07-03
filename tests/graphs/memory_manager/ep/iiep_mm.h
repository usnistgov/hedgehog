//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_IIEP_MM_H
#define HEDGEHOG_IIEP_MM_H

#include "../data/matrix_data.h"

class IIEPMM : public AbstractExecutionPipeline<MatrixData<int>, int> {
 public:
  IIEPMM(std::shared_ptr<Graph<MatrixData<int>, int>> const &graph,
         size_t const &numberGraphDuplications,
         std::vector<int> const &deviceIds,
         bool automaticStart) : AbstractExecutionPipeline(graph, numberGraphDuplications, deviceIds, automaticStart) {}
  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
};

#endif //HEDGEHOG_IIEP_MM_H
