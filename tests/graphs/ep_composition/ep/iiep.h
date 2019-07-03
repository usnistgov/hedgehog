//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_IIEP_H
#define HEDGEHOG_IIEP_H

#include "../../../../hedgehog/hedgehog.h"

std::vector<int> vDeviceIds(4, 0);

class IIEP : public AbstractExecutionPipeline<int, int> {
 public:
  explicit IIEP(std::shared_ptr<Graph<int, int>> const &graph) : AbstractExecutionPipeline(graph, 4, vDeviceIds) {}
  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
};

#endif //HEDGEHOG_IIEP_H
