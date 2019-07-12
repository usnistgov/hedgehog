//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_INNER_EP_H
#define HEDGEHOG_INNER_EP_H

#include "../../../../hedgehog/hedgehog.h"
#include "../types/a.h"

class InnerEP : public AbstractExecutionPipeline<int, double, A> {
 public:
  InnerEP(std::shared_ptr<Graph<int, double, A>> const &graph,
          size_t const &numberGraphDuplications,
          std::vector<int> const &deviceIds,
          bool automaticStart = false) : AbstractExecutionPipeline(graph,
                                                                   numberGraphDuplications,
                                                                   deviceIds,
                                                                   automaticStart) {}
  bool sendToGraph([[maybe_unused]]std::shared_ptr<double> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
  bool sendToGraph([[maybe_unused]]std::shared_ptr<A> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
};
#endif //HEDGEHOG_INNER_EP_H
