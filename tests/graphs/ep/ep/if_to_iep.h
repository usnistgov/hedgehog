//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_IF_TO_IEP_H
#define HEDGEHOG_IF_TO_IEP_H

#include "../../../../hedgehog/hedgehog.h"

class IFToIEP : public AbstractExecutionPipeline<int, int, float> {
 public:
  IFToIEP(std::string_view const &name,
          std::shared_ptr<Graph<int, int, float>> const &graph,
          size_t const &numberGraphDuplications,
          std::vector<int> const &deviceIds) : AbstractExecutionPipeline(name,
                                                                         graph,
                                                                         numberGraphDuplications,
                                                                         deviceIds) {}
  virtual ~IFToIEP() = default;
 protected:
  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
  bool sendToGraph([[maybe_unused]]std::shared_ptr<float> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
};

#endif //HEDGEHOG_IF_TO_IEP_H
