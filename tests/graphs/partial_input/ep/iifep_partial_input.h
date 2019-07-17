//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-07-17.
//

#ifndef HEDGEHOG_IIFEP_PARTIAL_INPUT_H
#define HEDGEHOG_IIFEP_PARTIAL_INPUT_H

#include "../../../../hedgehog/hedgehog.h"

class IIFEPPartialInput : public AbstractExecutionPipeline<int, int, float> {
 public:
  IIFEPPartialInput(
  	std::shared_ptr<Graph<int, int, float>> const &graph,
  	size_t const &numberGraphDuplications,
  	std::vector<int> const &deviceIds)
  	: AbstractExecutionPipeline("EP",graph,numberGraphDuplications,deviceIds,false) {}

  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
	return true;
  }
  bool sendToGraph([[maybe_unused]]std::shared_ptr<float> &data, [[maybe_unused]]size_t const &graphId) override {
	return true;
  }
};

#endif //HEDGEHOG_IIFEP_PARTIAL_INPUT_H
