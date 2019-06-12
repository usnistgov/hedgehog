//
// Created by anb22 on 6/3/19.
//

#ifndef HEDGEHOG_SWITCH_RULE_H
#define HEDGEHOG_SWITCH_RULE_H

#include <cstdio>
//#include "io/sender.h"

template<class GraphInput>
class SwitchRule {
 public:
  virtual ~SwitchRule() = default;
  virtual bool sendToGraph(std::shared_ptr<GraphInput> &data, size_t const &graphId) = 0;
};

#endif //HEDGEHOG_SWITCH_RULE_H
