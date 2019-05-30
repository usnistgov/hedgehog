//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_SENDER_H
#define HEDGEHOG_SENDER_H

#include <memory>
#include "../../core/node/core_task.h"
#include "../../core/node/core_graph.h"

template<class Output>
class Sender : public virtual Node {
 public:
  using output_t = Output;
};

#endif //HEDGEHOG_SENDER_H
