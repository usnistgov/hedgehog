//
// Created by anb22 on 6/6/19.
//

#ifndef HEDGEHOG_CORE_SWITCH_SENDER_H
#define HEDGEHOG_CORE_SWITCH_SENDER_H

#include <ostream>
#include "../../../behaviour/node.h"
#include "../../io/queue/sender/core_queue_sender.h"

template<class GraphInput>
class CoreSwitchSender : public CoreQueueSender<GraphInput> {
 public:
  CoreSwitchSender(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreQueueSender<GraphInput>(name, type, numberThreads) {}
};

#endif //HEDGEHOG_CORE_SWITCH_SENDER_H
