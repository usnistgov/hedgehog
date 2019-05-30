//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_GRAPH_MULTI_RECEIVERS_H
#define HEDGEHOG_CORE_GRAPH_MULTI_RECEIVERS_H

#include "../../base/receiver/core_slot.h"
#include "../../base/receiver/core_multi_receivers.h"
#include "core_graph_receiver.h"

template<class ...GraphInputs>
class CoreGraphMultiReceivers
    : public CoreMultiReceivers<GraphInputs...>, public CoreGraphReceiver<GraphInputs> ... {
 public:
  CoreGraphMultiReceivers(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreSlot(name, type, numberThreads),
        CoreMultiReceivers<GraphInputs...>(name, type, numberThreads),
        CoreGraphReceiver<GraphInputs>(name, type, numberThreads)... {
    HLOG_SELF(0, "Creating CoreGraphMultiReceivers with type: " << (int) type << " and name: " << name)
  }

  ~CoreGraphMultiReceivers() override {
    HLOG_SELF(0, "Destructing CoreGraphMultiReceivers")
  }

  bool receiversEmpty() final { return (static_cast<CoreGraphReceiver<GraphInputs> *>(this)->receiverEmpty() && ...); }

  Node *getNode() override {
    HLOG_SELF(0, __PRETTY_FUNCTION__)
    exit(42);
  }

};

#endif //HEDGEHOG_CORE_GRAPH_MULTI_RECEIVERS_H