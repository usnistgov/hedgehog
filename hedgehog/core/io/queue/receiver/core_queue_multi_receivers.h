//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_QUEUE_MULTI_RECEIVERS_H
#define HEDGEHOG_CORE_QUEUE_MULTI_RECEIVERS_H

#include "../../base/receiver/core_multi_receivers.h"
#include "core_queue_slot.h"
#include "core_queue_receiver.h"

template<class ...NodeInputs>
class CoreQueueMultiReceivers
    : public CoreMultiReceivers<NodeInputs...>, public CoreQueueSlot, public CoreQueueReceiver<NodeInputs> ... {
 public:
  explicit
  CoreQueueMultiReceivers(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads),
      CoreSlot(name, type, numberThreads),
      CoreReceiver<NodeInputs>(name, type, numberThreads)...,
      CoreMultiReceivers<NodeInputs...>(name, type, numberThreads),
      CoreQueueSlot(name, type, numberThreads),
      CoreQueueReceiver<NodeInputs>(name, type, numberThreads)
  ... {
    HLOG_SELF(0, "Creating CoreQueueMultiReceivers with type: " << (int) type << " and name: " << name)
  }

  ~CoreQueueMultiReceivers() override {
    HLOG_SELF(0, "Destructing CoreQueueMultiReceivers")
  }

  bool receiversEmpty() final {
    HLOG_SELF(2, "Test all receivers empty")
    return (static_cast<CoreReceiver<NodeInputs> *>(this)->receiverEmpty() && ...);
  }

  std::set<CoreSlot *> getSlots() final {
    return {this};
  }

  CoreQueueSlot *queueSlot() final {
    return this;
  };

  void copyInnerStructure(CoreQueueMultiReceivers<NodeInputs...> *rhs) {
    HLOG_SELF(0, "Duplicate information from " << rhs->name() << "(" << rhs->id() << ")")
    (CoreQueueReceiver < NodeInputs > ::copyInnerStructure(rhs),...);
    CoreQueueSlot::copyInnerStructure(rhs);
  }

};

#endif //HEDGEHOG_CORE_QUEUE_MULTI_RECEIVERS_H
