//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_TASK_MULTI_RECEIVERS_H
#define HEDGEHOG_CORE_TASK_MULTI_RECEIVERS_H

#include "../../base/receiver/core_multi_receivers.h"
#include "core_task_slot.h"
#include "core_task_receiver.h"

template<class ...TaskInputs>
class CoreTaskMultiReceivers
    : public CoreMultiReceivers<TaskInputs...>, public CoreTaskSlot, public CoreTaskReceiver<TaskInputs> ... {
 public:
  explicit
  CoreTaskMultiReceivers(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads),
      CoreSlot(name, type, numberThreads),
      CoreReceiver<TaskInputs>(name, type, numberThreads)...,
      CoreMultiReceivers<TaskInputs...>(name, type, numberThreads),
      CoreTaskSlot(name, type, numberThreads), CoreTaskReceiver<TaskInputs>(name, type, numberThreads)
  ... {
    HLOG_SELF(0, "Creating CoreTaskMultiReceivers with type: " << (int) type << " and name: " << name)
  }

  ~CoreTaskMultiReceivers() override {
    HLOG_SELF(0, "Destructing CoreTaskMultiReceivers")
  }

  bool receiversEmpty() final {
    HLOG_SELF(2, "Test all receivers empty")
    return (static_cast<CoreReceiver<TaskInputs> *>(this)->receiverEmpty() && ...);
  }

  std::set<CoreSlot *> getSlots() final {
    return {this};
  }

  CoreTaskSlot *getTaskSlot() final {
    return this;
  };

  void copyInnerStructure(CoreTaskMultiReceivers<TaskInputs...> *rhs) {
    HLOG_SELF(0, "Duplicate information from " << rhs->name() << "(" << rhs->id() << ")")
    (CoreTaskReceiver < TaskInputs > ::copyInnerStructure(rhs),...);
    CoreTaskSlot::copyInnerStructure(rhs);
  }

};

#endif //HEDGEHOG_CORE_TASK_MULTI_RECEIVERS_H
