//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_QUEUE_MULTI_RECEIVERS_H
#define HEDGEHOG_CORE_QUEUE_MULTI_RECEIVERS_H

#include "../../base/receiver/core_multi_receivers.h"
#include "core_queue_slot.h"
#include "core_queue_receiver.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Multi receivers for nodes possessing a queue of data
/// @tparam NodeInputs Node input types
template<class ...NodeInputs>
class CoreQueueMultiReceivers
    : public CoreMultiReceivers<NodeInputs...>, public CoreQueueSlot, public CoreQueueReceiver<NodeInputs> ... {
 public:

  // Suppress wrong static analysis, Constructor called used
  /// @brief CoreQueueMultiReceivers constructor
  /// @param name Node's name
  /// @param type Node's type
  /// @param numberThreads Node's number of thread
  explicit CoreQueueMultiReceivers(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads),
      CoreSlot(name, type, numberThreads),
      CoreReceiver<NodeInputs>(name, type, numberThreads)...,
      CoreMultiReceivers<NodeInputs...>(name, type, numberThreads),
      CoreQueueSlot(name, type, numberThreads),
      CoreQueueReceiver<NodeInputs>(name, type, numberThreads)
  ... {
    HLOG_SELF(0, "Creating CoreQueueMultiReceivers with type: " << (int) type << " and name: " << name)
  }

  /// @brief CoreQueueMultiReceivers destructor
  ~CoreQueueMultiReceivers() override {HLOG_SELF(0, "Destructing CoreQueueMultiReceivers")}

  /// @brief Test emptiness of all receivers
  /// @return True if all receivers are empty, else False
  bool receiversEmpty() final {
    HLOG_SELF(2, "Test all destinations empty")
    return (static_cast<CoreReceiver<NodeInputs> *>(this)->receiverEmpty() && ...);
  }

  /// @brief Sums the queue sizes for all receivers
  /// @return Sum of all queue sizes for all receivers
  size_t totalQueueSize() final { return (static_cast<CoreReceiver<NodeInputs> *>(this)->queueSize() + ...); }

  /// @brief Return a set of slots, {this}
  /// @return {this}
  std::set<CoreSlot *> getSlots() final { return {this}; }

  /// @brief Return the node's slot
  /// @return this
  CoreQueueSlot *queueSlot() final { return this; };

  // Suppress wrong static analysis
  /// @brief Copy the inner structure of all receivers and the slot from rhs to this
  /// @param rhs CoreQueueMultiReceivers to copy to this
  void copyInnerStructure(CoreQueueMultiReceivers<NodeInputs...> *rhs) {
    HLOG_SELF(0, "Copy Cluster information from " << rhs->name() << "(" << rhs->id() << ")")
    (CoreQueueReceiver < NodeInputs > ::copyInnerStructure(rhs),...);
    CoreQueueSlot::copyInnerStructure(rhs);
  }
};

}
#endif //HEDGEHOG_CORE_QUEUE_MULTI_RECEIVERS_H
