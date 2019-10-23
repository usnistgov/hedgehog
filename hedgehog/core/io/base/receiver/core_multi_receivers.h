//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_MULTI_RECEIVERS_H
#define HEDGEHOG_CORE_MULTI_RECEIVERS_H

#include "core_receiver.h"
#include "core_slot.h"
#include "../../../node/core_node.h"

/// @brief Hedgehog core namespace
namespace hh::core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of sender with a queue
/// @tparam Input Sender's input type
template<class Input>
class CoreQueueSender;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Multi receiver interface, gather multiple CoreReceiver
/// @tparam Inputs Multi receiver's inputs type
template<class ...Inputs>
class CoreMultiReceivers : public virtual CoreSlot, public virtual CoreReceiver<Inputs> ... {
 public:
  /// @brief CoreMultiReceivers constructor
  /// @param name Node's name
  /// @param type Node's type
  /// @param numberThreads Node's number of thread
  CoreMultiReceivers(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreSlot(name, type, numberThreads), CoreReceiver<Inputs>(name, type, numberThreads)... {
    HLOG_SELF(0, "Creating CoreMultiReceivers with type: " << (int) type << " and name: " << name)
  }

  /// @brief CoreMultiReceivers destructor
  ~CoreMultiReceivers() override {HLOG_SELF(0, "Destructing CoreMultiReceivers")}

  /// @brief Test if all receivers are empty
  /// @return True of all receivers are empty, else false
  virtual bool receiversEmpty() = 0;

  /// @brief Compute all receivers queue size
  /// @return All receivers queue size
  virtual size_t totalQueueSize() { return 0; }

  /// @brief Remove all coreNode's senders from this
  /// @param coreNode CoreNode representing the senders that will be removed from this
  void removeForAllSenders(CoreNode *coreNode) {
    (this->removeForAllSendersConditional<Inputs>(coreNode), ...);
  }

 private:
  /// @brief Remove all coreNode's senders from this for a specific Input type
  /// @tparam Input Sender Input's type
  /// @param coreNode CoreNode representing the senders that will be removed from this
  template<class Input>
  void removeForAllSendersConditional(CoreNode *coreNode) {
    // If coreNode *is* a CoreQueueSender for a specific Input type
    if (auto temp = dynamic_cast<CoreQueueSender<Input> *>(coreNode)) {
      // Remove the sender for this multi receiver
      static_cast<CoreReceiver<Input> *>(this)->removeSender(temp);
    }
  }

};

}
#endif //HEDGEHOG_CORE_MULTI_RECEIVERS_H
