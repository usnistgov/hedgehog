//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_RECEIVER_H
#define HEDGEHOG_CORE_RECEIVER_H

#include <queue>
#include <set>
#include <shared_mutex>
#include <algorithm>

#include "../../../node/core_node.h"
#include "core_slot.h"

/// @brief Hedgehog core namespace
namespace hh::core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward Declaration Core Sender
/// @tparam Input Sender Type
template<class Input>
class CoreSender;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Receiver Interface, receive one data type from CoreSender
/// @tparam Input Type of data received by the CoreReceiver
template<class Input>
class CoreReceiver : public virtual CoreNode {
 public:
  /// @brief Constructor with node name, node type and number of threads for the node
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Cluster number of threads
  CoreReceiver(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreNode(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreReceiver with type: " << (int) type << " and name: " << name)
  }

  /// @brief Default destructor
  ~CoreReceiver() override {HLOG_SELF(0, "Destructing CoreReceiver")}

  /// @brief Interface to add a CoreSender to the receiver
  /// @param sender CoreSender to add to this receiver
  virtual void addSender(CoreSender<Input> *sender) = 0;

  /// @brief Interface to remove a CoreSender from the receiver
  /// @param sender CoreSender to remove from this receiver
  virtual void removeSender(CoreSender<Input> *sender) = 0;

  /// @brief Interface to receive a data
  /// @param data Data received by this receiver
  virtual void receive(std::shared_ptr<Input> data) = 0;

  /// @brief Accessor to test emptiness on the receiver
  /// @return True if the receiver has no data, Else false
  virtual bool receiverEmpty() = 0;

  /// @brief Interface to get the number of element to be treated by this node for this type, by default return 0
  /// @return Get the number of element to be treated by this node for this type
  virtual size_t queueSize() { return 0; }

  /// @brief Accessor to  all receivers connected to this receiver
  /// @return All receivers connected to this receiver
  virtual std::set<CoreReceiver<Input> *> receivers() = 0;
};

}
#endif //HEDGEHOG_CORE_RECEIVER_H
