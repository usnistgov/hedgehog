//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_SLOT_H
#define HEDGEHOG_CORE_SLOT_H

#include "../../../node/core_node.h"

/// @brief Hedgehog core namespace
namespace hh::core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of CoreNotifier
class CoreNotifier;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Slot interface, receive notification from CoreNotifier
class CoreSlot : public virtual CoreNode {
 public:

  /// @brief Core slot constructor
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Node number of threads
  CoreSlot(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreSlot with type: " << (int) type << " and name: " << name)
  }

  /// @brief Core Slot destructor
  ~CoreSlot() override {HLOG_SELF(0, "Destructing CoreSlot")}

  /// @brief Interface to add a CoreNotifier to this slot
  /// @param notifier CoreNotifier to add to this slot
  virtual void addNotifier(CoreNotifier *notifier) = 0;

  /// @brief Interface to remove a CoreNotifier from this slot
  /// @param notifier CoreNotifier to remove from this notifier
  virtual void removeNotifier(CoreNotifier *notifier) = 0;

  /// @brief Test if notifiers are connected to this slot
  /// @return True if at least one notifier is connected to this slot, else False
  virtual bool hasNotifierConnected() = 0;

  /// @brief Return the number of notifiers connected to this slot
  /// @return The number of notifiers connected to this slot
  [[nodiscard]] virtual size_t numberInputNodes() const = 0;

  /// @brief Interface to define what the node do when it receive a signal
  virtual void wakeUp() = 0;

  /// @brief Interface to define how the node wait for a signal, and return if the node is terminated
  /// @return True if the node is terminated, else False
  virtual bool waitForNotification() = 0;
};

}
#endif //HEDGEHOG_CORE_SLOT_H
