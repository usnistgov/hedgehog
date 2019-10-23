//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_NOTIFIER_H
#define HEDGEHOG_CORE_NOTIFIER_H

#include <memory>
#include <set>
#include <algorithm>
#include <ostream>

#include "../receiver/core_slot.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Core Notifier interface, emit notification to CoreSlot
class CoreNotifier : public virtual CoreNode {
 public:
  /// @brief Deleted default constructor
  CoreNotifier() = delete;

  /// @brief Notifier constructor
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Node number of threads
  CoreNotifier(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreNotifier with type: " << (int) type << " and name: " << name)
  }

  /// @brief Notifier destructor
  ~CoreNotifier() override {HLOG_SELF(0, "Destructing CoreNotifier")}

  /// @brief Interface to add a CoreSlot to this notifier
  /// @param slot CoreSlot to add to this notifier
  virtual void addSlot(CoreSlot *slot) = 0;

  /// @brief Interface to remove a CoreSlot from this notifier
  /// @param slot CoreSlot to remove from this notifier
  virtual void removeSlot(CoreSlot *slot) = 0;

  /// @brief Notify all slot that the node is terminated
  virtual void notifyAllTerminated() = 0;
};

}
#endif //HEDGEHOG_CORE_NOTIFIER_H
