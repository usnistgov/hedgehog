//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_QUEUE_NOTIFIER_H
#define HEDGEHOG_CORE_QUEUE_NOTIFIER_H

#include "../../base/sender/core_notifier.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Notifier of CoreQueueSlot
class CoreQueueNotifier : public virtual CoreNotifier {
 private:
  std::shared_ptr<std::set<CoreSlot *>> slots_ = nullptr; ///< Set of connected slots

 public:
  /// @brief CoreQueueNotifier constructor
  /// @param name Node's name
  /// @param type Node's type
  /// @param numberThreads Node's number of thread
  CoreQueueNotifier(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNotifier(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueNotifier with type: " << (int) type << " and name: " << name)
    slots_ = std::make_shared<std::set<CoreSlot *>>();
  }

  /// @brief CoreQueueNotifier destructor
  ~CoreQueueNotifier() override {HLOG_SELF(0, "Destructing CoreQueueNotifier")}

  /// @brief Connected slots accessor
  /// @return Connected slots
  [[nodiscard]] std::shared_ptr<std::set<CoreSlot *>> const &slots() const { return slots_; }

  /// @brief Add a slot to the set of connected slots
  /// @param slot Slot to connect
  void addSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Add Slot " << slot->name() << "(" << slot->id() << ")")
    this->slots()->insert(slot);
  }

  /// @brief Remove a slot from the set of connected slots
  /// @param slot Slot to remove
  void removeSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Remove Slot " << slot->name() << "(" << slot->id() << ")")
    this->slots_->erase(slot);
  }

  /// @brief Notify all slots that the node is terminated
  void notifyAllTerminated() override {
    HLOG_SELF(2, "Notify all terminated")
    std::for_each(this->slots()->begin(), this->slots()->end(), [this](CoreSlot *s) { s->removeNotifier(this); });
    std::for_each(this->slots()->begin(), this->slots()->end(), [](CoreSlot *s) { s->wakeUp(); });
  }

  /// @brief Copy the inner structure of the notifier (set of slots and connections)
  /// @param rhs CoreQueueNotifier to copy to this
  void copyInnerStructure(CoreQueueNotifier *rhs) {
    HLOG_SELF(0, "Copy Cluster CoreQueueNotifier information from " << rhs->name() << "(" << rhs->id() << ")")
    for (CoreSlot *slot : *(rhs->slots_)) { slot->addNotifier(this); }
    this->slots_ = rhs->slots_;
  }
};

}
#endif //HEDGEHOG_CORE_QUEUE_NOTIFIER_H
