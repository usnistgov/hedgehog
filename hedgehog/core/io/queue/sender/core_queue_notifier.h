//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_QUEUE_NOTIFIER_H
#define HEDGEHOG_CORE_QUEUE_NOTIFIER_H

#include "../../base/sender/core_notifier.h"

class CoreQueueNotifier : public virtual CoreNotifier {
 private:
  std::shared_ptr<std::set<CoreSlot *>> slots_ = nullptr;

 public:
  CoreQueueNotifier(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreNotifier(name,
                                                                                                                  type,
                                                                                                                  numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueNotifier with type: " << (int) type << " and name: " << name)
    slots_ = std::make_shared<std::set<CoreSlot *>>();
  }

  ~CoreQueueNotifier() override {
    HLOG_SELF(0, "Destructing CoreQueueNotifier")
  }

  std::shared_ptr<std::set<CoreSlot *>> const &slots() const { return slots_; }

  void addSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Add Slot " << slot->name() << "(" << slot->id() << ")")
    this->slots()->insert(slot);
  }
  void removeSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Remove Slot " << slot->name() << "(" << slot->id() << ")")
    this->slots_->erase(slot);
  }

  void notifyAllTerminated() override {
    HLOG_SELF(2, "Notify all terminated")
    std::for_each(this->slots()->begin(),
                  this->slots()->end(),
                  [this](CoreSlot *s) { s->removeNotifier(this); });

    std::for_each(this->slots()->begin(), this->slots()->end(), [](CoreSlot *s) { s->wakeUp(); });
  }
  void copyInnerStructure(CoreQueueNotifier *rhs) {
    HLOG_SELF(0, "Duplicate CoreQueueNotifier information from " << rhs->name() << "(" << rhs->id() << ")")
    for (CoreSlot *slot : *(rhs->slots_)) {
      slot->addNotifier(this);
    }

    this->slots_ = rhs->slots_;
  }

 private:
  void slots(std::shared_ptr<std::set<CoreSlot *>> const &innerSlots) { slots_ = innerSlots; }
};

#endif //HEDGEHOG_CORE_QUEUE_NOTIFIER_H
