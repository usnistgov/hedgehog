//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_SLOT_H
#define HEDGEHOG_CORE_SLOT_H

#include "../../../node/core_node.h"

class CoreNotifier;

class CoreSlot : public virtual CoreNode {
 public:

  CoreSlot(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreNode(name,
                                                                                                     type,
                                                                                                     numberThreads) {
    HLOG_SELF(0, "Creating CoreSlot with type: " << (int) type << " and name: " << name)
  }

  ~CoreSlot() override {
    HLOG_SELF(0, "Destructing CoreSlot")
  }

  virtual void addNotifier(CoreNotifier *) = 0;
  virtual void removeNotifier(CoreNotifier *) = 0;
  virtual bool hasNotifierConnected() = 0;
  virtual size_t numberInputNodes() const = 0;
  virtual void wakeUp() = 0;
  virtual void waitForNotification() = 0;
  virtual std::set<CoreSlot *> getSlots() = 0;
};

#endif //HEDGEHOG_CORE_SLOT_H
