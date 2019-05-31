//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_NOTIFIER_H
#define HEDGEHOG_CORE_NOTIFIER_H

#include <memory>
#include <set>
#include <algorithm>

#include "../receiver/core_slot.h"

class CoreNotifier : public virtual CoreNode {
 public:
  CoreNotifier(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreNode(name,
                                                                                                         type,
                                                                                                         numberThreads) {
    HLOG_SELF(0, "Creating CoreNotifier with type: " << (int) type << " and name: " << name)
  }

  ~CoreNotifier() override {
    HLOG_SELF(0, "Destructing CoreNotifier")
  }

 public:
  virtual void addSlot(CoreSlot *slot) = 0;
  virtual void removeSlot(CoreSlot *slot) = 0;
  virtual void notifyAllTerminated() = 0;
};

#endif //HEDGEHOG_CORE_NOTIFIER_H