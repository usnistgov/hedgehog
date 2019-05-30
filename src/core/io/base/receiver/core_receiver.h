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

template<class Input>
class CoreSender;

template<class Input>
class CoreReceiver : public virtual CoreNode {

 public:
  CoreReceiver(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreNode(name,
                                                                                                         type,
                                                                                                         numberThreads) {
    HLOG_SELF(0, "Creating CoreReceiver with type: " << (int) type << " and name: " << name)
  }

  ~CoreReceiver() override {
    HLOG_SELF(0, "Destructing CoreReceiver")
  }

  virtual void addSender(CoreSender<Input> *) = 0;
  virtual void removeSender(CoreSender<Input> *) = 0;
  virtual void receive(std::shared_ptr<Input>) = 0;
  virtual bool receiverEmpty() = 0;
  virtual size_t queueSize() { return 0; }
  virtual size_t maxQueueSize() { return 0; }

  virtual std::set<CoreReceiver<Input> *> getReceivers() = 0;
};

#endif //HEDGEHOG_CORE_RECEIVER_H
