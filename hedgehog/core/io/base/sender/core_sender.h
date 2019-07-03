//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_SENDER_H
#define HEDGEHOG_CORE_SENDER_H

#include <set>
#include "core_notifier.h"
#include "../../../node/core_node.h"
#include "../receiver/core_receiver.h"

template<class Output>
class CoreSender : public virtual CoreNotifier {
 public:
  CoreSender(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreNode(name,
                                                                                                       type,
                                                                                                       numberThreads) {
    HLOG_SELF(0, "Creating CoreSender with type: " << (int) type << " and name: " << name)
  }
  ~CoreSender() override {
    HLOG_SELF(0, "Destructing CoreSender")
  }

  virtual void addReceiver(CoreReceiver<Output> *) = 0;
  virtual void removeReceiver(CoreReceiver<Output> *) = 0;
  virtual void sendAndNotify(std::shared_ptr<Output>) = 0;
  virtual std::set<CoreSender<Output> *> getSenders() = 0;

  void duplicateEdge(CoreNode *duplicateNode,
                     std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
    std::cout << "***" << this->name() << " / " << this->id() << "is not a real sender, going inside of it: "
              << std::endl;
    for (auto sender : this->getSenders()) {
      sender->duplicateEdge(duplicateNode, correspondenceMap);
    }
  }
};

#endif //HEDGEHOG_CORE_SENDER_H
