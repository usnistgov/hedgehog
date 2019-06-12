//
// Created by anb22 on 6/6/19.
//

#ifndef HEDGEHOG_CORE_SWITCH_H
#define HEDGEHOG_CORE_SWITCH_H

#include "core_switch_sender.h"

template<class ...GraphInputs>
class CoreSwitch : public Node, public CoreSwitchSender<GraphInputs> ... {
 public:
  CoreSwitch(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreNode(name, type, numberThreads),
        CoreNotifier(name, type, numberThreads),
        CoreSwitchSender<GraphInputs>(name, type, numberThreads)... {}

  Node *node() override { return this; }
  CoreNode *core() override { return this; }
  void visit([[maybe_unused]]AbstractPrinter *printer) override {}
  void copyWholeNode([[maybe_unused]]std::shared_ptr<std::multimap<std::string,
                                                                   std::shared_ptr<Node>>> &insideNodesGraph) override {}

  void addSlot(CoreSlot *slot) override { (CoreQueueSender<GraphInputs>::addSlot(slot), ...); }
  void removeSlot(CoreSlot *slot) override { (CoreQueueSender<GraphInputs>::removeSlot(slot), ...); }
  void notifyAllTerminated() override { (CoreQueueSender<GraphInputs>::notifyAllTerminated(), ...); }
};

#endif //HEDGEHOG_CORE_SWITCH_H
