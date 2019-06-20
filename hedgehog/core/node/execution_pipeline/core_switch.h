//
// Created by anb22 on 6/6/19.
//

#ifndef HEDGEHOG_CORE_SWITCH_H
#define HEDGEHOG_CORE_SWITCH_H

#include <ostream>
#include "core_switch_sender.h"

template<class ...GraphInputs>
class CoreSwitch : public CoreSwitchSender<GraphInputs> ... {
 public:

  CoreSwitch(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreNode(name, type, numberThreads),
        CoreNotifier(name, type, numberThreads),
        CoreQueueNotifier(name, type, numberThreads),
        CoreSwitchSender<GraphInputs>(name, type, numberThreads)... {}

  CoreSwitch(CoreSwitch<GraphInputs...> const &rhs) : CoreSwitch(rhs.name(), rhs.type(), rhs.numberThreads()) {}

  std::shared_ptr<CoreNode> clone() override {
    return std::make_shared<CoreSwitch<GraphInputs...>>(*this);
  }

  Node *node() override {
    HLOG_SELF(0, __PRETTY_FUNCTION__)
    exit(42);
  }
  void visit([[maybe_unused]]AbstractPrinter *printer) override {}

  void addSlot(CoreSlot *slot) override { (CoreQueueSender<GraphInputs>::addSlot(slot), ...); }
  void removeSlot(CoreSlot *slot) override { (CoreQueueSender<GraphInputs>::removeSlot(slot), ...); }
  void notifyAllTerminated() override { (CoreQueueSender<GraphInputs>::notifyAllTerminated(), ...); }
 protected:
  void duplicateEdge(CoreNode *duplicateNode,
                     std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
    (CoreQueueSender<GraphInputs>::duplicateEdge(duplicateNode, correspondenceMap), ...);
  }

 public:

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  friend std::ostream &operator<<(std::ostream &os, CoreSwitch const &aSwitch){
//    std::cout << "switch : " << aSwitch.id() << ", " << aSwitch.name() << " is connected to: "<< std::endl;
//
//    CoreSwitch const* ptrASwitch = &aSwitch;
//    CoreSwitch * rawptrASwitch = const_cast<CoreSwitch*>(ptrASwitch);
//    (dynamic_cast<CoreSwitchSender<GraphInputs> *>(rawptrASwitch)->toPrint(os), ...);
//    return os;
//  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

};

#endif //HEDGEHOG_CORE_SWITCH_H
