//
// Created by anb22 on 5/9/19.
//

#ifndef HEDGEHOG_CORE_GRAPH_SOURCE_H
#define HEDGEHOG_CORE_GRAPH_SOURCE_H

#include "../../queue/sender/core_queue_sender.h"
#include "../../../../behaviour/node.h"

template<class ...GraphInputs>
class CoreGraphSource : public CoreQueueSender<GraphInputs> ... {
 public:
  CoreGraphSource() :
      CoreNode("Source", NodeType::Source, 1),
      CoreNotifier("Source", NodeType::Source, 1),
      CoreQueueNotifier("Source", NodeType::Source, 1),
      CoreQueueSender<GraphInputs>("Source", NodeType::Source, 1)... {
    HLOG_SELF(0, "Creating CoreGraphSource")
  }

  ~CoreGraphSource() override {
    HLOG_SELF(0, "Destructing CoreGraphSource")
  }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
      (CoreQueueSender<GraphInputs>::visit(printer), ...);
    }
  }

  void addSlot(CoreSlot *slot) final {
    HLOG_SELF(0, "Add slot: " << slot->name() << "(" << slot->id() << ")")
    (CoreQueueSender<GraphInputs>::addSlot(slot), ...);
  }

  //Test remove
//  void removeSlot([[maybe_unused]]CoreSlot *slot) final {
//    HLOG_SELF(0, "[[Should not be called]]Remove slot: " << slot->name() << "(" << slot->id() << ")")
//    HLOG_SELF(0, __PRETTY_FUNCTION__)
//    exit(42);
//  }

  void notifyAllTerminated() final {
    HLOG_SELF(2, "Notify all terminated")
    (CoreQueueSender<GraphInputs>::notifyAllTerminated(), ...);
  }
  std::shared_ptr<CoreNode> clone() override {
    return std::make_shared<CoreGraphSource<GraphInputs...>>();
  }

  Node *node() override {
    HLOG_SELF(0, __PRETTY_FUNCTION__)
    exit(42);
  }

  void duplicateEdge(CoreNode *duplicateNode,
                     std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
    (CoreQueueSender<GraphInputs>::duplicateEdge(duplicateNode, correspondenceMap), ...);
  }
};

#endif //HEDGEHOG_CORE_GRAPH_SOURCE_H
