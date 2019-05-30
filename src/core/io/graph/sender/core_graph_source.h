//
// Created by anb22 on 5/9/19.
//

#ifndef HEDGEHOG_CORE_GRAPH_SOURCE_H
#define HEDGEHOG_CORE_GRAPH_SOURCE_H

#include "../../task/sender/core_task_notifier.h"
#include "../../../../behaviour/node.h"

template<class ...GraphInputs>
class CoreGraphSource : public Node, public CoreTaskSender<GraphInputs> ... {
 public:
  CoreGraphSource() : CoreNode("Source", NodeType::Source, 1),
                      CoreNotifier("Source", NodeType::Source, 1),
                      CoreTaskSender<GraphInputs>("Source", NodeType::Source, 1)... {
    HLOG_SELF(0, "Creating CoreGraphSource")
  }
  ~CoreGraphSource() override {
    HLOG_SELF(0, "Destructing CoreGraphSource")
  }

  CoreNode *getCore() override {
    return this;
  }

  Node *getNode() override {
    return nullptr;
  }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
      (CoreTaskSender<GraphInputs>::visit(printer), ...);
    }
  }

  void copyWholeNode([[maybe_unused]]std::shared_ptr<std::multimap<std::string,
                                                                   std::shared_ptr<Node>>> &insideNodesGraph) final {}

  void addSlot(CoreSlot *slot) final {
    HLOG_SELF(0, "Add slot: " << slot->name() << "(" << slot->id() << ")")
    (CoreTaskSender<GraphInputs>::addSlot(slot), ...);
  }

  void removeSlot([[maybe_unused]]CoreSlot *slot) final {
    HLOG_SELF(0, "[[Should not be called]]Remove slot: " << slot->name() << "(" << slot->id() << ")")
    HLOG_SELF(0, __PRETTY_FUNCTION__)
    exit(42);
  }
  void notifyAllTerminated() final {
    HLOG_SELF(2, "Notify all terminated")
    (CoreTaskSender<GraphInputs>::notifyAllTerminated(), ...);
  }

};

#endif //HEDGEHOG_CORE_GRAPH_SOURCE_H
