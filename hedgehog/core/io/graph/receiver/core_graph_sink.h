//
// Created by anb22 on 5/9/19.
//

#ifndef HEDGEHOG_CORE_GRAPH_SINK_H
#define HEDGEHOG_CORE_GRAPH_SINK_H

#include "../../task/receiver/core_task_multi_receivers.h"
template<class GraphOutput>
class CoreGraphSink : public CoreTaskMultiReceivers<GraphOutput>, public Node {

 public:
  CoreGraphSink() : CoreNode("Sink", NodeType::Sink, 1),
                    CoreSlot("Sink", NodeType::Sink, 1),
                    CoreReceiver<GraphOutput>("Sink", NodeType::Sink, 1),
                    CoreTaskMultiReceivers<GraphOutput>("Sink", NodeType::Sink, 1) {
    HLOG_SELF(0, "Creating CoreGraphSink")
  }

  ~CoreGraphSink() override {
    HLOG_SELF(0, "Destructing CoreGraphSink")
  }

  void copyWholeNode([[maybe_unused]]std::shared_ptr<std::multimap<std::string,
                                                                   std::shared_ptr<Node>>> &insideNodesGraph) final {}
  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
    }
  }

  Node *getNode() override {
    return this;
  }

  void waitForNotification() override {
    std::unique_lock<std::mutex> lock(*(this->slotMutex()));

    HLOG_SELF(2, "Wait for the notification")

    this->notifyConditionVariable()->wait(lock,
                                          [this]() {
                                            return !this->receiversEmpty() || this->numberInputNodes() == 0;
                                          });
    HLOG_SELF(2, "Notification received")
  }

  CoreNode *getCore() override {
    return this;
  }
};

#endif //HEDGEHOG_CORE_GRAPH_SINK_H
