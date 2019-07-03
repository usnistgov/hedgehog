//
// Created by anb22 on 5/9/19.
//

#ifndef HEDGEHOG_CORE_GRAPH_SINK_H
#define HEDGEHOG_CORE_GRAPH_SINK_H

#include "../../queue/receiver/core_queue_multi_receivers.h"
template<class GraphOutput>
class CoreGraphSink : public CoreQueueMultiReceivers<GraphOutput> {

 public:
  CoreGraphSink() : CoreNode("Sink", NodeType::Sink, 1),
                    CoreSlot("Sink", NodeType::Sink, 1),
                    CoreReceiver<GraphOutput>("Sink", NodeType::Sink, 1),
                    CoreQueueMultiReceivers<GraphOutput>("Sink", NodeType::Sink, 1) {
    HLOG_SELF(0, "Creating CoreGraphSink")
  }

  ~CoreGraphSink() override {
    HLOG_SELF(0, "Destructing CoreGraphSink")
  }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
    }
  }

  Node *node() override {
    HLOG_SELF(0, __PRETTY_FUNCTION__)
    exit(42);
  }

  std::shared_ptr<CoreNode> clone() override {
    return std::make_shared<CoreGraphSink<GraphOutput>>();
  }

  bool waitForNotification() override {
    std::unique_lock<std::mutex> lock(*(this->slotMutex()));

    HLOG_SELF(2, "Wait for the notification")

    this->notifyConditionVariable()->wait(lock,
                                          [this]() {
                                            return !this->receiversEmpty() || this->numberInputNodes() == 0;
                                          });
    HLOG_SELF(2, "Notification received")

    return true;
  }

  void duplicateEdge([[maybe_unused]]CoreNode *duplicateNode,
                     [[maybe_unused]]std::map<CoreNode *,
                                              std::shared_ptr<CoreNode>> &correspondenceMap) override {
    HLOG_SELF(0, __PRETTY_FUNCTION__)
    exit(42);
  }

};

#endif //HEDGEHOG_CORE_GRAPH_SINK_H
