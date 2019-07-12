//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_GRAPH_RECEIVER_H
#define HEDGEHOG_CORE_GRAPH_RECEIVER_H

#include "../../base/receiver/core_receiver.h"

template<class GraphInput>
class CoreGraphReceiver : public virtual CoreReceiver<GraphInput> {
 private:
  std::unique_ptr<std::set<CoreReceiver<GraphInput> *>> graphReceiverInputs_ = nullptr;

 public:
  CoreGraphReceiver(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreNode(name,
                                                                                                              type,
                                                                                                              numberThreads),
                                                                                                     CoreReceiver<
                                                                                                         GraphInput>(
                                                                                                         name,
                                                                                                         type,
                                                                                                         numberThreads) {
    HLOG_SELF(0, "Creating CoreGraphReceiver with type: " << (int) type << " and name: " << name)
    this->graphReceiverInputs_ = std::make_unique<std::set<CoreReceiver<GraphInput> *>>();
  }

  virtual ~CoreGraphReceiver() {
    HLOG_SELF(0, "Destructing CoreGraphReceiver")
  }

  void addSender(CoreSender<GraphInput> *sender) final {
    HLOG_SELF(0, "Add sender " << sender->name() << "(" << sender->id() << ")")
    for (CoreReceiver<GraphInput> *inputNode: *(this->graphReceiverInputs_)) {
      inputNode->addSender(sender);
    }
  }

  void removeSender(CoreSender<GraphInput> *sender) final {
    HLOG_SELF(0, "Remove sender " << sender->name() << "(" << sender->id() << ")")
    for (CoreReceiver<GraphInput> *inputNode: *(this->graphReceiverInputs_)) {
      inputNode->removeSender(sender);
    }
  }

  void receive(std::shared_ptr<GraphInput> ptr) final {
    /////////////////////////////////////
//    std::ostringstream oss;
//	oss << this->id() << "," << this->name() << ", " << ptr;
//	oss << ":" << std::endl << "\t";
//	oss << std::endl;
//	printUnderMutex(oss.str());
	//////////////////////////////////////////
    HLOG_SELF(2, "Receive data")
    for (CoreReceiver<GraphInput> *inputNode: *(this->graphReceiverInputs_)) {

      inputNode->receive(ptr);

    }
  }

  void addGraphReceiverInput(CoreReceiver<GraphInput> *receiver) {
    HLOG_SELF(0, "Add Graph Receiver Input " << receiver->name() << "(" << receiver->id() << ")")
    this->graphReceiverInputs_->insert(receiver);
  }

  void removeGraphReceiverInput(CoreReceiver<GraphInput> *receiver) {
    HLOG_SELF(0, "Remove Graph Receiver Input " << receiver->name() << "(" << receiver->id() << ")")
    this->graphReceiverInputs_->erase(receiver);
  }

  bool receiverEmpty() final {
    auto result = true;
    for (CoreReceiver<GraphInput> *inputNode: *(this->graphReceiverInputs_)) {
      result &= inputNode->receiverEmpty();
    }
    HLOG_SELF(2, "Test receiver empty " << "(" << result << ")")
    return result;
  }

  std::set<CoreReceiver<GraphInput> *> receivers() override {
    std::set<CoreReceiver<GraphInput> *> receivers{};
    std::set<CoreReceiver<GraphInput> *> tempReceivers;
    for (CoreReceiver<GraphInput> *receiver : *(this->graphReceiverInputs_)) {
      tempReceivers = receiver->receivers();
      receivers.insert(tempReceivers.begin(), tempReceivers.end());
    }
    return receivers;
  }

  Node *node() override {
    HLOG_SELF(0, "Internal error CoreGraphReceiver node")
    exit(42);
  }
};

#endif //HEDGEHOG_CORE_GRAPH_RECEIVER_H
