//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_QUEUE_SENDER_H
#define HEDGEHOG_CORE_QUEUE_SENDER_H

#include "core_queue_notifier.h"
#include "../receiver/core_queue_receiver.h"
#include "../../base/sender/core_sender.h"
#include "../../../../tools/traits.h"

template<class NodeOutput>
class CoreQueueSender : public CoreSender<NodeOutput>, public virtual CoreQueueNotifier {
  std::shared_ptr<std::set<CoreQueueReceiver<NodeOutput> *>> receivers_ = nullptr;

 public:
  CoreQueueSender(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreQueueNotifier(name, type, numberThreads),
        CoreSender<NodeOutput>(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueSender with type: " << (int) type << " and name: " << name)
    receivers_ = std::make_shared<std::set<CoreQueueReceiver<NodeOutput> *>>();
  }

  ~CoreQueueSender() override {
    HLOG_SELF(0, "Destructing CoreQueueSender")
  }

  std::shared_ptr<std::set<CoreQueueReceiver<NodeOutput> *>> const &receivers() const {
    return receivers_;
  }

  void addReceiver(CoreReceiver<NodeOutput> *receiver) override {
    HLOG_SELF(0, "Add receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (auto queueReceiver: receiver->receivers()) {
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//      std::cout << "sender.addReceiver : " << this->name() << " " << this->id() << "(" << this << ")" << " -> " << receiver->name() << " "  << receiver->id() << std::endl;
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      auto r = dynamic_cast<CoreQueueReceiver<NodeOutput> *>(queueReceiver);
      assert(r != nullptr);
      this->receivers_->insert(r);
    }
  }

  void removeReceiver(CoreReceiver<NodeOutput> *receiver) final {
    HLOG_SELF(0, "Remove receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (auto queueReceiver: receiver->receivers()) {
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//      std::cout << "sender.removeReceiver : " << this->name() << " " << this->id() << "(" << this << ")" << " -/> " << receiver->name() << " "  << receiver->id() << std::endl;
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      auto r = dynamic_cast<CoreQueueReceiver<NodeOutput> *>(queueReceiver);
      assert(r != nullptr);
      this->receivers_->erase(r);
    }
  }

  void sendAndNotify(std::shared_ptr<NodeOutput> ptr) final {
    for (CoreQueueReceiver<NodeOutput> *receiver : *(this->receivers_)) {
      receiver->queueSlot()->lockUniqueMutex();

      HLOG_SELF(2, "Send data to " << receiver->name() << "(" << receiver->id() << ")")
      receiver->receive(ptr);
      receiver->queueSlot()->unlockUniqueMutex();

      HLOG_SELF(2, "Wake up " << receiver->name() << "(" << receiver->id() << ")")
      receiver->queueSlot()->wakeUp();
    }
  }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    for (CoreReceiver<NodeOutput> *receiver : *(this->receivers())) {
      if (receiver->type() != NodeType::Switch || receiver->type() != NodeType::ExecutionPipeline) {
        printer->printEdge(this,
                           receiver,
                           HedgehogTraits::type_name<NodeOutput>(),
                           receiver->queueSize(),
                           receiver->maxQueueSize(),
                           HedgehogTraits::is_managed_memory_v<NodeOutput>);
      }
    }
  }

  std::set<CoreSender<NodeOutput> *> getSenders() override {
    return {this};
  }

  void copyInnerStructure(CoreQueueSender<NodeOutput> *rhs) {
    HLOG_SELF(0, "Copy Cluster CoreQueueSender information from " << rhs->name() << "(" << rhs->id() << ")")
    this->receivers_ = rhs->receivers_;
    CoreQueueNotifier::copyInnerStructure(rhs);
  }

 protected:
  void duplicateEdge(CoreNode *duplicateNode,
                     std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
//    /////////////////////////////////////////////////////////////////////////////////////////////////////////
//     std::cout << std::endl;
//     std::cout
//      << "Duplicating EP edge from the node "
//      << this->name() << " / " << this->id()
//      << " to clone: " << duplicateNode->name() << " / " << duplicateNode->id()
//      << std::endl;
//     /////////////////////////////////////////////////////////////////////////////////////////////////////////

    for (CoreQueueReceiver<NodeOutput> *originalReceiver : *(this->receivers())) {
      ///////////////////////////////////////////////////////////////////////////////////////////////////////
//      std::cout << "\tReceiver: " << originalReceiver->name() << " / " << originalReceiver->id();
//      auto coreNodeReceiver = static_cast<CoreNode*>(originalReceiver);
//      auto coreDuplica = correspondenceMap.find(coreNodeReceiver);
//      auto correspondanceFound = coreDuplica->second.get();
//      std::cout << " correspondance found: " << correspondanceFound->name() << " / " << correspondanceFound->id() << " do the linkage!"<< std::endl;
//      std::cout << "Edge found: " << this->id() << " -> " << coreNodeReceiver->id() << std::endl;
      ///////////////////////////////////////////////////////////////////////////////////////////////////////
      auto nodeReceiverFound = correspondenceMap.find(static_cast<CoreNode *>(originalReceiver));
      if (nodeReceiverFound != correspondenceMap.end()) {
        if (nodeReceiverFound->second->id() == this->id()) {
          std::cerr << "Receiver found is the same as the original receiver" << std::endl;
          exit(42);
        }
        connectSenderToReceiverDuplication(duplicateNode, nodeReceiverFound->second.get());
      } else {
        std::cerr << "Receiver id " << originalReceiver->id() << " not found in map!" << std::endl;
        exit(42);
      }

    }
  }

 private:
  void connectSenderToReceiver(CoreNode *coreNodeSender, CoreNode *coreNodeReceiver) {
    auto coreReceiver = dynamic_cast<CoreReceiver<NodeOutput> *>(coreNodeReceiver);
    auto coreSlot = dynamic_cast<CoreSlot *>(coreNodeReceiver);
    auto coreNotifier = dynamic_cast<CoreNotifier *>(coreNodeSender);
    for (auto r : coreReceiver->receivers()) {
      dynamic_cast<CoreQueueSender<NodeOutput> *>(coreNodeSender)->addReceiver(r);
    }
    for (CoreSlot *slot : coreSlot->getSlots()) { coreNotifier->addSlot(slot); }
  }

  void connectReceiverToSender(CoreNode *coreNodeSender, CoreNode *coreNodeReceiver) {
    auto coreReceiver = dynamic_cast<CoreReceiver<NodeOutput> *>(coreNodeReceiver);
    auto coreSlot = dynamic_cast<CoreSlot *>(coreNodeReceiver);
    for (auto s : dynamic_cast<CoreQueueSender<NodeOutput> *>(coreNodeSender)->getSenders()) {
      coreReceiver->addSender(s);
      coreSlot->addNotifier(s);
    }
  }

  void connectSenderToReceiverDuplication(CoreNode *coreNodeSender, CoreNode *coreNodeReceiver) {
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
//     std::cout << "\tCreating edge " << coreNodeSender->id() << " -> " << coreNodeReceiver->id() << std::endl;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

//     connectSenderToReceiver(coreNodeSender, coreNodeReceiver);
//     connectReceiverToSender(coreNodeSender, coreNodeReceiver);
    switch (coreNodeSender->type()) {
      case NodeType::Graph:
      case NodeType::Task:
      case NodeType::StateManager:
      case NodeType::Switch:
        switch (coreNodeReceiver->type()) {
          case NodeType::Graph:
          case NodeType::Task:
          case NodeType::StateManager:
          case NodeType::ExecutionPipeline:connectSenderToReceiver(coreNodeSender, coreNodeReceiver);
          default:break;
        }
      default:break;
    }

    switch (coreNodeSender->type()) {
      case NodeType::Graph:
      case NodeType::Task:
      case NodeType::StateManager:
      case NodeType::ExecutionPipeline:
        switch (coreNodeReceiver->type()) {
          case NodeType::Graph:
          case NodeType::Task:
          case NodeType::StateManager:
          case NodeType::Switch:connectReceiverToSender(coreNodeSender, coreNodeReceiver);
          default:break;
        }
      default:break;
    }
  }
};

#endif //HEDGEHOG_CORE_QUEUE_SENDER_H
