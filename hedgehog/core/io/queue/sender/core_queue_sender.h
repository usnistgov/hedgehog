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
class CoreQueueSender : public CoreSender<NodeOutput>, public CoreQueueNotifier {
  std::shared_ptr<std::set<CoreQueueReceiver<NodeOutput> *>> receivers_ = nullptr;

 public:
  CoreQueueSender(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreSender<NodeOutput>(name, type, numberThreads),
        CoreQueueNotifier(name, type, numberThreads) {
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
      auto r = dynamic_cast<CoreQueueReceiver<NodeOutput> *>(queueReceiver);
      assert(r != nullptr);
      this->receivers_->insert(r);
    }
  }

  void removeReceiver(CoreReceiver<NodeOutput> *receiver) final {
    HLOG_SELF(0, "Remove receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (auto queueReceiver: receiver->receivers()) {
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
                           receiver->maxQueueSize());
      }
    }
  }

  std::set<CoreSender<NodeOutput> *> getSenders() override {
    return {this};
  }

  void copyInnerStructure(CoreQueueSender<NodeOutput> *rhs) {
    HLOG_SELF(0, "Duplicate CoreQueueSender information from " << rhs->name() << "(" << rhs->id() << ")")
    this->receivers_ = rhs->receivers_;
    CoreQueueNotifier::copyInnerStructure(rhs);
  }
};

#endif //HEDGEHOG_CORE_QUEUE_SENDER_H
