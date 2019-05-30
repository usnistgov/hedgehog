//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_TASK_SENDER_H
#define HEDGEHOG_CORE_TASK_SENDER_H

#include "../../base/sender/core_sender.h"
#include "core_task_notifier.h"

template<class TaskOutput>
class CoreTaskSender : public CoreSender<TaskOutput>, public CoreTaskNotifier {
  std::shared_ptr<std::set<CoreTaskReceiver<TaskOutput> *>> receivers_ = nullptr;

 public:
  CoreTaskSender(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreSender<TaskOutput>(name, type, numberThreads),
        CoreTaskNotifier(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreTaskSender with type: " << (int) type << " and name: " << name)
    receivers_ = std::make_shared<std::set<CoreTaskReceiver<TaskOutput> *>>();
  }

  ~CoreTaskSender() override {
    HLOG_SELF(0, "Destructing CoreTaskSender")
  }

  std::shared_ptr<std::set<CoreTaskReceiver<TaskOutput> *>> const &receivers() const {
    return receivers_;
  }

  void addReceiver(CoreReceiver<TaskOutput> *receiver) final {
    HLOG_SELF(0, "Add receiver " << receiver->name() << "(" << receiver->id() << ")")
    auto r = dynamic_cast<CoreTaskReceiver<TaskOutput> *>(receiver);
    assert(r != nullptr);
    this->receivers_->insert(r);
  }

  void removeReceiver(CoreReceiver<TaskOutput> *receiver) final {
    HLOG_SELF(0, "Remove receiver " << receiver->name() << "(" << receiver->id() << ")")
    auto r = dynamic_cast<CoreTaskReceiver<TaskOutput> *>(receiver);
    assert(r != nullptr);
    this->receivers_->erase(r);
  }

  void sendAndNotify(std::shared_ptr<TaskOutput> ptr) final {
    for (CoreTaskReceiver<TaskOutput> *receiver : *(this->receivers_)) {
      receiver->getTaskSlot()->lockUniqueMutex();

      HLOG_SELF(2, "Send data to " << receiver->name() << "(" << receiver->id() << ")")
      receiver->receive(ptr);
      receiver->getTaskSlot()->unlockUniqueMutex();

      HLOG_SELF(2, "Wake up " << receiver->name() << "(" << receiver->id() << ")")
      receiver->getTaskSlot()->wakeUp();
    }
  }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    for (CoreReceiver<TaskOutput> *receiver : *(this->receivers())) {
      printer->printEdge(this,
                         receiver,
                         HedgehogTraits::type_name<TaskOutput>(),
                         receiver->queueSize(),
                         receiver->maxQueueSize());
    }
  }

  std::set<CoreSender<TaskOutput> *> getSenders() override {
    return {this};
  }

  void copyInnerStructure(CoreTaskSender<TaskOutput> *rhs) {
    HLOG_SELF(0, "Duplicate CoreTaskSender information from " << rhs->name() << "(" << rhs->id() << ")")
    this->receivers_ = rhs->receivers_;
    CoreTaskNotifier::copyInnerStructure(rhs);
  }
};

#endif //HEDGEHOG_CORE_TASK_SENDER_H
