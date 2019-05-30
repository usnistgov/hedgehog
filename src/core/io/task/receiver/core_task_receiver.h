//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_TASK_RECEIVER_H
#define HEDGEHOG_CORE_TASK_RECEIVER_H

#include "../../base/receiver/core_receiver.h"

template<class TaskInput>
class CoreTaskReceiver : public virtual CoreReceiver<TaskInput> {
 private:
  std::shared_ptr<std::queue<std::shared_ptr<TaskInput>>> queue_ = nullptr;
  std::shared_ptr<std::set<CoreSender<TaskInput> *>> senders_ = nullptr;
  size_t maxQueueSize_ = 0;

 public:
  CoreTaskReceiver(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreReceiver<
      TaskInput>(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreTaskReceiver with type: " << (int) type << " and name: " << name)
    queue_ = std::make_shared<std::queue<std::shared_ptr<TaskInput>>>();
    senders_ = std::make_shared<std::set<CoreSender<TaskInput> *>>();
  }

  ~CoreTaskReceiver() override {
    HLOG_SELF(0, "Destructing CoreTaskReceiver")
  }

  virtual CoreTaskSlot *getTaskSlot() = 0;

  //Virtual
  size_t queueSize() override { return this->queue_->size(); }

  size_t maxQueueSize() override { return this->maxQueueSize_; }

  void addSender(CoreSender<TaskInput> *sender) final {
    HLOG_SELF(0, "Adding sender " << sender->name() << "(" << sender->id() << ")")
    this->senders_->insert(sender);
  }

  void removeSender(CoreSender<TaskInput> *sender) final {
    HLOG_SELF(0, "Remove sender " << sender->name() << "(" << sender->id() << ")")
    this->senders_->erase(sender);
  }

  void receive(std::shared_ptr<TaskInput> data) final {
    this->queue_->push(data);
    HLOG_SELF(2, "Receives data new queue Size " << this->queueSize())
    if (this->queueSize() > this->maxQueueSize_) { this->maxQueueSize_ = this->queueSize(); }
  }

  bool receiverEmpty() final {
    HLOG_SELF(2, "Test queue emptiness")
    return this->queue_->empty();
  }

  std::set<CoreReceiver<TaskInput> *> getReceivers() override {
    return {this};
  }

  std::shared_ptr<TaskInput> popFront() {
    HLOG_SELF(2, "Pop & front from queue")
    assert(!queue_->empty());
    auto element = queue_->front();
    assert(element != nullptr);
    queue_->pop();
    return element;
  }

  void copyInnerStructure(CoreTaskReceiver<TaskInput> *rhs) {
    HLOG_SELF(0, "Duplicate CoreTaskReceiver information from " << rhs->name() << "(" << rhs->id() << ")")
    this->queue_ = rhs->queue_;
    this->senders_ = rhs->senders_;
  }

};
#endif //HEDGEHOG_CORE_TASK_RECEIVER_H
