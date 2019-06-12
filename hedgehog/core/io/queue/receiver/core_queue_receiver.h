//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_QUEUE_RECEIVER_H
#define HEDGEHOG_CORE_QUEUE_RECEIVER_H

#include "../../base/receiver/core_receiver.h"
#include "core_queue_slot.h"

template<class CoreInput>
class CoreQueueReceiver : public virtual CoreReceiver<CoreInput> {
 private:
  std::shared_ptr<std::queue<std::shared_ptr<CoreInput>>> queue_ = nullptr;
  std::shared_ptr<std::set<CoreSender<CoreInput> *>> senders_ = nullptr;
  size_t maxQueueSize_ = 0;

 public:
  CoreQueueReceiver(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreReceiver<
      CoreInput>(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueReceiver with type: " << (int) type << " and name: " << name)
    queue_ = std::make_shared<std::queue<std::shared_ptr<CoreInput>>>();
    senders_ = std::make_shared<std::set<CoreSender<CoreInput> *>>();
  }

  ~CoreQueueReceiver() override {
    HLOG_SELF(0, "Destructing CoreQueueReceiver")
  }

  virtual CoreQueueSlot *queueSlot() = 0;

  std::shared_ptr<std::set<CoreSender<CoreInput> *>> const &senders() const {
    return senders_;
  }

  //Virtual
  size_t queueSize() override { return this->queue_->size(); }

  size_t maxQueueSize() override { return this->maxQueueSize_; }

  void addSender(CoreSender<CoreInput> *sender) final {
    HLOG_SELF(0, "Adding sender " << sender->name() << "(" << sender->id() << ")")
    this->senders_->insert(sender);
  }

  void removeSender(CoreSender<CoreInput> *sender) final {
    HLOG_SELF(0, "Remove sender " << sender->name() << "(" << sender->id() << ")")
    this->senders_->erase(sender);
  }

  void receive(std::shared_ptr<CoreInput> data) final {
    this->queue_->push(data);
    HLOG_SELF(2, "Receives data new queue Size " << this->queueSize())
    if (this->queueSize() > this->maxQueueSize_) { this->maxQueueSize_ = this->queueSize(); }
  }

  bool receiverEmpty() final {
    HLOG_SELF(2, "Test queue emptiness")
    return this->queue_->empty();
  }

  std::set<CoreReceiver<CoreInput> *> receivers() override {
    return {this};
  }

  std::shared_ptr<CoreInput> popFront() {
    HLOG_SELF(2, "Pop & front from queue")
    assert(!queue_->empty());
    auto element = queue_->front();
    assert(element != nullptr);
    queue_->pop();
    return element;
  }

  void copyInnerStructure(CoreQueueReceiver<CoreInput> *rhs) {
    HLOG_SELF(0, "Duplicate CoreQueueReceiver information from " << rhs->name() << "(" << rhs->id() << ")")
    this->queue_ = rhs->queue_;
    this->senders_ = rhs->senders_;
  }

};
#endif //HEDGEHOG_CORE_QUEUE_RECEIVER_H
