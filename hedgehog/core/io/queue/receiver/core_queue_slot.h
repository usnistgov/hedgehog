//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_QUEUE_SLOT_H
#define HEDGEHOG_CORE_QUEUE_SLOT_H
#include <mutex>
#include <condition_variable>
#include <memory>
#include <set>
#include <cassert>

#include "../../base/receiver/core_slot.h"

class CoreQueueSlot : public virtual CoreSlot {
 private:
  std::shared_ptr<std::mutex> slotMutex_ = nullptr;
  std::shared_ptr<std::condition_variable> notifyConditionVariable_ = nullptr;
  std::shared_ptr<std::set<CoreNotifier *>> notifiers_ = nullptr;

 public:
  CoreQueueSlot(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreSlot(name,
                                                                                                          type,
                                                                                                          numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueSlot with type: " << (int) type << " and name: " << name)
    notifiers_ = std::make_shared<std::set<CoreNotifier *>>();
    slotMutex_ = std::make_shared<std::mutex>();
    notifyConditionVariable_ = std::make_shared<std::condition_variable>();
  }

  ~CoreQueueSlot() override {HLOG_SELF(0, "Destructing CoreQueueSlot")}

  size_t numberInputNodes() const final { return this->notifiers()->size(); }

  void addNotifier(CoreNotifier *notifier) final {
    std::lock_guard<std::mutex> lc(*(this->slotMutex_));
    this->notifiers()->insert(notifier);
  }
  void removeNotifier(CoreNotifier *notifier) final {
    std::lock_guard<std::mutex> lc(*(this->slotMutex_));
    this->notifiers()->erase(notifier);
  }
  bool hasNotifierConnected() final {
    HLOG_SELF(2,
              "Test has notifier connected " << "(" << std::boolalpha << (bool) (this->numberInputNodes() != 0) << ")")
    return this->numberInputNodes() != 0;
  }

  void wakeUp() final {
    HLOG_SELF(2, "Wake up and notify one")
    this->notifyConditionVariable()->notify_one();
  }

  std::shared_ptr<std::condition_variable> const &notifyConditionVariable() const { return notifyConditionVariable_; }

  void lockUniqueMutex() {
    HLOG_SELF(2, "Lock unique mutex " << this->slotMutex_.get())
    slotMutex_->lock();
  }
  void unlockUniqueMutex() {
    HLOG_SELF(2, "Unlock unique mutex " << this->slotMutex_.get())
    slotMutex_->unlock();
  }

  std::shared_ptr<std::mutex> const &slotMutex() const {
    return slotMutex_;
  }

  void copyInnerStructure(CoreQueueSlot *rhs) {
    HLOG_SELF(0, "Duplicate CoreQueueSlot information from " << rhs->name() << "(" << rhs->id() << ")")
    this->slotMutex_ = rhs->slotMutex_;
    this->notifyConditionVariable_ = rhs->notifyConditionVariable_;
    this->notifiers_ = rhs->notifiers_;
  }

 private:
  std::shared_ptr<std::set<CoreNotifier *>> const &notifiers() const { return notifiers_; }

};

#endif //HEDGEHOG_CORE_QUEUE_SLOT_H
