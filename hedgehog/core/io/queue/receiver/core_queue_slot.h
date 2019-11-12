// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.


#ifndef HEDGEHOG_CORE_QUEUE_SLOT_H
#define HEDGEHOG_CORE_QUEUE_SLOT_H
#include <mutex>
#include <condition_variable>
#include <memory>
#include <set>
#include <cassert>

#include "../../base/receiver/core_slot.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Slot of CoreQueueMultiReceiver, receiving from CoreQueueNotifier
class CoreQueueSlot : public virtual CoreSlot {
 private:
  std::shared_ptr<std::mutex> slotMutex_ = nullptr; ///< Mutex locking the CoreQueueMultiReceiver
  std::shared_ptr<std::condition_variable> notifyConditionVariable_ = nullptr; ///< Condition Variable linked to the
  ///< CoreQueueSlot::slotMutex_
  std::shared_ptr<std::set<CoreNotifier *>> notifiers_ = nullptr; ///< Set of notifiers linked to this CoreQueueSlot

 public:

  /// @brief CoreQueueSlot constructor
  /// @param name Node's name
  /// @param type Node's type
  /// @param numberThreads Node's number of thread
  CoreQueueSlot(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreSlot(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueSlot with type: " << (int) type << " and name: " << name)
    notifiers_ = std::make_shared<std::set<CoreNotifier *>>();
    slotMutex_ = std::make_shared<std::mutex>();
    notifyConditionVariable_ = std::make_shared<std::condition_variable>();
  }

  /// @brief CoreQueueSlot destructor
  ~CoreQueueSlot() override {HLOG_SELF(0, "Destructing CoreQueueSlot")}

  /// @brief Condition variable accessor
  /// @return Condition variable
  [[nodiscard]] std::shared_ptr<std::condition_variable> const &notifyConditionVariable() const {
    return notifyConditionVariable_;
  }

  /// @brief Mutex accessor
  /// @return mutex
  [[nodiscard]] std::shared_ptr<std::mutex> const &slotMutex() const { return slotMutex_; }

  /// @brief Number of CoreNotifier linked accessor
  /// @attention Not thread safe
  /// @return Number of CoreNotifier linked
  [[nodiscard]] size_t numberInputNodes() const final { return this->notifiers()->size(); }

  /// @brief Add a notifier to set of CoreNotifier
  /// @attention Thread safe
  /// @param notifier CoreNotifier to add
  void addNotifier(CoreNotifier *notifier) final {
    std::lock_guard<std::mutex> lc(*(this->slotMutex_));
    this->notifiers()->insert(notifier);
  }

  /// @brief Remove a notifier from set of CoreNotifier
  /// @attention Thread safe
  /// @param notifier CoreNotifier to remove
  void removeNotifier(CoreNotifier *notifier) final {
    std::lock_guard<std::mutex> lc(*(this->slotMutex_));
    this->notifiers()->erase(notifier);
  }

  /// @brief Test if CoreNotifier are linked to this CoreQueueSlot
  /// @attention Not thread safe
  /// @return True if CoreNotifier are linked to this CoreQueueSlot, else False
  bool hasNotifierConnected() final {
    HLOG_SELF(2,
              "Test has notifier connected " << "(" << std::boolalpha << (bool) (this->numberInputNodes() != 0) << ")")
    return this->numberInputNodes() != 0;
  }

  /// @brief Wake up and notify a node connected to the condition variable CoreQueueSlot::notifyConditionVariable_
  void wakeUp() final {
    HLOG_SELF(2, "Wake up and notify one")
    this->notifyConditionVariable()->notify_one();
  }

  /// @brief Lock the mutex
  void lockUniqueMutex() {
    HLOG_SELF(2, "Lock unique mutex " << this->slotMutex_.get())
    slotMutex_->lock();
  }

  /// @brief Unlock the mutex
  void unlockUniqueMutex() {
    HLOG_SELF(2, "Unlock unique mutex " << this->slotMutex_.get())
    slotMutex_->unlock();
  }

  /// @brief Copy the inner structure of the receiver (mutex, condition variable and set of notifiers)
  /// @param rhs CoreQueueSlot to copy to this
  void copyInnerStructure(CoreQueueSlot *rhs) {
    HLOG_SELF(0, "Copy Cluster CoreQueueSlot information from " << rhs->name() << "(" << rhs->id() << ")")
    this->slotMutex_ = rhs->slotMutex_;
    this->notifyConditionVariable_ = rhs->notifyConditionVariable_;
    this->notifiers_ = rhs->notifiers_;
  }

 protected:
  /// @brief Protected accessor to the set of notifiers connected to the CoreQueueSlot
  /// @return Set of notifiers connected to the CoreQueueSlot
  [[nodiscard]] std::shared_ptr<std::set<CoreNotifier *>> const &notifiers() const { return notifiers_; }
};

}
#endif //HEDGEHOG_CORE_QUEUE_SLOT_H
