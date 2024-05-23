//  NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
//  software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
//  derivative works of the software or any portion of the software, and you may copy and distribute such modifications
//  or works. Modified works should carry a notice stating that you changed the software and should note the date and
//  nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
//  source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
//  EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
//  WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
//  CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
//  THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
//  are solely responsible for determining the appropriateness of using and distributing the software and you assume
//  all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
//  with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
//  operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
//  damage to property. The software developed by NIST employees is not subject to copyright protection within the
//  United States.

#ifndef HEDGEHOG_DEFAULT_SLOT_H
#define HEDGEHOG_DEFAULT_SLOT_H

#include <set>
#include <mutex>

#include "../../implementor/implementor_slot.h"
#include "../../implementor/implementor_notifier.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of slot interface
/// @details Utilise mutexes to protect its list of notifiers, and for the condition_variable / wait mechanism
class DefaultSlot : public ImplementorSlot {
 private:
  std::unique_ptr<std::set<abstraction::NotifierAbstraction *>> const notifiers_ =
      std::make_unique<std::set<abstraction::NotifierAbstraction * >>(); ///< List of notifiers linked to this slot

  std::mutex
      mutexNotifierAccess_{}, ///< Mutex to protect the list of notifiers
      mutexSleep_{}; ///< Mutex used for the condition_variable / wait mechanism
  std::condition_variable conditionVariable_{}; ///< Condition variable to make the node wait



 public:
  /// @brief Test if there is any notifiers connected
  /// @return True if there is any, else false
  [[nodiscard]] bool hasNotifierConnected() override {
    std::lock_guard<std::mutex> lck(mutexNotifierAccess_);
    return !notifiers_->empty();
  }

  /// @brief Accessor to the number of notifiers connected
  /// @return Number of notifiers connected
  size_t nbNotifierConnected() override {
    std::lock_guard<std::mutex> lck(mutexNotifierAccess_);
    return notifiers_->size();
  }

  /// @brief Accessor to the connected notifiers
  /// @return A set of connected notifiers
  [[nodiscard]] std::set<abstraction::NotifierAbstraction *> const &connectedNotifiers() const override {
    return *notifiers_;
  }

  /// @brief Add a notifier to the list of connected notifiers
  /// @param notifier Notifier to add to the list of connected notifiers
  void addNotifier(abstraction::NotifierAbstraction *const notifier) override {
    std::lock_guard<std::mutex> lck(mutexNotifierAccess_);
    notifiers_->insert(notifier);
  }

  /// @brief Remove a notifier to the list of connected notifiers
  /// @param notifier Notifier to remove from the list of connected notifiers
  void removeNotifier(abstraction::NotifierAbstraction *const notifier) override {
    std::lock_guard<std::mutex> lck(mutexNotifierAccess_);
    notifiers_->erase(notifier);
  }

  /// @brief Sleep mechanism used to make the thread enter in a sleep state
  /// @param slot Slot abstraction (core attache to the thread), used for callbacks
  /// @return True if the node can terminate, else false
  bool sleep(abstraction::SlotAbstraction *slot) override {
    std::unique_lock<std::mutex> lock(mutexSleep_);
    conditionVariable_.wait(lock, [&slot]() { return slot->waitTerminationCondition(); });
    return slot->canTerminate();
  }

  /// @brief Function used to wake up a thread attached to this condition variable
  void wakeUp() override {
    // This lock is important to avoid that when checking for waitTerminationCondition in sleep function, the queue may
    // start by being empty, the test store the queue emptiness, some data come in and the slot is notified. Then
    // waitTerminationCondition returns with a wrong value. the thread sleeps without being awakened.
    std::lock_guard<std::mutex> lck(mutexSleep_);
    conditionVariable_.notify_one();
  }

};
}
}
}

#endif //HEDGEHOG_DEFAULT_SLOT_H
