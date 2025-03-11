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

#ifndef HEDGEHOG_ATOMIC_SLOT_H
#define HEDGEHOG_ATOMIC_SLOT_H

#include <set>
#include <atomic>
#include <execution>
#include "../../../../tools/intrinsics.h"
#include "../../implementor/implementor_slot.h"
#include "../../implementor/implementor_notifier.h"
#include "../../../../../constants.h"
#include "../../../abstractions/base/input_output/notifier_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Concrete implementation of slot interface using atomics
/// @details Utilise atomic flags to protect its list of notifiers, and for the wait mechanism
class AtomicSlot : public ImplementorSlot {
  std::unique_ptr<std::set<abstraction::NotifierAbstraction *>> const notifiers_ =
      std::make_unique<std::set<abstraction::NotifierAbstraction *>>(); ///< List of notifiers linked to this slot

  alignas(CACHE_LINE_SIZE) std::atomic<bool> notifierFlag_{false}; ///< Flag used to protect the list of notifiers
  alignas(CACHE_LINE_SIZE) std::atomic_flag waitFlag_{}; ///< Flag used to put to sleep/wake up current thread

 public:

  /// @brief Test if there is any notifiers connected
  /// @return True if there is any, else false
  [[nodiscard]] bool hasNotifierConnected() override {
    while (notifierFlag_.exchange(true, std::memory_order_acquire)) { cross_platform_yield(); }
    auto ret = !notifiers_->empty();
    notifierFlag_.store(false, std::memory_order_release);
    return ret;
  }

  /// @brief Accessor to the number of notifiers connected
  /// @return Number of notifiers connected
  size_t nbNotifierConnected() override {
    while (notifierFlag_.exchange(true, std::memory_order_acquire)) { cross_platform_yield(); }
    auto ret = notifiers_->size();
    notifierFlag_.store(false, std::memory_order_release);
    return ret;
  }

  /// @brief Accessor to the connected notifiers
  /// @return A set of connected notifiers
  [[nodiscard]] std::set<abstraction::NotifierAbstraction *> const &connectedNotifiers() const override { return *notifiers_; }

  /// @brief Add a notifier to the list of connected notifiers
  /// @param notifier Notifier to add to the list of connected notifiers
  void addNotifier(abstraction::NotifierAbstraction *notifier) override {
    while (notifierFlag_.exchange(true, std::memory_order_acquire)) { cross_platform_yield(); }
    notifiers_->insert(notifier);
    notifierFlag_.store(false, std::memory_order_release);
  }

  /// @brief Remove a notifier to the list of connected notifiers
  /// @param notifier Notifier to remove from the list of connected notifiers
  void removeNotifier(abstraction::NotifierAbstraction *notifier) override {
    while (notifierFlag_.exchange(true, std::memory_order_acquire)) { cross_platform_yield(); }
    notifiers_->erase(notifier);
    notifierFlag_.store(false, std::memory_order_release);
  }

  /// @brief Sleep mechanism used to make the thread enter in a sleep state
  /// @param slot Slot abstraction (core attache to the thread), used for callbacks
  /// @return True if the node can terminate, else false
  bool sleep(abstraction::SlotAbstraction *slot) override {
    while (!slot->waitTerminationCondition()) {
      waitFlag_.wait(false);
      waitFlag_.clear();
    }
    return slot->canTerminate();
  }

  /// @brief Function used to wake up a thread attached to the atomic flag
  void wakeUp() override {
    waitFlag_.test_and_set();
    waitFlag_.notify_one();
  }

};
}
}
}

#endif //HEDGEHOG_ATOMIC_SLOT_H
