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

#ifndef HEDGEHOG_DEFAULT_NOTIFIER_H
#define HEDGEHOG_DEFAULT_NOTIFIER_H

#include <mutex>

#include "../implementor/implementor_notifier.h"
#include "../implementor/implementor_slot.h"
#include "../../abstractions/base/input_output/slot_abstraction.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of notifier interface
class DefaultNotifier : public ImplementorNotifier {
 private:
  std::unique_ptr<std::set<abstraction::SlotAbstraction *>> const
      slots_ = nullptr; ///< Slot getting the message
  std::mutex mutex_{}; ///< Mutex used to protect the list of connected slots

 public:
  /// @brief Default constructor
  explicit DefaultNotifier() : slots_(std::make_unique<std::set<abstraction::SlotAbstraction *>>()){}

  /// @brief Default destructor
  ~DefaultNotifier() override = default;

  /// @brief Add a slot to transmit messages to
  /// @param slot Slot to add
  void addSlot(abstraction::SlotAbstraction *const slot) override {
    std::lock_guard<std::mutex> lck(mutex_);
    slots_->insert(slot);
  }

  /// @brief Remove a slot to transmit messages to
  /// @param slot Slot to remove
  void removeSlot(abstraction::SlotAbstraction *const slot) override {
    std::lock_guard<std::mutex> lck(mutex_);
    slots_->erase(slot);
  }

  /// @brief Accessor to the connected slots
  /// @return Connected slots
  [[nodiscard]] std::set<abstraction::SlotAbstraction *> const &connectedSlots() const override { return *slots_; }

  /// @brief Notify method, calls wakeUp on all connected slots
  void notify() override {
    std::lock_guard<std::mutex> lck(mutex_);
    for (const auto &slot : *slots_) { slot->wakeUp(); }
  }

  /// @brief Remove the notifier connection from all connected slots, and calls wakeUp on all
  void notifyAllTerminated() override {
    std::lock_guard<std::mutex> lck(mutex_);
    for (auto notifier : *(this->abstractNotifiers_)) {
      for (abstraction::SlotAbstraction *slot : *slots_) { slot->removeNotifier(notifier); }
    }
    for (abstraction::SlotAbstraction *slot : *slots_) { slot->wakeUp(); }
    while (!slots_->empty()) { slots_->erase(slots_->begin()); }
  }

};
}
}
}

#endif //HEDGEHOG_DEFAULT_NOTIFIER_H
