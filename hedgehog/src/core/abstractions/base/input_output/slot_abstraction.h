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

#ifndef HEDGEHOG_SLOT_ABSTRACTION_H
#define HEDGEHOG_SLOT_ABSTRACTION_H
#pragma once

#include <set>
#include <memory>
#include <utility>
#include <iostream>
#include <condition_variable>

#include "../../../implementors/implementor/implementor_slot.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog implementor namespace
namespace implementor {
/// @brief Forward declaration ImplementorSlot
class ImplementorSlot;
}

#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog abstraction namespace
namespace abstraction {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration NotifierAbstraction
class NotifierAbstraction;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Core's abstraction to receive a signal
class SlotAbstraction {
 private:
  std::shared_ptr<implementor::ImplementorSlot>
      concreteSlot_ = nullptr; ///< Concrete implementation of the slot

 public:
  /// @brief Constructor using a concrete slot implementation
  /// @param concreteSlot Concrete slot implementation
  explicit SlotAbstraction(std::shared_ptr<implementor::ImplementorSlot> concreteSlot)
      : concreteSlot_(std::move(concreteSlot)) { concreteSlot_->initialize(this); }

  /// @brief Default destructor
  virtual ~SlotAbstraction() = default;

  /// Const accessor to slots
  /// @brief Present the slots linked to this abstraction, usually one, may be multiple for the graph presenting all
  /// of the input node slots
  /// @return Const reference to slots
  [[nodiscard]] std::set<SlotAbstraction *> const &slots() const { return concreteSlot_->slots(); }

  /// Accessor to slots
  /// @brief Present the slots linked to this abstraction, usually one, may be multiple for the graph presenting all
  /// of the input node slots
  /// @return Reference to slots
  [[nodiscard]] std::set<SlotAbstraction *> &slots() { return concreteSlot_->slots(); }

  /// @brief Accessor to the NotifierAbstraction attached to this slot, protected with mutex
  /// @return The NotifierAbstraction attached to this slot
  [[nodiscard]] std::set<NotifierAbstraction *> const &connectedNotifiers() const {
    return concreteSlot_->connectedNotifiers();
  }

  /// @brief Add a NotifierAbstraction to this slot
  /// @param notifier NotifierAbstraction to add
  void addNotifier(NotifierAbstraction *const notifier) { concreteSlot_->addNotifier(notifier); }

  /// @brief Remove a NotifierAbstraction to this slot
  /// @param notifier NotifierAbstraction to add
  void removeNotifier(NotifierAbstraction *const notifier) { concreteSlot_->removeNotifier(notifier); }

  /// @brief Callback to the concrete slot wake up function
  void wakeUp() { concreteSlot_->wakeUp(); }

  /// @brief Callback to the concrete slot sleep function
  /// @return True if the node can terminate, else false
  bool sleep() { return concreteSlot_->sleep(this); }

  /// @brief Test if there is at least one notifier connected
  /// @return True if there is at least one notifier connected, else false
  [[nodiscard]] bool hasNotifierConnected() const { return concreteSlot_->hasNotifierConnected(); };

  /// @brief Callback to the concrete number of notifiers connected
  /// @return The number of notifiers connected
  size_t nbNotifierConnected() { return concreteSlot_->nbNotifierConnected(); }

  /// @brief Copy the inner structure of the slot by duplicating the concrete slot
  /// @param copyableCore Core to copy into this
  void copyInnerStructure(SlotAbstraction *copyableCore) { this->concreteSlot_ = copyableCore->concreteSlot_; }

  /// @brief Callback to the concrete wait termination condition
  /// @return True if the node can terminate, else false
  [[nodiscard]] virtual bool waitTerminationCondition() = 0;

  /// @brief Callback to the concrete can terminate condition
  /// @return True if the node can terminate, else false
  [[nodiscard]] virtual bool canTerminate() = 0;

};
}
}
}

#endif //HEDGEHOG_SLOT_ABSTRACTION_H
