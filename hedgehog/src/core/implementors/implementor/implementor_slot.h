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



#ifndef HEDGEHOG_IMPLEMENTOR_SLOT_H
#define HEDGEHOG_IMPLEMENTOR_SLOT_H

#include <memory>

#include "../../abstractions/base/node/node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Implementor for the SlotAbstraction
class ImplementorSlot {
 protected:
  std::unique_ptr<std::set<abstraction::SlotAbstraction *>>
      abstractSlots_ = nullptr; ///< Set of linked SlotAbstraction

 public:
  /// @brief Default constructor
  explicit ImplementorSlot() :
      abstractSlots_(std::make_unique<std::set<abstraction::SlotAbstraction *>>()) {}

  /// @brief Default destructor
  virtual ~ImplementorSlot() = default;

  /// @brief Accessor to the linked SlotAbstraction
  /// @return Set of linked SlotAbstraction
  [[nodiscard]] std::set<abstraction::SlotAbstraction *> &slots() { return *abstractSlots_; }

  /// @brief Initialize the implementor Slot by setting the corresponding abstraction
  /// @param slotAbstraction Slot abstraction to set
  virtual void initialize(abstraction::SlotAbstraction *slotAbstraction) {
    abstractSlots_->insert(slotAbstraction);
  }

  /// @brief Accessor to the connected notifiers
  /// @return Set of connected NotifierAbstraction
  [[nodiscard]] virtual std::set<abstraction::NotifierAbstraction *> const &connectedNotifiers() const = 0;

  /// @brief Test if there is any notifiers connected
  /// @return True if at least a notifier is connected, else false
  [[nodiscard]] virtual bool hasNotifierConnected() const = 0;

  /// @brief Add a notifier to the set of NotifierAbstraction
  /// @param notifier NotifierAbstraction to add
  virtual void addNotifier(abstraction::NotifierAbstraction *notifier) = 0;

  /// @brief Remove notifier to the set of NotifierAbstraction
  /// @param notifier NotifierAbstraction to remove
  virtual void removeNotifier(abstraction::NotifierAbstraction *notifier) = 0;
};
}
}
}
#endif //HEDGEHOG_IMPLEMENTOR_SLOT_H
