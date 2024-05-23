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

#ifndef HEDGEHOG_IMPLEMENTOR_NOTIFIER_H
#define HEDGEHOG_IMPLEMENTOR_NOTIFIER_H

#include <memory>
#include <set>

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

/// @brief Hedgehog abstraction namespace
namespace abstraction {
class SlotAbstraction;
}
/// @brief Hedgehog implementor namespace
namespace implementor {
/// @brief Implementor for the NotifierAbstraction
class ImplementorNotifier {
 protected:
  std::unique_ptr<std::set<abstraction::NotifierAbstraction *>>
      abstractNotifiers_ = nullptr; ///< Set of linked NotifierAbstraction

 public:
  /// @brief Default constructor
  explicit ImplementorNotifier()
      : abstractNotifiers_(std::make_unique<std::set<abstraction::NotifierAbstraction *>>()) {}

  /// @brief Default destructor
  virtual ~ImplementorNotifier() = default;

  /// @brief Accessor to the linked NotifierAbstraction
  /// @return Set of NotifierAbstraction
  [[nodiscard]] std::set<abstraction::NotifierAbstraction *> &notifiers() { return *abstractNotifiers_; }

  /// @brief Initialize the implementor Notifier by setting the corresponding abstraction
  /// @param notifierAbstraction Notifier abstraction to set
  virtual void initialize(abstraction::NotifierAbstraction *notifierAbstraction) {
    abstractNotifiers_->insert(notifierAbstraction);
  }

  /// @brief  Accessor to the connected slots
  /// @return Set of connected slots
  [[nodiscard]] virtual std::set<abstraction::SlotAbstraction *> const &connectedSlots() const = 0;

  /// @brief Add a slot to the connected slots
  /// @param slot Slot to add
  virtual void addSlot(abstraction::SlotAbstraction *slot) = 0;

  /// @brief Remove a slot to the connected slots
  /// @param slot Slot to remove
  virtual void removeSlot(abstraction::SlotAbstraction *slot) = 0;

  /// @brief Notify the connected slots to wake up
  virtual void notify() = 0;

  /// @brief Notify the connected slots that this notifier is terminated
  virtual void notifyAllTerminated() = 0;
};
}
}
}

#endif //HEDGEHOG_IMPLEMENTOR_NOTIFIER_H
