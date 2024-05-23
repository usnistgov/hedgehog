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

#ifndef HEDGEHOG_NOTIFIER_ABSTRACTION_H
#define HEDGEHOG_NOTIFIER_ABSTRACTION_H
#pragma once

#include <utility>
#include <ostream>

#include "slot_abstraction.h"
#include "../../../implementors/implementor/implementor_notifier.h"
#include "../clonable_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog implementor namespace
namespace implementor {
/// @brief Forward declaration Notifier implementor
class ImplementorNotifier;
}

#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Core abstraction to notify slots
class NotifierAbstraction {
 private:
  std::shared_ptr<implementor::ImplementorNotifier>
      concreteNotifier_ = nullptr; ///< Concrete implementation of the notifier used in the node
 public:
  /// @brief Constructor utilising a concrete implementation
  /// @param notifier Concrete notifier implementation
  explicit NotifierAbstraction(std::shared_ptr<implementor::ImplementorNotifier> notifier)
      : concreteNotifier_(std::move(notifier)) {
    concreteNotifier_->initialize(this);
  }

  /// @brief Default destructor
  virtual ~NotifierAbstraction() = default;

  /// Const accessor to notifiers
  /// @brief Present the notifiers linked to this abstraction, usually one, may be multiple for the graph presenting all
  /// of the input node notifiers
  /// @return Const reference to notifiers
  [[nodiscard]] std::set<NotifierAbstraction *> const &notifiers() const { return concreteNotifier_->notifiers(); }

  /// Accessor to notifiers
  /// @brief Present the notifiers linked to this abstraction, usually one, may be multiple for the graph presenting all
  /// of the input node notifiers
  /// @return Reference to notifiers
  [[nodiscard]] std::set<NotifierAbstraction *> &notifiers() { return concreteNotifier_->notifiers(); }

  /// @brief Add a SlotAbstraction to this notifier
  /// @param slot SlotAbstraction to add
  void addSlot(SlotAbstraction *const slot) { concreteNotifier_->addSlot(slot); }

  /// @brief Remove SlotAbstraction from this notifier
  /// @param slot SlotAbstraction to remove
  void removeSlot(SlotAbstraction *const slot) { concreteNotifier_->removeSlot(slot); }

  /// @brief Accessor to the slots attached to this notifier
  /// @return The SlotAbstraction attached to this notifier
  [[nodiscard]] std::set<SlotAbstraction *> const &connectedSlots() const { return concreteNotifier_->connectedSlots(); }

  /// @brief Notify a slot to wake up
  void notify() { concreteNotifier_->notify(); }

  /// @brief Notifier all slots that this node is terminated
  void notifyAllTerminated() { concreteNotifier_->notifyAllTerminated(); }

 protected:
  /// @brief Duplicate edges of the current notifier to slots to clone in map
  /// @param mapping Map of the nodes -> clone
  /// @throw throw std::runtime_error if the current node is not mapped to its clone, if the clone is not a
  /// SlotAbstraction, if a slot is not a node
  void duplicateEdgeNotifier(
      std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping) {
    std::shared_ptr<NodeAbstraction> duplicateSlot;
    auto notifierAsNode = dynamic_cast<abstraction::NodeAbstraction *>(this);
    if (!mapping.contains(notifierAsNode)) {
      throw std::runtime_error("A node that we are trying to connect is not mapped yet.");
    }
    auto mappedNotifier = std::dynamic_pointer_cast<abstraction::NotifierAbstraction>(mapping.at(notifierAsNode));
    if (mappedNotifier == nullptr) {
      throw std::runtime_error("The mapped type of a node is not of the right type: abstraction::NodeAbstraction.");
    }

    for (auto &notifier : this->notifiers()) {
      for (auto &slot : notifier->connectedSlots()) {
        for (auto &s : slot->slots()) {
          if (auto slotAsNode = dynamic_cast<abstraction::NodeAbstraction *>(s)) {
            if (mapping.contains(slotAsNode)) {
              auto mappedSlot = std::dynamic_pointer_cast<abstraction::SlotAbstraction>(mapping.at(slotAsNode));
              if (mappedSlot == nullptr) {
                throw std::runtime_error("The mapped type of a node is not of the right type: SlotAbstraction.");
              }

              for (auto mmapedS : mappedSlot->slots()) {
                for (auto mmapedN : mappedNotifier->notifiers()) {
                  mmapedN->addSlot(mmapedS);
                  mmapedS->addNotifier(mmapedN);
                }
              }

            }
          } else {
            throw std::runtime_error("A slot is not a node when duplicating edges.");
          }
        }
      }
    }
  }
};
}
}
}

#endif //HEDGEHOG_NOTIFIER_ABSTRACTION_H
