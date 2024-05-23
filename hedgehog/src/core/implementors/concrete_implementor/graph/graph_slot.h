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

#ifndef HEDGEHOG_GRAPH_SLOT_H
#define HEDGEHOG_GRAPH_SLOT_H

#include <memory>

#include "../../implementor/implementor_slot.h"
#include "../../../abstractions/base/input_output/notifier_abstraction.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of the slot abstraction for the graph core
class GraphSlot : public ImplementorSlot {
 public:

  /// @brief Default constructor
  explicit GraphSlot() = default;

  /// @brief Default destructor
  ~GraphSlot() override = default;

  /// @brief Redefine the implementor to do nothing, the graph do nothing by itself
  /// @param slotAbstraction Abstraction not used
  void initialize([[maybe_unused]]abstraction::SlotAbstraction *slotAbstraction) override {}

  /// @brief Do nothing, throw an error
  /// @return Nothing, throw an error
  /// @throw std::runtime_error A graph has no connected connectedNotifiers by itself
  [[nodiscard]] bool hasNotifierConnected() override {
    throw std::runtime_error("A graph has no connected notifiers connected");
  }

  size_t nbNotifierConnected() override { return 0; }

  bool sleep([[maybe_unused]] abstraction::SlotAbstraction *slot) override {
    throw std::runtime_error("A graph slot cannot sleep, it is not attached to a thread.");
  }

  void wakeUp() override { for (auto slot : *abstractSlots_) { slot->wakeUp(); }}

  /// @brief Add a notifier to all input nodes
  /// @param notifier Notifier top add to the input nodes
  void addNotifier(abstraction::NotifierAbstraction *notifier) override {
    for (auto slot : *abstractSlots_) { slot->addNotifier(notifier); }
  }

  /// @brief Remove a notifier to all input nodes
  /// @param notifier Notifier top remove to the input nodes
  void removeNotifier(abstraction::NotifierAbstraction *notifier) override {
    for (auto slot : *abstractSlots_) { slot->removeNotifier(notifier); }
  }

  [[nodiscard]] std::set<abstraction::NotifierAbstraction *> const &connectedNotifiers() const override {
    throw std::runtime_error("A graph has no connected notifiers connected");
  }
};
}
}
}

#endif //HEDGEHOG_GRAPH_SLOT_H
