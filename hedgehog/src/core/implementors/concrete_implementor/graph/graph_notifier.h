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

#ifndef HEDGEHOG_GRAPH_NOTIFIER_H
#define HEDGEHOG_GRAPH_NOTIFIER_H

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of the notifier abstraction for the graph core
class GraphNotifier : public ImplementorNotifier {
 public:

  /// @brief Default constructor
  explicit GraphNotifier() = default;

  /// @brief Default destructor
  ~GraphNotifier() override = default;

  /// @brief Redefine the implementor to do nothing, the graph do nothing by itself
  /// @param notifierAbstraction Abstraction not used
  void initialize([[maybe_unused]] abstraction::NotifierAbstraction *notifierAbstraction) override {}

  /// @brief Add slot to the graph, add slot to all output nodes recursively
  /// @param slot Slot to add
  void addSlot(abstraction::SlotAbstraction *slot) override {
    for (auto notifier : *(this->abstractNotifiers_)) {
      notifier->addSlot(slot);
    }
  }

  /// @brief Remove slot to the graph, add slot to all output nodes recursively
  /// @param slot Slot to remove
  void removeSlot(abstraction::SlotAbstraction *slot) override {
    for (auto notifier : *(this->abstractNotifiers_)) { notifier->removeSlot(slot); }
  }

  /// @brief Do not use, a graph does not have slots by itself
  /// @return nothing throw a std::runtime_error
  /// @throw std::runtime_error A graph has no connected connectedSlots by itself
  [[nodiscard]] std::set<abstraction::SlotAbstraction *> const &connectedSlots() const override {
    throw std::runtime_error("A graph has no connected connectedSlots by itself.");
  }

  /// @brief Do not use, a graph does not notify
  /// @throw std::runtime_error A graph has no slot to notify
  void notify() override { throw std::runtime_error("A graph has no slot to notify."); }

  /// @brief Do not use, a graph does not notify
  /// @throw std::runtime_error A graph has no slot to notify
  void notifyAllTerminated() override { throw std::runtime_error("A graph has no slot to notify."); }
};
}
}
}

#endif //HEDGEHOG_GRAPH_NOTIFIER_H
