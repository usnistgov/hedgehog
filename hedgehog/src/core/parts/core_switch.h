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

#ifndef HEDGEHOG_CORE_SWITCH_H
#define HEDGEHOG_CORE_SWITCH_H

#include "../../behavior/switch/multi_switch_rules.h"

#include "../abstractions/base/node/node_abstraction.h"
#include "../abstractions/node/task_outputs_management_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

/// @brief Switch core
/// @tparam Inputs Input types list
template<class ...Inputs>
class CoreSwitch : public abstraction::NodeAbstraction,
                   public abstraction::TaskOutputsManagementAbstraction<Inputs...>{

 private:
  behavior::MultiSwitchRules<Inputs...> *const
      multiSwitchRules_ = nullptr; ///< User-defined abstraction with switch rules

 public:
  /// @brief Construct a CoreSwitch with a user-defined abstraction with switch rules
  /// @tparam Switch Type of the user-defined abstraction with switch rules
  /// @param multiSwitchRules User-defined abstraction with switch rules
  template<class Switch>
  requires std::is_base_of_v<behavior::MultiSwitchRules<Inputs...>, Switch>
  explicit CoreSwitch(Switch *const multiSwitchRules)
  : abstraction::NodeAbstraction("Switch"),
  multiSwitchRules_(static_cast<behavior::MultiSwitchRules<Inputs...> *const>(multiSwitchRules)) {}

  /// @brief Default destructor
  ~CoreSwitch() override = default;

  /// @brief Interface to user-defined switch rules
  /// @tparam Input Input data type
  /// @param data Input data
  /// @param graphId Graph id
  /// @return True if the data needs to be send to graph of id graphId, else false
  template<tool::ContainsConcept<Inputs...> Input>
  bool callSendToGraph(std::shared_ptr<Input> &data, size_t const &graphId) {
    return static_cast<behavior::SwitchRule<Input> *>(multiSwitchRules_)->sendToGraph(data, graphId);
  }

  /// @brief Node ids [nodeId, nodeGroupId] accessor
  /// @return  Node ids [nodeId, nodeGroupId]
  [[nodiscard]] std::vector<std::pair<std::string const, std::string const>> ids() const override {
    return {{this->id(), this->id()}};
  }

  /// @brief Getter to the node counterpart
  /// @return Nothing, throw an error because there is no Node attached to the core
  /// @throw std::runtime_error because there is no Node attached to the core
  [[nodiscard]] behavior::Node *node() const override {
    throw std::runtime_error("Try to get a node out of a core switch while there is none.");
  }
};
}
}

#endif //HEDGEHOG_CORE_SWITCH_H
