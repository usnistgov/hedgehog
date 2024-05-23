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

#ifndef HEDGEHOG_STATE_MANAGER_NODE_ABSTRACTION_H
#define HEDGEHOG_STATE_MANAGER_NODE_ABSTRACTION_H

#include "task_node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Task core abstraction used to define some method for state manager, heavily base on TaskNodeAbstraction
class StateManagerNodeAbstraction : public TaskNodeAbstraction {
  std::chrono::nanoseconds
      acquireStateDuration_ = std::chrono::nanoseconds::zero(), ///< State acquisition duration
      emptyRdyListDuration_ = std::chrono::nanoseconds::zero(); ///< Empty ready list duration
 public:
  /// @brief Create a state manager node from its name and the user defined node
  /// @param name Name of the node
  /// @param node User defined node
  StateManagerNodeAbstraction(std::string const &name, behavior::Node *node) : TaskNodeAbstraction(name, node) {}

  /// @brief Duration the SM used to acquire its state
  /// @return Accumulated duration to acquire the state
  [[nodiscard]] std::chrono::nanoseconds const &acquireStateDuration() const { return acquireStateDuration_; }

  /// @brief Duration the SM used to empty its ready list
  /// @return Accumulated duration to empty its ready list
  [[nodiscard]] std::chrono::nanoseconds const &emptyRdyListDuration() const {
    return emptyRdyListDuration_;
  }
 protected:
  /// @brief Increment the duration to acquire the state
  /// @param exec Time to acquire the state to add
  void incrementAcquireStateDuration(std::chrono::nanoseconds const &exec){ acquireStateDuration_ += exec; }

  /// @brief Increment the duration to empty the ready list
  /// @param exec Time to empty the ready list to add
  void incrementEmptyRdyListDuration(std::chrono::nanoseconds const &exec){ emptyRdyListDuration_ += exec; }
};

}
}
}

#endif //HEDGEHOG_STATE_MANAGER_NODE_ABSTRACTION_H
