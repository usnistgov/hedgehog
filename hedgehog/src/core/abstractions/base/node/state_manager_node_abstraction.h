#ifndef HEDGEHOG_HEDGEHOG_SRC_CORE_ABSTRACTIONS_BASE_NODE_STATE_MANAGER_NODE_ABSTRACTION_H_
#define HEDGEHOG_HEDGEHOG_SRC_CORE_ABSTRACTIONS_BASE_NODE_STATE_MANAGER_NODE_ABSTRACTION_H_

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
  StateManagerNodeAbstraction(std::string const &name, behavior::Node *node) : TaskNodeAbstraction(name, node) {}
  ~StateManagerNodeAbstraction() override = default;

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

#endif //HEDGEHOG_HEDGEHOG_SRC_CORE_ABSTRACTIONS_BASE_NODE_STATE_MANAGER_NODE_ABSTRACTION_H_
