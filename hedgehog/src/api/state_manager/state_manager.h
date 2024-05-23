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

#ifndef HEDGEHOG_STATE_MANAGER_H
#define HEDGEHOG_STATE_MANAGER_H

#include <utility>

#include "abstract_state.h"
#include "../../behavior/copyable.h"
#include "../../behavior/task_node.h"

/// @brief Hedgehog main namespace
namespace hh {

/// AbstractState manager
/// @brief
/// The state manager is a Hedgehog node that manages locally the state of the computation. To do so, it uses a state
/// protected by a mutex. The state holds the data structures that are used to manage the flow of data and organize
/// rendez-vous points or other synchronization mechanisms.
/// The default order of execution is:
///     -# The StateManager will acquire data,
///     -# The StateManager will lock the AbstractState
///     -# The StateManager will send the data to the AbstractState,
///     -# The compatible Execute::execute is called within the data,
///     -# During the call of Execute::execute, if the method StateMultiSenders::addResult is invoked the "result data"
/// is stored in a ready list,
///     -# When the Execute::execute has returned, the waiting list is emptied as output of the StateManager,
///     -# The StateManager will unlock the AbstractState.
///
/// The state is protected because it can be shared between multiple state managers, either with multiple state
/// managers in the same graph or if the state managers belongs to a graph that is duplicated with an execution
/// pipeline. In this case, each of the state managers in every graph copy will share the same state. This can be
/// used when dealing with computation across multiple GPUs, to synchronize data or share information between devices.
///
/// The state manager can be derived to change its termination rule for example. The method copyStateManager can be
/// derived to customise the copy mechanism of the state manager.
/// @attention A state manager can not be part of a group of threads (multi-threaded).
/// @attention In case of a cycle in the graph CanTerminate::canTerminate needs to be overloaded or the graph will
///// deadlock. By default, AbstractTask::canTerminate will be true if there is no "input node" connected AND no data
///// available in the task input queue (CF tutorial 3).
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class StateManager
    : public behavior::TaskNode,
      public behavior::CanTerminate,
      public behavior::Cleanable,
      public behavior::Copyable<StateManager<Separator, AllTypes...>>,
      public tool::BehaviorMultiReceiversTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>> {
 private:
  std::shared_ptr<AbstractState<Separator, AllTypes...>> state_ = nullptr; ///< AbstractState managed
  std::shared_ptr<core::CoreStateManager<Separator, AllTypes...>>
      coreStateManager_ = nullptr; ///<AbstractState manager core

 public:
  /// @brief Main state manager constructor
  /// @param state AbstractState managed by the state manager
  /// @param name Name of the state manager (default: "AbstractState manager")
  /// @param automaticStart Flag to start the execution of the state manager without data (sending automatically nullptr
  /// to each of the input types)
  /// @throw std::runtime_error the state is not valid (== nullptr or does not derive from CoreStateManager)
  explicit StateManager(
      std::shared_ptr<AbstractState<Separator, AllTypes...>> state,
      std::string const &name = "State manager",
      bool const automaticStart = false)
      : behavior::TaskNode(
      std::make_shared<core::CoreStateManager<Separator, AllTypes...>>(this, state, name, automaticStart)),
        behavior::Copyable<StateManager<Separator, AllTypes...>>(1),
        tool::BehaviorMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(),
        state_(state) {
    if (state == nullptr) {
      throw std::runtime_error("The state given to the state manager should be valid (!= nullptr).");
    }

    if (auto
        coreStateManager = std::dynamic_pointer_cast<core::CoreStateManager<Separator, AllTypes...>>(this->core())) {
      coreStateManager_ = coreStateManager;
    } else {
      throw std::runtime_error("The core used by the state manager should be a CoreStateManager.");
    }
  }

  /// AbstractState Manager constructor with a user-custom core
  /// @brief A custom core can be used to customize how the state manager behaves internally. For example, by default
  /// any input is stored in a std::queue, it can be changed to a std::priority_queue instead through the core.
  /// @param core Custom core used to change the behavior of the state manager
  /// @param state AbstractState managed by the state manager
  /// @throw std::runtime_error the state is not valid (== nullptr or does not derive from CoreStateManager)
  explicit StateManager(
      std::shared_ptr<core::CoreStateManager<Separator, AllTypes...>> core,
      std::shared_ptr<AbstractState<Separator, AllTypes...>> state)
      : behavior::TaskNode(std::move(core)),
        behavior::Copyable<StateManager<Separator, AllTypes...>>(1),
        tool::BehaviorMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>() {
    if (state == nullptr) {
      throw std::runtime_error("The state given to the state manager should be valid (!= nullptr).");
    }
    if (this->core() == nullptr) {
      throw std::runtime_error("The core given to the state manager should be valid (!= nullptr).");
    }
    if (auto
        coreStateManager = std::dynamic_pointer_cast<core::CoreStateManager<Separator, AllTypes...>>(this->core())) {
      coreStateManager_ = coreStateManager;
    } else {
      throw std::runtime_error("The core used by the state manager should be a CoreStateManager.");
    }
  }

  /// @brief Default destructor for the state manager
  ~StateManager() override = default;

  /// @brief AbstractState accessor
  /// @return AbstractState managed by the state manager
  std::shared_ptr<AbstractState<Separator, AllTypes...>> const &state() const { return state_; }

  /// @brief Automatic start flag accessor
  /// @return True if the state manager is set to automatically start, else false
  [[nodiscard]] bool automaticStart() const { return this->coreStateManager()->automaticStart(); }

  /// @brief Accessor to the core
  /// @return Core tot he state manager
  std::shared_ptr<core::CoreStateManager<Separator, AllTypes...>> const &coreStateManager() const {
    return coreStateManager_;
  }

  /// @brief Default termination rule, it terminates if there is no predecessor connection and there is no input data
  /// @return True if the state manager can terminate, else false
  [[nodiscard]] bool canTerminate() const override {
    return !coreStateManager_->hasNotifierConnected() && coreStateManager_->receiversEmpty();
  }

  /// @brief Provide a copy of the state manager
  /// @return A copy of the state manager
  /// @throw std::runtime_error a state manager copy is not valid (== nullptr or do not share the same state)
  std::shared_ptr<StateManager<Separator, AllTypes...>> copy() final {
    auto copy = copyStateManager(this->state_);
    if (copy == nullptr) {
      std::ostringstream oss;
      oss
          << "A copy of the state manager " << this->name() << " has been invoked but return nullptr. "
          << "Please implement the copyStateManager method to create a copy of the current state manager with the same state.";
      throw std::runtime_error(oss.str());
    }
    if (copy->state_ != this->state_) {
      std::ostringstream oss;
      oss << "A copy and the state manager \"" << this->name() << "\" do not share the same state.\n";
      throw std::runtime_error(oss.str());
    }
    return copy;
  }

  /// @brief Customizable copy method
  /// @param state AbstractState to insert in the state manager copy
  /// @return New state manager with the same state
  virtual std::shared_ptr<StateManager<Separator, AllTypes...>>
  copyStateManager(std::shared_ptr<AbstractState<Separator, AllTypes...>> state) {
    return std::make_shared<StateManager<Separator, AllTypes...>>(state, this->name(), this->automaticStart());
  }

};
}

#endif //HEDGEHOG_STATE_MANAGER_H
