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

#include "../abstract_task.h"
#include "abstract_state.h"

/// @brief Hedgehog main namespace
namespace hh {
// Have to add -Woverloaded-virtual for clang because execute hides overloaded virtual function
#if defined (__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif //__clang//

/// @brief Hedgehog behavior namespace
namespace behavior {

/// @brief Base State Manager, needed to avoid virtual inheritance of AbstractTask
/// @tparam StateOutput State output type
/// @tparam StateInputs State input types
template<class StateOutput, class ...StateInputs>
class BaseStateManager : public AbstractTask<StateOutput, StateInputs...> {
 public:
  /// @brief DEfault constructor, set the base property for a state manager
  BaseStateManager()
      : AbstractTask<StateOutput, StateInputs...>("StateManager", 1, core::NodeType::StateManager, false) {}

};

/// @brief Implementation of the execute method for the StateManager
/// @details
/// The default order of execution is:
///     -# The StateManager will acquire data,
///     -# The StateManager will lock the AbstractState (with AbstractState::stateMutex_),
///     -# The StateManager will send the data to the possessed AbstractState,
///     -# The compatible Execute::execute is called with the data,
///     -# During the call of Execute::execute, if the method AbstractState::push is invoked the "result data" is
/// stored in a ready list AbstractState::readyList_,
///     -# When the Execute::execute has returned, the waiting list is emptied as output of the
/// AbstractStateManager,
///     -# The StateManager will unlock the AbstractState (with AbstractState::stateMutex_).
/// @attention Should not be used directly, use StateManager instead.
/// @tparam StateInput State input type for definition of Execute::execute
/// @tparam StateOutput State output type
/// @tparam StateInputs State input types
template<class StateInput, class StateOutput, class ...StateInputs>
class StateManagerExecuteDefinition : public virtual BaseStateManager<StateOutput, StateInputs...> {
 private:
  std::shared_ptr<AbstractState<StateOutput, StateInputs...>> es_; ///< State used in execute definition
 public:
  /// @brief Deleted default constructor
  StateManagerExecuteDefinition() = delete;
  /// @brief Constructor for an AbstractStateManager with name, state, and automaticStart as mandatory parameters
  /// @param state State to manage
  explicit StateManagerExecuteDefinition(std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state)
      : es_(state) { }

  /// @brief Default StateManagerExecute destructor
  virtual ~StateManagerExecuteDefinition() = default;

  /// @brief Default execute method definition
  /// @param input Data of type StateInput to be send to the managed state
  void execute(std::shared_ptr<StateInput> input) final {
    this->es_->lock();
    std::static_pointer_cast<behavior::Execute<StateInput>>(this->es_)->execute(input);
    while (!this->es_->readyList()->empty()) { this->addResult(this->es_->frontAndPop()); }
    this->es_->unlock();
  }
};

}
#if defined (__clang__)
#pragma clang diagnostic pop
#endif //__clang//

/// @brief Hedgehog graph's node, used to manage an AbstractState
/// @details To locally manage a computation, an AbstractStateManager with an AbstractState is used. An
/// AbstractStateManager is a Hedgehog graphs's node so it can be linked to any other nodes in a graph.
/// The default order of execution is:
///     -# The StateManager will acquire a data,
///     -# The StateManager will lock the AbstractState (with AbstractState::stateMutex_),
///     -# The StateManager will send the data to the possessed AbstractState,
///     -# The compatible Execute::execute is called with the data,
///     -# During the call of Execute::execute, if the method AbstractState::push is invoked the "result data" is
/// stored in a ready list AbstractState::readyList_,
///     -# When the Execute::execute has returned, the waiting list is emptied as output of the
/// AbstractStateManager,
///     -# The StateManager will unlock the AbstractState (with AbstractState::stateMutex_).
/// @par Virtual functions
/// AbstractTask::copy (only used if the AbstractStateManager is in an ExecutionPipeline) <br>
/// AbstractTask::initialize <br>
/// AbstractTask::shutdown <br>
/// Node::canTerminate <br>
/// Node::extraPrintingInformation
/// @tparam StateOutput State output type
/// @tparam StateInputs State input types
template<class StateOutput, class ...StateInputs>
 class StateManager : public behavior::StateManagerExecuteDefinition<StateInputs, StateOutput, StateInputs...> ... {

 private:
  std::shared_ptr<AbstractState<StateOutput, StateInputs...>> state_ = nullptr; ///< State to manage

 public:
  using behavior::StateManagerExecuteDefinition<StateInputs, StateOutput, StateInputs...>::execute...;

  /// @brief Deleted default constructor
  StateManager() = delete;

  /// @brief Constructor for a StateManager with state as mandatory parameter
  /// @details By default the node name is "StateManager", and there is no automatic start
  /// @tparam StateType User defined state type
  /// @tparam IsCompatibleState Type defined if the object given is a compatible State (derived from State with the
  /// same StateOutput and StateInputs)
  /// @param state State to manage
  template<
      class StateType,
      class IsCompatibleState = typename std::enable_if_t<
          std::is_base_of_v<AbstractState<StateOutput, StateInputs...>, StateType>
      >
  >
  explicit StateManager(std::shared_ptr<StateType> const state) :
      behavior::StateManagerExecuteDefinition<StateInputs, StateOutput, StateInputs...>(state)...,
  state_(state) {}

  /// @brief Constructor for an StateManager with state and automatic start as mandatory parameter
  /// @details The default node name is "StateManager"
  /// @tparam StateType User defined state type
  /// @tparam IsCompatibleState Type defined if the object given is a compatible State (derived from State with the
  /// same StateOutput and StateInputs)
  /// @param state State to manage
  /// @param automaticStart Node automatic start
  template<
      class StateType,
      class IsCompatibleState = typename std::enable_if_t<
          std::is_base_of_v<AbstractState<StateOutput, StateInputs...>, StateType>
      >
  >
  StateManager(std::shared_ptr<StateType> const state, bool automaticStart) :
      behavior::StateManagerExecuteDefinition<StateInputs, StateOutput, StateInputs...>(state)...,
  state_(state) {
    std::dynamic_pointer_cast<core::CoreTask<StateOutput, StateInputs...>>(this->core())->automaticStart(automaticStart);
  }

  /// @brief Constructor for a StateManager with name and state as mandatory parameter, and automatic start as
  /// optional parameter
  /// @tparam StateType User defined state type
  /// @tparam IsCompatibleState Type defined if the object given is a compatible State (derived from State with the
  /// same StateOutput and StateInputs)
  /// @param name Node name
  /// @param state State to manage
  /// @param automaticStart Node automatic start
  template<
      class StateType,
      class IsCompatibleState = typename std::enable_if_t<
          std::is_base_of_v<AbstractState<StateOutput, StateInputs...>, StateType>
      >
  >
  StateManager(std::string_view const name,
               std::shared_ptr<StateType> const state,
               bool automaticStart = false) :
      behavior::StateManagerExecuteDefinition<StateInputs, StateOutput, StateInputs...>(state)..., state_(state) {
    this->core()->name(name);
    std::dynamic_pointer_cast<core::CoreTask<StateOutput, StateInputs...>>(this->core())->automaticStart(automaticStart);
  }

  /// @brief Default override of the copy method, copy the name, the state and the automatic start.
  /// @return A copy of the current StateManager with the same name, state and with the automatic start.
  std::shared_ptr<AbstractTask<StateOutput, StateInputs...>> copy() override {
    return std::make_shared<StateManager<StateOutput, StateInputs...>>(this->name(),
                                                                       this->state(),
                                                                       this->automaticStart());
  }

 protected:
  /// @brief Managed state accessor
  /// @return Managed state
  std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state() const { return state_; }
};
}
#endif //HEDGEHOG_STATE_MANAGER_H
