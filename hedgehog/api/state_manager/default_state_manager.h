//
// Created by anb22 on 6/10/19.
//

#ifndef HEDGEHOG_DEFAULT_STATE_MANAGER_H
#define HEDGEHOG_DEFAULT_STATE_MANAGER_H

#include "abstract_state_manager.h"

// Have to add -Woverloaded-virtual for clang because execute hides overloaded virtual function
#if defined (__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif //__clang//
template<class StateInput, class StateOutput, class ...StateInputs>
class DefaultStateManagerExecute : public virtual AbstractStateManager<StateOutput, StateInputs...> {
 public:
  DefaultStateManagerExecute() = delete;
  DefaultStateManagerExecute(std::string_view const &name,
                             std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state,
                             bool automaticStart) :

      AbstractStateManager<StateOutput, StateInputs...>(name,
                                                        state,
                                                        automaticStart) {}

  void execute(std::shared_ptr<StateInput> input) final {
    this->state()->lock();
    std::static_pointer_cast<Execute<StateInput>>(this->state())->execute(input);
    while (!this->state()->readyList()->empty()) {
      this->addResult(this->state()->frontAndPop());
    }
    this->state()->unlock();
  }
};
#if defined (__clang__)
#pragma clang diagnostic pop
#endif //__clang//

template<class StateOutput, class ...StateInputs>
class DefaultStateManager
    : public DefaultStateManagerExecute<StateInputs, StateOutput, StateInputs...> ... {
 public:
  using DefaultStateManagerExecute<StateInputs, StateOutput, StateInputs...>::execute...;

  explicit DefaultStateManager(std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state,
                               bool automaticStart = false)
      :
      AbstractStateManager<StateOutput, StateInputs...>("DefaultStateManager", state, automaticStart),
      DefaultStateManagerExecute<StateInputs, StateOutput, StateInputs...>("DefaultStateManager",
                                                                           state,
                                                                           automaticStart)... {}

  DefaultStateManager(std::string_view const name,
                      std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state,
                      bool automaticStart = false) :
      AbstractStateManager<StateOutput, StateInputs...>(name, state, automaticStart),
      DefaultStateManagerExecute<StateInputs, StateOutput, StateInputs...>(name, state, automaticStart)... {}

};
#endif //HEDGEHOG_DEFAULT_STATE_MANAGER_H
