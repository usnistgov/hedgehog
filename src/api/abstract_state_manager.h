//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_ABSTRACT_STATE_MANAGER_H
#define HEDGEHOG_ABSTRACT_STATE_MANAGER_H

#include "abstract_task.h"
#include "abstract_state.h"

template<class StateOutput, class ...StateInputs>
class AbstractStateManager : public AbstractTask<StateOutput, StateInputs...> {
 private:
  std::shared_ptr<AbstractState<StateOutput, StateInputs...>> state_ = nullptr;

 protected:
 public:
  AbstractStateManager() = delete;

  explicit AbstractStateManager(std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state,
                                bool automaticStart = false)
      : AbstractTask<StateOutput, StateInputs...>("StateManager", 1, NodeType::StateManager, automaticStart) {
    assert(state != nullptr);
    this->state_ = state;
  }

  AbstractStateManager(std::string_view const &name,
                       std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state,
                       bool automaticStart = false)
      : AbstractTask<StateOutput, StateInputs...>(name, 1, NodeType::StateManager, automaticStart) {
    assert(state != nullptr);
    this->state_ = state;
  }

  ~AbstractStateManager() override = default;

 protected:
  std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state() const {
    return state_;
  }
  std::shared_ptr<AbstractTask<StateOutput, StateInputs...>> copy() final { return nullptr; }

};

template<class StateInput, class StateOutput, class ...StateInputs>
class DefaultStateManagerExecute : public virtual AbstractStateManager<StateOutput, StateInputs...> {
 public:
  void execute(std::shared_ptr<StateInput> input) final {
    this->state()->lock();
    std::static_pointer_cast<Execute<StateInput>>(this->state())->execute(input);
    while (!this->state()->readyList()->empty()) {
      this->addResult(this->state()->frontAndPop());
    }
    this->state()->unlock();
  }
};

template<class StateOutput, class ...StateInputs>
class DefaultStateManager
    : public DefaultStateManagerExecute<StateInputs, StateOutput, StateInputs...> ... {
 public:
  using DefaultStateManagerExecute<StateInputs, StateOutput, StateInputs...>::execute...;
  explicit DefaultStateManager(std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state,
                               bool automaticStart = false)
      : AbstractStateManager<StateOutput, StateInputs...>("DefaultStateManager", state, automaticStart) {}

  DefaultStateManager(std::string_view const name,
                      std::shared_ptr<AbstractState<StateOutput, StateInputs...>> const &state,
                      bool automaticStart = false) : AbstractStateManager<StateOutput, StateInputs...>(name,
                                                                                                       state,
                                                                                                       automaticStart) {}

};

#endif //HEDGEHOG_ABSTRACT_STATE_MANAGER_H
