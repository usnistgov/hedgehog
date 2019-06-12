//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_ABSTRACT_STATE_MANAGER_H
#define HEDGEHOG_ABSTRACT_STATE_MANAGER_H

#include "../task/abstract_task.h"
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

#endif //HEDGEHOG_ABSTRACT_STATE_MANAGER_H
