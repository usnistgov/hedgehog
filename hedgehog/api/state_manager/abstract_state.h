//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_ABSTRACT_STATE_H
#define HEDGEHOG_ABSTRACT_STATE_H

#include <memory>
#include <queue>
#include <shared_mutex>

#include "../../behaviour/execute.h"

template<class StateOutput, class ...StateInputs>
class AbstractState : public Execute<StateInputs> ... {
 private:
  mutable std::unique_ptr<std::shared_mutex> stateMutex_ = nullptr;
  std::unique_ptr<std::queue<std::shared_ptr<StateOutput>>> readyList_ = nullptr;

 public:
  AbstractState() {
    stateMutex_ = std::make_unique<std::shared_mutex>();
    readyList_ = std::make_unique<std::queue<std::shared_ptr<StateOutput>>>();
  }

  virtual ~AbstractState() = default;

  std::unique_ptr<std::queue<std::shared_ptr<StateOutput>>> const &readyList() const {
    return readyList_;
  }

  void push(std::shared_ptr<StateOutput> const &elem) {
    readyList_->push(elem);
  }

  std::shared_ptr<StateOutput> frontAndPop() {
    std::shared_ptr<StateOutput> elem = readyList_->front();
    readyList_->pop();
    return elem;
  }

  void lock() {
    stateMutex_->lock();
  }

  void unlock() {
    stateMutex_->unlock();
  }

 protected:
  void outputReady(std::shared_ptr<StateOutput> const &readyOutput) {
    std::shared_lock lock(readyList_);
    readyList_->insert(readyOutput);
  }
};

#endif //HEDGEHOG_ABSTRACT_STATE_H
