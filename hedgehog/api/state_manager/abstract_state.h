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


#ifndef HEDGEHOG_ABSTRACT_STATE_H
#define HEDGEHOG_ABSTRACT_STATE_H

#include <memory>
#include <queue>
#include <shared_mutex>

#include "../../behavior/execute.h"
#include "../../tools/traits.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief State Interface for managing computation, need a corresponding AbstractStateManager to be embedded in a
/// Hedgehog Graph
/// @details An AbstractState is a tool used by Hedgehog when data synchronization is needed. When overloaded,
/// data structures that are added to the class can manage the state of computation.
/// AbstractState owns a mutex to guarantee safety, because it can be shared among multiple AbstractStateManager.
///
/// The default order of execution is:
///     -# The DefaultStateManager will acquire a data,
///     -# The DefaultStateManager will lock the AbstractState (with AbstractState::stateMutex_),
///     -# The DefaultStateManager will send the data to the possessed AbstractState,
///     -# The compatible Execute::execute is called with the data,
///     -# During the call of Execute::execute, if the method AbstractState::push is invoked the "result data" is
/// stored in a ready list AbstractState::readyList_,
///     -# When the Execute::execute has returned, the waiting list is emptied as output of the
/// AbstractStateManager,
///     -# The DefaultStateManager will unlock the AbstractState (with AbstractState::stateMutex_).
///
/// @par Virtual functions
/// Execute::execute (one for each of StateInputs)
///
/// @attention An AbstractState can be only owned by a compatible AbstractStateManager, i.e. they have the same
/// StateOutput and the same StateInputs.
/// @tparam StateOutput State output type
/// @tparam StateInputs State input types
template<class StateOutput, class ...StateInputs>
class AbstractState : public behavior::Execute<StateInputs> ... {
  static_assert(traits::isUnique<StateInputs...>, "A Task can't accept multiple inputs with the same type.");
  static_assert(sizeof... (StateInputs) >= 1, "A state need to have one output type and at least one output type.");
 private:
  mutable std::unique_ptr<std::shared_mutex> stateMutex_ = nullptr; ///< State Mutex
  std::unique_ptr<std::queue<std::shared_ptr<StateOutput>>> readyList_ = nullptr; ///< State Ready list

 public:
  /// @brief Default constructor, initialize the mutex (AbstractState::stateMutex_) and the ready list
  /// (AbstractState::readyList_)
  AbstractState() {
    stateMutex_ = std::make_unique<std::shared_mutex>();
    readyList_ = std::make_unique<std::queue<std::shared_ptr<StateOutput>>>();
  }

  /// @brief Default destructor
  virtual ~AbstractState() = default;

  /// @brief Ready list accessor
  /// @return Ready list
  std::unique_ptr<std::queue<std::shared_ptr<StateOutput>>> const &readyList() const { return readyList_; }

  /// @brief Add an element to the ready list
  /// @param elem Element to add
  void push(std::shared_ptr<StateOutput> const &elem) { readyList_->push(elem); }

  /// @brief Used by AbstractStateManager to get the ready list's front element
  /// @return The ready list's front element
  std::shared_ptr<StateOutput> frontAndPop() {
    std::shared_ptr<StateOutput> elem = readyList_->front();
    readyList_->pop();
    return elem;
  }

  /// @brief Lock the state
  void lock() { stateMutex_->lock(); }

  /// @brief Unlock the state
  void unlock() { stateMutex_->unlock(); }
};

}
#endif //HEDGEHOG_ABSTRACT_STATE_H
