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

#ifndef HEDGEHOG_ABSTRACT_STATE_H_
#define HEDGEHOG_ABSTRACT_STATE_H_

#include <shared_mutex>

#include "../../tools/traits.h"
#include "../../behavior/cleanable.h"
#include "../../behavior/multi_execute.h"
#include "../../behavior/input_output/state_multi_senders.h"
#include "../../core/nodes/core_state_manager.h"

#include "../memory_manager/manager/abstract_memory_manager.h"

/// @brief Hedgehog main namespace
namespace hh {

/// Hedgehog AbstractState
/// @brief The state holds the data structures that are used to manage the flow of data and organize
/// rendez-vous points or other synchronization mechanisms. It is managed and used through a StateManager, and is thread safe.
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class AbstractState
    : public behavior::Cleanable,
      public tool::BehaviorMultiExecuteTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorStateMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>> {
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  /// @brief Declare core::CoreStateManager as friend
  friend core::CoreStateManager<Separator, AllTypes...>;
#endif //DOXYGEN_SHOULD_SKIP_THIS
 private:
  mutable std::unique_ptr<std::shared_mutex> mutex_ = nullptr; ///< Mutex to protect the state
  StateManager<Separator, AllTypes...> *stateManager_ = nullptr; ///< AbstractState manager currently using the state
 public:
  /// @brief Default state constructor
  AbstractState() : mutex_(std::make_unique<std::shared_mutex>()) {}
  /// @brief Default state destructor
  ~AbstractState() override = default;

  /// @brief Accessor to the managed memory if a memory manager has been attached to a StateManager
  /// @return Return a managed memory from the memory manager attached to the state manager
  std::shared_ptr<ManagedMemory> getManagedMemory() { return stateManager_->getManagedMemory(); }

  /// @brief Lock the state
  void lock() { mutex_->lock(); }

  /// @brief Unlock the state
  void unlock() { mutex_->unlock(); }

 private:
  /// AbstractState manager setter
  /// @details Setter used by Hedgehog to indicate which state manager is currently using a specific state.
  /// @param stateManager AbstractState manager managing the state
  void stateManager(StateManager<Separator, AllTypes...> *stateManager) { stateManager_ = stateManager; }
};
}
#endif //HEDGEHOG_ABSTRACT_STATE_H_
