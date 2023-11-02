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

#ifndef HEDGEHOG_CORE_STATE_MANAGER_H_
#define HEDGEHOG_CORE_STATE_MANAGER_H_

#include <ostream>
#include <sstream>

#include "../../tools/traits.h"
#include "../../tools/concepts.h"
#include "../../tools/meta_functions.h"

#include "../abstractions/base/cleanable_abstraction.h"
#include "../abstractions/base/copyable_abstraction.h"
#include "../abstractions/base/can_terminate_abstraction.h"
#include "../abstractions/node/task_inputs_management_abstraction.h"
#include "../abstractions/node/task_outputs_management_abstraction.h"
#include "../abstractions/base/node/state_manager_node_abstraction.h"
#include "../implementors/concrete_implementor/multi_queue_receivers.h"
#include "../implementors/concrete_implementor/default_multi_executes.h"
#include "../../behavior/input_output/state_sender.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration StateManager
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class StateManager;

/// @brief Forward declaration AbstractState
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class AbstractState;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog core namespace
namespace core {

/// @brief Type alias for an TaskInputsManagementAbstraction from the list of template parameters
template<size_t Separator, class ...AllTypes>
using TIM = tool::TaskInputsManagementAbstractionTypeDeducer_t
    <tool::Inputs<Separator, AllTypes...>>;

/// @brief Type alias for an TaskOutputsManagementAbstraction from the list of template parameters
template<size_t Separator, class ...AllTypes>
using TOM = tool::TaskOutputsManagementAbstractionTypeDeducer_t
    <tool::Outputs<Separator, AllTypes...>>;

/// @brief AbstractState manager core
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class CoreStateManager
    : public abstraction::StateManagerNodeAbstraction,
      public abstraction::ClonableAbstraction,
      public abstraction::CleanableAbstraction,
      public abstraction::CanTerminateAbstraction,
      public abstraction::CopyableAbstraction<StateManager<Separator, AllTypes...>>,
      public TIM<Separator, AllTypes...>,
      public TOM<Separator, AllTypes...> {
 private:
  StateManager<Separator, AllTypes...> *const
      stateManager_ = nullptr; ///< User defined state manager

  std::shared_ptr<AbstractState<Separator, AllTypes...>> const
      state_ = nullptr; ///< Managed state

  bool const
      automaticStart_ = false; ///< Flag for automatic start

 public:
  /// @brief Construct a state manager from the user state manager and its state
  /// @param stateManager User-defined state manager
  /// @param state AbstractState to manage
  explicit CoreStateManager(StateManager<Separator, AllTypes...> *const stateManager,
                            std::shared_ptr<AbstractState<Separator, AllTypes...>> const &state)
      : CoreStateManager<Separator, AllTypes...>(stateManager, state, "AbstractState Manager", false) {}

  /// @brief Construct a state manager from the user state manager, its state, a name and the automatic start flag
  /// @param stateManager User-defined state manager
  /// @param state AbstractState to manage
  /// @param name AbstractState manager name
  /// @param automaticStart Flag for automatic start
  CoreStateManager(
      StateManager<Separator, AllTypes...> *const stateManager,
      std::shared_ptr<AbstractState<Separator, AllTypes...>> const &state,
      std::string const &name, bool const automaticStart)
      : StateManagerNodeAbstraction(name, stateManager),
        CleanableAbstraction(static_cast<behavior::Cleanable *>(stateManager)),
        CanTerminateAbstraction(static_cast<behavior::CanTerminate *>(stateManager)),
        abstraction::CopyableAbstraction<StateManager<Separator, AllTypes...>>(stateManager),
        TIM<Separator, AllTypes...>(
            this,
            std::make_shared<implementor::DefaultSlot>(),
            std::make_shared<tool::MultiQueueReceiversTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>>(),
            std::make_shared<hh::tool::DefaultMultiExecutesTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>>(
                state.get()
            )
        ),
        TOM<Separator, AllTypes...>(),
        stateManager_(stateManager),
        state_(state),
        automaticStart_(automaticStart) {}

  /// @brief Construct a state manager from the state manager and its concrete abstraction's implementations
  /// @tparam ConcreteMultiReceivers Type of concrete implementation of ReceiverAbstraction for multiple types
  /// @tparam ConcreteMultiExecutes Type of concrete implementation of ExecuteAbstraction for multiple types
  /// @tparam ConcreteMultiSenders Type of concrete implementation of SenderAbstraction for multiple types
  /// @param stateManager User-defined state manager
  /// @param state AbstractState to manage
  /// @param name AbstractState manager name
  /// @param automaticStart Flag for automatic start
  /// @param concreteSlot Concrete implementation of SlotAbstraction
  /// @param concreteMultiReceivers Concrete implementation of ReceiverAbstraction for multiple types
  /// @param concreteMultiExecutes Concrete implementation of ExecuteAbstraction for multiple types
  /// @param concreteNotifier Concrete implementation of NotifierAbstraction
  /// @param concreteMultiSenders Concrete implementation of SenderAbstraction for multiple types
  template<class ConcreteMultiReceivers, class ConcreteMultiExecutes, class ConcreteMultiSenders>
  CoreStateManager(StateManager<Separator, AllTypes...> *const stateManager,
                   std::shared_ptr<AbstractState<Separator, AllTypes...>> const &state,
                   std::string const &name, bool const automaticStart,
                   std::shared_ptr<implementor::ImplementorSlot> const &concreteSlot,
                   std::shared_ptr<ConcreteMultiReceivers> concreteMultiReceivers,
                   std::shared_ptr<ConcreteMultiExecutes> concreteMultiExecutes,
                   std::shared_ptr<implementor::ImplementorNotifier> const &concreteNotifier,
                   std::shared_ptr<ConcreteMultiSenders> concreteMultiSenders) :
      StateManagerNodeAbstraction(name, stateManager),
      CleanableAbstraction(static_cast<behavior::Cleanable *>(stateManager)),
      CanTerminateAbstraction(static_cast<behavior::CanTerminate *>(stateManager)),
      abstraction::CopyableAbstraction<StateManager<Separator, AllTypes...>>(stateManager),
      TIM<Separator, AllTypes...>(this, concreteSlot, concreteMultiReceivers, concreteMultiExecutes),
      TOM<Separator, AllTypes...>(concreteNotifier, concreteMultiSenders),
      stateManager_(stateManager),
      state_(state),
      automaticStart_(automaticStart) {}

  /// Default destructor
  ~CoreStateManager() override = default;

  /// @brief Accessor to the automatic start flag
  /// @return true if the core start automatically, else false
  [[nodiscard]] bool automaticStart() const { return automaticStart_; }

  /// @brief Call user-definable termination
  /// @param lock Flag if the call to the user-definable termination need to be protected
  /// @return True if the node can terminate, else false
  bool callCanTerminate(bool lock) override {
    bool result;
    if (lock) { this->lockSlotMutex(); }
    result = this->callNodeCanTerminate();
    if (lock) { this->unlockSlotMutex(); }
    return result;
  }

  /// @brief Visit the state manager
  /// @param printer Printer gathering node information
  void visit(Printer *printer) override {
    if (printer->registerNode(this)) {
      printer->printNodeInformation(this);
      TIM<Separator, AllTypes...>::printEdgesInformation(printer);
    }
  }

  /// @brief Node ids [nodeId, nodeGroupId] accessor
  /// @return  Node ids [nodeId, nodeGroupId]
  [[nodiscard]] std::vector<std::pair<std::string const, std::string const>> ids() const override {
    return {{this->id(), this->id()}};
  }

  /// @brief Initialize the state manager
  /// @details Call user define initialize, initialize memory manager if present
  void preRun() override {
    this->nvtxProfiler()->startRangeInitializing();
    this->stateManager_->initialize();
    if (this->stateManager_->memoryManager() != nullptr) {
      this->stateManager_->memoryManager()->profiler(this->nvtxProfiler());
      this->stateManager_->memoryManager()->deviceId(this->deviceId());
      this->stateManager_->memoryManager()->initialize();
    }
    this->nvtxProfiler()->endRangeInitializing();
    this->setInitialized();
  }

  /// @brief Main core state manager logic
  /// @details
  /// - if automatic start
  ///     - lock the state
  ///     - link the state-manager to the state
  ///     - call user-defined state's execute method with nullptr
  ///     - empty ready list
  ///     - unlink the state-manager to the state
  ///     - unlock the state
  /// - while the state-manager runs
  ///     - wait for data or termination
  ///     - if can terminate, break
  ///     - lock the state
  ///     - get a piece of data from the queue
  ///     - link the state-manager to the state
  ///     - call user-defined state's execute method with data
  ///     - empty ready list
  ///     - unlink the state-manager to the state (ensures correct state manager is used, such as with memory manager)
  ///     - unlock the state
  /// - shutdown the state manager
  /// - notify successors state manager terminated
  void run() override {
    using Outputs_t = tool::Outputs<Separator, AllTypes...>;
    using Indices = std::make_index_sequence<std::tuple_size_v<Outputs_t>>;
    Indices indices{};
    std::chrono::time_point<std::chrono::system_clock>
        start,
        finish;

    this->isActive(true);
    this->nvtxProfiler()->initialize(0);
    this->preRun();

    if (this->automaticStart_) {
      start = std::chrono::system_clock::now();
      state_->lock();
      state_->stateManager(this->stateManager_);
      finish = std::chrono::system_clock::now();
      this->incrementAcquireStateDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));

//      start = std::chrono::system_clock::now();
      this->callAllExecuteWithNullptr();
      emptyReadyLists<Outputs_t>(indices);
      state_->stateManager(nullptr);
      state_->unlock();
//      finish = std::chrono::system_clock::now();
//      this->incrementDequeueExecutionDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));
    }

    // Actual computation loop
    while (!this->callCanTerminate(true)) {
      // Wait for a data to arrive or termination
      start = std::chrono::system_clock::now();
      volatile bool canTerminate = this->wait();
      finish = std::chrono::system_clock::now();
      this->incrementWaitDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));

      // If it can terminate break the loop early
      if (canTerminate) { break; }

      // Operate the connectedReceivers to get a data and send it to execute
      start = std::chrono::system_clock::now();
      state_->lock();
      state_->stateManager(this->stateManager_);
      finish = std::chrono::system_clock::now();
      this->incrementAcquireStateDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));

//      start = std::chrono::system_clock::now();
      this->operateReceivers();
      start = std::chrono::system_clock::now();
      emptyReadyLists<Outputs_t>(indices);
            finish = std::chrono::system_clock::now();
      this->incrementEmptyRdyListDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));
      state_->stateManager(nullptr);
      state_->unlock();
//      finish = std::chrono::system_clock::now();
//      this->incrementDequeueExecutionDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));
    }

    // Do the shutdown phase
    this->postRun();
    // Wake up node that this is linked to
    this->wakeUp();
  }

  /// @brief Post run logic, shutdown the state manager, notify successor nodes of state manager termination
  void postRun() override {
    this->nvtxProfiler()->startRangeShuttingDown();
    this->isActive(false);
    this->stateManager_->shutdown();
    this->notifyAllTerminated();
    this->nvtxProfiler()->endRangeShuttingDown();
  }

  /// @brief Test if a memory manager is attached
  /// @return True if there is a memory manager attached, else false
  [[nodiscard]] bool hasMemoryManagerAttached() const override { return this->memoryManager() != nullptr; }

  /// @brief Accessor to user-defined extra information for the state-manager
  /// @return User-defined extra information for the state-manager
  [[nodiscard]] std::string extraPrintingInformation() const override {
    return this->stateManager_->extraPrintingInformation();
  }

  /// @brief Accessor to the memory manager
  /// @return The attached memory manager
  [[nodiscard]] std::shared_ptr<AbstractMemoryManager> memoryManager() const override {
    return this->stateManager_->memoryManager();
  }

 protected:
  /// @brief Wait statement when the node is alive and no input data are available
  /// @return True if the node can terminate, else false
  bool wait() {
    this->nvtxProfiler()->startRangeWaiting(this->totalNumberElementsReceived());
    std::unique_lock<std::mutex> lock(*(this->mutex()));
    this->slotConditionVariable()->wait(
        lock,
        [this]() { return !this->receiversEmpty() || this->callCanTerminate(false); }
    );
    this->nvtxProfiler()->endRangeWaiting();
    return callCanTerminate(false);
  }

 private:
  /// @brief Gather the state manager and the state into the cleanableSet
  /// @param cleanableSet Set of cleanable nodes
  void gatherCleanable(std::unordered_set<hh::behavior::Cleanable *> &cleanableSet) override {
    CleanableAbstraction::gatherCleanable(cleanableSet);
    cleanableSet.insert(static_cast<behavior::Cleanable *>(this->state_.get()));
  }

  /// @brief Interface method to call emptyReadyLists for a type when decomposing types from a tuple
  /// @tparam Outputs_t Tuple of types
  /// @tparam Indexes Indexes of tuple
  template<class Outputs_t, size_t ...Indexes>
  void emptyReadyLists(std::index_sequence<Indexes...>) {
    (emptyReadyLists<std::tuple_element_t<Indexes, Outputs_t>>(), ...);
  }

  /// @brief Empty the ready list for a type
  /// @tparam Output Ready list output type
  template<class Output>
  void emptyReadyLists() {
    auto &rdyList = std::static_pointer_cast<behavior::StateSender<Output>>(state_)->readyList();
    std::shared_ptr<Output> data = nullptr;
    while (!rdyList->empty()) {
      data = rdyList->front();
      rdyList->pop();
      this->sendAndNotify(data);
    }
  }

  /// @brief Clone method, to duplicate a state manager when it is part of another graph in an execution pipeline
  /// @param correspondenceMap Correspondence map of belonging graph's node
  /// @return Clone of this state manager
  std::shared_ptr<abstraction::NodeAbstraction> clone(
      [[maybe_unused]] std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &correspondenceMap) override {
    auto clone = std::dynamic_pointer_cast<StateManager<Separator, AllTypes...>>(this->callCopy());
    if (this->hasMemoryManagerAttached()) { clone->connectMemoryManager(this->memoryManager()->copy()); }
    return clone->core();
  }

  /// @brief Duplicate the state manager edge
  /// @param mapping Correspondence map of belonging graph's node
  void duplicateEdge(std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping) override {
    this->duplicateOutputEdges(mapping);
  }

  /// @brief Accessor to the execution duration per input
  /// @return A Map where the key is the type as string, and the value is the associated duration
  [[nodiscard]] std::map<std::string, std::chrono::nanoseconds> const &executionDurationPerInput() const final {
    return this->executionDurationPerInput_;
  }

  /// @brief Accessor to the number of elements per input
  /// @return A Map where the key is the type as string, and the value is the associated number of elements received
  [[nodiscard]] std::map<std::string, std::size_t> const &nbElementsPerInput() const final {
    return this->nbElementsPerInput_;
  }

  /// @brief Accessor to the dequeue + execution duration per input
  /// @return Map in which the key is the type and the value is the duration
  [[nodiscard]] std::map<std::string, std::chrono::nanoseconds> const &dequeueExecutionDurationPerInput() const final {
    return this->dequeueExecutionDurationPerInput_;
  }

};

}
}
#endif //HEDGEHOG_CORE_STATE_MANAGER_H_
