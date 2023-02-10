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

#ifndef HEDGEHOG_CORE_TASK_H
#define HEDGEHOG_CORE_TASK_H

#include <ostream>
#include <sstream>

#include "../../tools/concepts.h"
#include "../../tools/traits.h"

#include "../abstractions/base/clonable_abstraction.h"
#include "../abstractions/base/groupable_abstraction.h"
#include "../abstractions/base/cleanable_abstraction.h"
#include "../abstractions/base/can_terminate_abstraction.h"

#include "../abstractions/base/node/task_node_abstraction.h"

#include "../abstractions/node/task_inputs_management_abstraction.h"
#include "../abstractions/node/task_outputs_management_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration AbstractTask
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class AbstractTask;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog core namespace
namespace core {

/// @brief Type alias for an TaskInputsManagementAbstraction from the list of template parameters
template<size_t Separator, class ...AllTypes>
using TIM = tool::TaskInputsManagementAbstractionTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>;

/// @brief Type alias for an TaskOutputsManagementAbstraction from the list of template parameters
template<size_t Separator, class ...AllTypes>
using TOM = tool::TaskOutputsManagementAbstractionTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>;

/// @brief Task core
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class CoreTask
    : public abstraction::TaskNodeAbstraction,
      public abstraction::ClonableAbstraction,
      public abstraction::CleanableAbstraction,
      public abstraction::CanTerminateAbstraction,
      public abstraction::GroupableAbstraction<AbstractTask<Separator, AllTypes...>, CoreTask<Separator, AllTypes...>>,
      public TIM<Separator, AllTypes...>,
      public TOM<Separator, AllTypes...> {

 private:
  AbstractTask<Separator, AllTypes...> *const
      task_ = nullptr; ///< User defined task

  bool const
      automaticStart_ = false; ///< Flag for automatic start

 public:
  /// @brief Create a CoreTask from a user-defined AbstractTask with one thread
  /// @param task User-defined AbstractTask
  explicit CoreTask(AbstractTask<Separator, AllTypes...> *const task) :
      TaskNodeAbstraction("Task", task),
      CleanableAbstraction(static_cast<behavior::Cleanable *>(task)),
      CanTerminateAbstraction(static_cast<behavior::CanTerminate *>(task)),
      abstraction::GroupableAbstraction<AbstractTask<Separator, AllTypes...>, CoreTask<Separator, AllTypes...>>(task,
                                                                                                                1),
      TIM<Separator, AllTypes...>(task, this),
      TOM<Separator, AllTypes...>(),
      task_(task),
      automaticStart_(false) {}

  /// @brief Create a CoreTask from a user-defined AbstractTask, its  name, the number of threads and the automatic
  /// start flag
  /// @param task User-defined AbstractTask
  /// @param name Task's name
  /// @param numberThreads Number of threads
  /// @param automaticStart Flag for automatic start
  CoreTask(AbstractTask<Separator, AllTypes...> *const task,
           std::string const &name, size_t const numberThreads, bool const automaticStart) :
      TaskNodeAbstraction(name, task),
      CleanableAbstraction(static_cast<behavior::Cleanable *>(task)),
      CanTerminateAbstraction(static_cast<behavior::CanTerminate *>(task)),
      abstraction::GroupableAbstraction<AbstractTask<Separator, AllTypes...>, CoreTask<Separator, AllTypes...>>(task,
                                                                                                                numberThreads),
      TIM<Separator, AllTypes...>(task, this),
      TOM<Separator, AllTypes...>(),
      task_(task),
      automaticStart_(automaticStart) {
    if (this->numberThreads() == 0) { throw std::runtime_error("A task needs at least one thread."); }
  }

  /// @brief Construct a task from the user-defined task and its concrete abstraction's implementations
  /// @tparam ConcreteMultiReceivers Type of concrete implementation of ReceiverAbstraction for multiple types
  /// @tparam ConcreteMultiExecutes Type of concrete implementation of ExecuteAbstraction for multiple types
  /// @tparam ConcreteMultiSenders Type of concrete implementation of SenderAbstraction for multiple types
  /// @param task User-defined task
  /// @param name Task's name
  /// @param numberThreads Number of threads for the task
  /// @param automaticStart Flag for automatic start
  /// @param concreteSlot Concrete implementation of SlotAbstraction
  /// @param concreteMultiReceivers Concrete implementation of ReceiverAbstraction for multiple types
  /// @param concreteMultiExecutes Concrete implementation of ExecuteAbstraction for multiple type
  /// @param concreteNotifier Concrete implementation of NotifierAbstraction
  /// @param concreteMultiSenders Concrete implementation of SenderAbstraction for multiple types
  /// @throw std::runtime_error if the number of threads is 0
  template<class ConcreteMultiReceivers, class ConcreteMultiExecutes, class ConcreteMultiSenders>
  CoreTask(AbstractTask<Separator, AllTypes...> *const task,
           std::string const &name, size_t const numberThreads, bool const automaticStart,
           std::shared_ptr<implementor::ImplementorSlot> concreteSlot,
           std::shared_ptr<ConcreteMultiReceivers> concreteMultiReceivers,
           std::shared_ptr<ConcreteMultiExecutes> concreteMultiExecutes,
           std::shared_ptr<implementor::ImplementorNotifier> concreteNotifier,
           std::shared_ptr<ConcreteMultiSenders> concreteMultiSenders) :
      TaskNodeAbstraction(name, task),
      CleanableAbstraction(static_cast<behavior::Cleanable *>(task)),
      CanTerminateAbstraction(static_cast<behavior::CanTerminate *>(task)),
      abstraction::GroupableAbstraction<AbstractTask<Separator, AllTypes...>, CoreTask<Separator, AllTypes...>>
          (task, numberThreads),
      TIM<Separator, AllTypes...>(this, concreteSlot, concreteMultiReceivers, concreteMultiExecutes),
      TOM<Separator, AllTypes...>(concreteNotifier, concreteMultiSenders),
      task_(task),
      automaticStart_(automaticStart) {
    if (this->numberThreads() == 0) { throw std::runtime_error("A task needs at least one thread."); }
  }

  /// @brief Default destructor
  ~CoreTask() override = default;

  /// @brief Accessor to the memory manager
  /// @return The attached memory manager
  [[nodiscard]] std::shared_ptr<AbstractMemoryManager> memoryManager() const override {
    return this->task_->memoryManager();
  }

  /// @brief Call user-definable termination
  /// @param lock Flag if the call to the user-definable termination need to be protected
  /// @return True if the node can terminate, else false
  bool callCanTerminate(bool lock) override {
    bool result;
    if (lock) { this->lockSlotMutex(); }
    result = this->callNodeCanTerminate();
    if (lock) { this->unlockSlotMutex(); }
    return result;
  };

  /// @brief Initialize the task
  /// @details Call user define initialize, initialise memory manager if present
  void preRun() override {
    this->nvtxProfiler()->startRangeInitializing();
    this->task_->initialize();
    if (this->task_->memoryManager() != nullptr) {
      this->task_->memoryManager()->profiler(this->nvtxProfiler());
      this->task_->memoryManager()->deviceId(this->deviceId());
      this->task_->memoryManager()->initialize();
    }
    this->nvtxProfiler()->endRangeInitializing();
    this->setInitialized();
  }

  /// @brief Main core task logic
  /// @details
  /// - if automatic start
  ///     - call user-defined task's execute method with nullptr
  /// - while the task runs
  ///     - wait for data or termination
  ///     - if can terminate, break
  ///     - get a piece of data from the queue
  ///     - call user-defined state's execute method with data
  /// - shutdown the task
  /// - notify successors task terminated
  void run() override {
    std::chrono::time_point<std::chrono::system_clock>
        start,
        finish;

    this->isActive(true);
    this->nvtxProfiler()->initialize(this->threadId());
    this->preRun();

    if (this->automaticStart_) {
      start = std::chrono::system_clock::now();
      this->callAllExecuteWithNullptr();
      finish = std::chrono::system_clock::now();
      this->incrementExecutionDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));
    }

    // Actual computation loop
    while (!this->callCanTerminate(true)) {
      // Wait for a data to arrive or termination
      start = std::chrono::system_clock::now();
      volatile bool canTerminate = this->wait();
      finish = std::chrono::system_clock::now();
      this->incrementWaitDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));

      // If loop can terminate break the loop early
      if (canTerminate) { break; }

      // Operate the connectedReceivers to get a data and send it to execute
      start = std::chrono::system_clock::now();
      this->operateReceivers();
      finish = std::chrono::system_clock::now();
      this->incrementExecutionDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));
    }

    // Do the shutdown phase
    this->postRun();
    // Wake up node that this is linked to
    this->wakeUp();
  }

  /// @brief When a task terminates, set the task to non active, call user-defined shutdown, and disconnect the task to
  /// successor nodes
  void postRun() override {
    this->nvtxProfiler()->startRangeShuttingDown();
    this->isActive(false);
    this->task_->shutdown();
    this->notifyAllTerminated();
    this->nvtxProfiler()->endRangeShuttingDown();
  }

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

  /// @brief Create a group for this task, and connect each copies to the predecessor and successor nodes
  /// @param map  Map of nodes and groups
  /// @throw std::runtime_error it the task is ill-formed or the copy is not of the right type
  void createGroup(std::map<NodeAbstraction *, std::vector<NodeAbstraction *>> &map) override {
    abstraction::SlotAbstraction *coreCopyAsSlot;
    abstraction::NotifierAbstraction *coreCopyAsNotifier;

    for (size_t threadId = 1; threadId < this->numberThreads(); ++threadId) {
      auto taskCopy = this->callCopyAndRegisterInGroup();

      if (taskCopy == nullptr) {
        std::ostringstream oss;
        oss << "A copy for the task \"" << this->name()
            << "\" has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and "
               "return a valid object.";
        throw (std::runtime_error(oss.str()));
      }

      // Copy the memory manager
      taskCopy->connectMemoryManager(this->task_->memoryManager());

      auto taskCoreCopy = dynamic_cast<CoreTask<Separator, AllTypes...> *>(taskCopy->core().get());

      if (taskCoreCopy == nullptr) {
        std::ostringstream oss;
        oss << "A copy for the task \"" << this->name()
            << "\" does not have the same type of cores than the original task.";
        throw (std::runtime_error(oss.str()));
      }

      // Deal with the group registration in the graph
      map.at(static_cast<NodeAbstraction *>(this)).push_back(taskCoreCopy);

      // Copy inner structures
      taskCoreCopy->copyInnerStructure(this);

      // Make necessary connections
      coreCopyAsSlot = static_cast<abstraction::SlotAbstraction *>(taskCoreCopy);
      coreCopyAsNotifier = static_cast<abstraction::NotifierAbstraction *>(taskCoreCopy);

      for (auto predecessorNotifier : static_cast<abstraction::SlotAbstraction *>(this)->connectedNotifiers()) {
        for (auto notifier : predecessorNotifier->notifiers()) {
          for (auto slot : coreCopyAsSlot->slots()) {
            slot->addNotifier(notifier);
            notifier->addSlot(slot);
          }
        }
      }

      for (auto successorSlot : static_cast<abstraction::NotifierAbstraction *>(this)->connectedSlots()) {
        for (auto slot : successorSlot->slots()) {
          for (auto notifier : coreCopyAsNotifier->notifiers()) {
            slot->addNotifier(notifier);
            notifier->addSlot(slot);
          }
        }
      }
    }
  }

  /// @brief Test if a memory manager is attached
  /// @return True if there is a memory manager attached, else false
  [[nodiscard]] bool hasMemoryManagerAttached() const override { return this->memoryManager() != nullptr; }

  /// @brief Accessor to user-defined extra information for the task
  /// @return User-defined extra information for the task
  [[nodiscard]] std::string extraPrintingInformation() const override {
    return this->task_->extraPrintingInformation();
  }

  /// @brief Copy task's inner structure
  /// @param copyableCore Task to copy from
  void copyInnerStructure(CoreTask<Separator, AllTypes...> *copyableCore) override {
    TIM<Separator, AllTypes...>::copyInnerStructure(copyableCore);
    TOM<Separator, AllTypes...>::copyInnerStructure(copyableCore);
  }

  /// @brief Node ids [nodeId, nodeGroupId] accessor
  /// @return  Node ids [nodeId, nodeGroupId]
  [[nodiscard]] std::vector<std::pair<std::string const, std::string const>> ids() const override {
    return {{this->id(), this->groupRepresentativeId()}};
  }
  /// @brief Visit the task
  /// @param printer Printer gathering task information
  void visit(Printer *printer) override {
    if (printer->registerNode(this)) {
      printer->printNodeInformation(this);
      TIM<Separator, AllTypes...>::printEdgesInformation(printer);
    }
  }

  /// @brief Clone method, to duplicate a task when it is part of another graph in an execution pipeline
  /// @param correspondenceMap Correspondence map of belonging graph's node
  /// @return Clone of this task
  std::shared_ptr<abstraction::NodeAbstraction> clone(
      [[maybe_unused]] std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &correspondenceMap) override {
    auto clone = std::dynamic_pointer_cast<AbstractTask<Separator, AllTypes...>>(this->callCopy());
    if (this->hasMemoryManagerAttached()) { clone->connectMemoryManager(this->memoryManager()->copy()); }
    return clone->core();
  }

  /// @brief Duplicate the task edge
  /// @param mapping Correspondence map of belonging graph's node
  void duplicateEdge(std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping) override {
    this->duplicateOutputEdges(mapping);
  }

};
}
}
#endif //HEDGEHOG_CORE_TASK_H
