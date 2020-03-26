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

#include <variant>
#include <string_view>

#include "../../tools/traits.h"
#include "../io/queue/receiver/core_queue_multi_receivers.h"
#include "../io/queue/sender/core_queue_sender.h"

#include "../../behavior/execute.h"

#include "../../tools/logger.h"
#include "../../tools/nvtx_profiler.h"
#include "../behavior/core_execute.h"

#include "../../api/memory_manager/abstract_memory_manager.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief AbstractTask forward declaration
/// @tparam TaskOutput Task output type
/// @tparam TaskInputs Task input types
template<class TaskOutput, class ...TaskInputs>
class AbstractTask;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog core namespace
namespace core {

/// @brief Core of the task node
/// @tparam TaskOutput Task output type
/// @tparam TaskInputs Task input types
template<class TaskOutput, class ...TaskInputs>
class CoreTask
    : public virtual CoreQueueSender<TaskOutput>,
      public CoreQueueMultiReceivers<TaskInputs...>,
      public virtual CoreExecute<TaskInputs> ... {
 private:
  AbstractTask<TaskOutput, TaskInputs...> *task_ = nullptr; ///< Task node pointer (just for reference)
  bool automaticStart_ = false; ///< Automatic start property
  std::vector<std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>>>
      clusterAbstractTask_ = {}; ///< Store all AbstractTask clones (hold memory)

  std::shared_ptr<NvtxProfiler> nvtxProfiler_ = nullptr; ///< Store a nvtx profiler for the task

 public:
  /// @brief CoreTask constructor (Used for AbstractTask and StateManager)
  /// @param name Task name
  /// @param numberThreads Task number of threads
  /// @param type Task type (the core is also used for the state manager)
  /// @param task Task node
  /// @param automaticStart Automatic start property
  CoreTask(std::string_view const &name,
           size_t const numberThreads,
           NodeType const type,
           AbstractTask<TaskOutput, TaskInputs...> *task,
           bool automaticStart) : CoreNode(name, type, numberThreads),
                                  CoreNotifier(name, type, numberThreads),
                                  CoreQueueSender<TaskOutput>(name, type, numberThreads),
                                  CoreSlot(name, type, numberThreads),
                                  CoreReceiver<TaskInputs>(name, type, numberThreads)...,
  CoreQueueMultiReceivers<TaskInputs...>(name, type, numberThreads), task_(task),
  automaticStart_(automaticStart) {
    HLOG_SELF(0, "Creating CoreTask with task: " << task << " type: " << (int) type << " and name: " << name)
    nvtxProfiler_ = std::make_shared<NvtxProfiler>(this->name());
  }

  /// @brief Core Task destructor
  ~CoreTask() override {
    HLOG_SELF(0, "Destructing CoreTask")
    task_ = nullptr;
  }

  /// @brief Automatic start property accessor
  /// @return Automatic start property
  [[nodiscard]] bool automaticStart() const { return automaticStart_; }

  /// @brief NVTX profiler accessor
  /// @return NVTX profiler
  std::shared_ptr<NvtxProfiler> &nvtxProfiler() { return nvtxProfiler_; }

  /// @brief Node accessor
  /// @return node as Node
  behavior::Node *node() override { return task_; }

  /// @brief  Node accessor
  /// @return node as AbstractTask
  AbstractTask<TaskOutput, TaskInputs...> *task() const { return task_; }

  /// @brief Automatic start property accessor
  /// @param automaticStart Automatic start property
  void automaticStart(bool automaticStart) { automaticStart_ = automaticStart; }

  /// @brief Special visit method for a CoreTask
  /// @param printer Printer visitor to print the CoreTask
  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
      // Print the CoreQueueSender informations, i.e. the edges
      CoreQueueSender<TaskOutput>::visit(printer);
    }
  }

  /// @brief Copy the inner structure from rhs to this CoreTask
  /// @param rhs CoreTask to copy
  void copyInnerStructure(CoreTask<TaskOutput, TaskInputs...> *rhs) {
    HLOG_SELF(0, "Copy cluster CoreTask information from " << rhs->name() << "(" << rhs->id() << ")")
    CoreQueueMultiReceivers < TaskInputs...>::copyInnerStructure(rhs);
    CoreQueueSender<TaskOutput>::copyInnerStructure(rhs);
    CoreNode::copyInnerStructure(rhs);
  }

  /// @brief Create a cluster for a CoreTask
  /// @param insideNodesGraph CoreTask owner structure to store the copy of the task
  void createCluster(std::shared_ptr<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>> &insideNodesGraph) override {
    auto mm = this->task()->memoryManager();
    // Set cluster property
    if (this->numberThreads() > 1) { this->setInCluster(); }

    // For each copy
    for (size_t threadId = 1; threadId < this->numberThreads(); threadId++) {
      // Copy the task
      auto taskCopy = createCopyFromThis();
      // Duplicate the memory manager, each of them will have a separate instance
      connectMemoryManager(mm, taskCopy);
      // Set property
      auto coreCopy = std::dynamic_pointer_cast<CoreTask<TaskOutput, TaskInputs...>>(taskCopy->core());
      coreCopy->threadId(threadId);
      coreCopy->coreClusterNode(this);
      coreCopy->copyInnerStructure(this);
      coreCopy->setInCluster();
      insideNodesGraph->insert({this, coreCopy});
    }
  }

  /// @brief Main loop for the CoreTask
  void run() override {
    HLOG_SELF(2, "Run")
    std::chrono::time_point<std::chrono::high_resolution_clock>
        start,
        finish;

    this->isActive(true);
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->initialize(this->threadId());
#endif
    // Do the initialization phase
    this->preRun();

    // If automatic start is enable send nullptr to all input nodes and wake them up
    if (this->automaticStart()) {
#ifndef HH_DISABLE_PROFILE
      start = std::chrono::high_resolution_clock::now();
#endif
      (static_cast<CoreExecute<TaskInputs> *>(this)->callExecute(nullptr), ...);
#ifndef HH_DISABLE_PROFILE
      finish = std::chrono::high_resolution_clock::now();
      this->incrementExecutionDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
#endif
    }

    // Actual computation loop
    while (!this->callCanTerminate(true)) {
#ifndef HH_DISABLE_PROFILE
      start = std::chrono::high_resolution_clock::now();
#endif
      // Wait for a data to arrive or termination
      volatile bool canTerminate = this->waitForNotification();
#ifndef HH_DISABLE_PROFILE
      finish = std::chrono::high_resolution_clock::now();
      this->incrementWaitDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
#endif
      // If can terminate break the loop early
      if (canTerminate) { break; }
#ifndef HH_DISABLE_PROFILE
      start = std::chrono::high_resolution_clock::now();
#endif
      // Operate the receivers to get a data and send it to execute
      (this->operateReceiver<TaskInputs>(), ...);
#ifndef HH_DISABLE_PROFILE
      finish = std::chrono::high_resolution_clock::now();
      this->incrementExecutionDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
#endif
    }

    // Do the shutdown phase
    this->postRun();
    // Wake up node that this is linked to
    this->wakeUp();
  }

  /// @brief Call user-defined or default canTerminate and lock/unlock the mutex if lock is true
  /// @param lock Flag to lock the mutex
  /// @return True if the task can terminate, else False
  virtual bool callCanTerminate(bool lock) {
    bool result;
    if (lock) { this->lockUniqueMutex(); }
    result = this->node()->canTerminate();
    HLOG_SELF(2, "callCanTerminate: " << std::boolalpha << result)
    if (lock) { this->unlockUniqueMutex(); }
    return result;
  };

  /// @brief Operate a CoreTasks's receiver for a specific type, thread safe
  /// @tparam Input Receiver Input
  template<class Input>
  void operateReceiver() {
    HLOG_SELF(2, "Operate receivers")
    // Lock the mutex
    this->lockUniqueMutex();
    // Get the receiver with the right type
    auto receiver = static_cast<CoreQueueReceiver<Input> *>(this);
    // If receiver's queue not empty
    if (!receiver->receiverEmpty()) {
      // Get the data
      std::shared_ptr<Input> data = receiver->popFront();
      this->unlockUniqueMutex();
      // Call execute
      static_cast<CoreExecute<Input> *>(this)->callExecute(data);
    } else {
      // Unlock the mutex
      this->unlockUniqueMutex();
    }
  }

  /// @brief Wait method for notification
  /// @details By default wait if the receivers queues are empty and the node have input nodes links
  /// @return True if the node can terminate, else False
  bool waitForNotification() override {
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->startRangeWaiting(this->totalQueueSize());
#endif
    std::unique_lock<std::mutex> lock(*(this->slotMutex()));
    HLOG_SELF(2, "Wait for notification")
    this->notifyConditionVariable()->wait(lock,
                                          [this]() {
                                            bool receiversEmpty = this->receiversEmpty();
                                            bool callCanTerminate = this->callCanTerminate(false);
                                            HLOG_SELF(2,
                                                      "Check for notification: " << std::boolalpha
                                                          << (bool) (!receiversEmpty) << "||"
                                                          << std::boolalpha
                                                          << (bool) callCanTerminate)
                                            return !receiversEmpty || callCanTerminate;
                                          });
    HLOG_SELF(2, "Notification received")
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->endRangeWaiting();
#endif
    return callCanTerminate(false);
  }

  /// @brief Create a copy from this instance
  /// @return A copy of this instance
  std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> createCopyFromThis() {
    HLOG_SELF(0, "Copy Whole Task")

    std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> sharedAbstractTaskCopy = this->task()->copy();
    // The copy method has not been redefined and return nullptr by default
    if (!sharedAbstractTaskCopy) {
      std::ostringstream oss;
      oss << "A copy for the task \"" << this->name()
          << "\" has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and "
             "return a valid object.";
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }

    // Add the copy into the cluster
    this->clusterAbstractTask_.push_back(sharedAbstractTaskCopy);

    // Get the core from the copy
    auto coreCopy = std::dynamic_pointer_cast<CoreTask<TaskOutput, TaskInputs...>>(sharedAbstractTaskCopy->core());

    if (coreCopy == nullptr) {
      std::ostringstream oss;
      oss << "A copy for the task " << this->name()
          << " has been invoked but the AbstractTask constructor has not been called. To fix this error, call the "
             "AbstractTask constructor in your specialized task constructor.";
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }

    // Copy the property
    coreCopy->automaticStart_ = this->automaticStart();
    coreCopy->threadId(this->threadId());
    coreCopy->isInside(true);
    if (this->isInCluster()) { coreCopy->setInCluster(); }
    coreCopy->numberThreads(this->numberThreads());

    // Copy the memory manager
    if (this->task()->memoryManager()) {
      auto copyMemoryManager = this->task()->memoryManager()->copy();
      if (!copyMemoryManager) {
        std::ostringstream oss;
        oss << "An execution pipeline has been created with a graph that hold a task named \""
            << this->name()
            << "\" connected to a memory manager. Or the memory manager does not have a compliant copy method. "
               "Please implement it.";
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
      HLOG_SELF(0, "Copy the MM " << this->task()->memoryManager() << " to: " << copyMemoryManager
          << " and set it to the task: " << coreCopy->name() << " / " << coreCopy->id())
      connectMemoryManager(copyMemoryManager, sharedAbstractTaskCopy);
    }
    return sharedAbstractTaskCopy;
  }

 private:
  /// @brief Connect a memory manager to a task if the type is valid (HedgehogTraits::is_managed_memory_v<TaskOutput>),
  /// and the memory manager defined
  /// @param mm Memory manager to connect
  /// @param taskCopy Task to connect the memory manager
  void connectMemoryManager(std::shared_ptr<AbstractMemoryManager<TaskOutput>> const &mm,
                            std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> taskCopy) {
    if constexpr (traits::is_managed_memory_v<TaskOutput>) {
      if (mm) {
        HLOG_SELF(1, "Copy the MM pointer " << mm
                                            << " from: " << this->name() << " / " << this->id()
                                            << " to: " << taskCopy->core()->name() << " / " << taskCopy->core()->id())
        taskCopy->connectMemoryManager(mm);
      }
    }
  }

};

}
}
#endif //HEDGEHOG_CORE_TASK_H
