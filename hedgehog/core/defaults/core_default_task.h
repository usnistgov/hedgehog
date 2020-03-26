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


#ifndef HEDGEHOG_CORE_DEFAULT_TASK_H
#define HEDGEHOG_CORE_DEFAULT_TASK_H

#include "../node/core_task.h"

/// @brief Hedgehog core namespace
namespace hh::core {

// Have to add -Woverloaded-virtual for clang because execute hides overloaded virtual function
#if defined (__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif //__clang//
/// @brief Middle class used to propose a default definition of CoreExecute::callExecute for CoreDefaultTask
/// @tparam TaskInput Type of data to send to CoreExecute::callExecute
/// @tparam TaskOutput Task's output type
/// @tparam TaskInputs Task's inputs types
template<class TaskInput, class TaskOutput, class ...TaskInputs>
class DefaultCoreTaskExecute : public virtual CoreTask<TaskOutput, TaskInputs...> {
 protected:
 public:
  /// @brief DefaultCoreTaskExecute constructor
  /// @param name Task name
  /// @param numberThreads Task's number of threads
  /// @param type Node Type
  /// @param task User-defined task
  /// @param automaticStart Automatic start task's property
  DefaultCoreTaskExecute(std::string_view const &name,
                         size_t const numberThreads,
                         NodeType const type,
                         AbstractTask<TaskOutput, TaskInputs...> *task,
                         bool automaticStart) : CoreNode(name, type, numberThreads),
                                                CoreNotifier(name, type, numberThreads),
                                                CoreQueueSender<TaskOutput>(name, type, numberThreads),
                                                CoreSlot(name, type, numberThreads),
                                                CoreReceiver<TaskInputs>(name, type, numberThreads)...,
  CoreTask<TaskOutput, TaskInputs...>(name,
                                      numberThreads,
                                      type,
                                      task,
                                      automaticStart) {}

  /// @brief Definition of CoreExecute::callExecute for DefaultCoreTaskExecute
  /// @param data Data send to the task
  void callExecute(std::shared_ptr<TaskInput> data) final {
    HLOG_SELF(2, "Call execute")
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->startRangeExecuting();
#endif
    static_cast<behavior::Execute<TaskInput> *>(this->task())->execute(data);
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->endRangeExecuting();
#endif
  }
};
#if defined (__clang__)
#pragma clang diagnostic pop
#endif //__clang//

/// @brief Core of the default Task to be use
/// @tparam TaskOutput Task output type
/// @tparam TaskInputs Task inputs type
template<class TaskOutput, class ...TaskInputs>
class CoreDefaultTask
    : public DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...> ... {
 public:
  using DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...>::callExecute...;
  /// @brief CoreDefaultTask constructor
  /// @param name Task name
  /// @param numberThreads Task's number of threads
  /// @param type Node Type
  /// @param task User-defined task
  /// @param automaticStart Automatic start task's property
  CoreDefaultTask(std::string_view const &name,
                  size_t const numberThreads,
                  NodeType const type,
                  AbstractTask<TaskOutput, TaskInputs...> *task,
                  bool automaticStart) :
      CoreNode(name, type, numberThreads),
      CoreNotifier(name, type, numberThreads),
      CoreQueueNotifier(name, type, numberThreads),
      CoreQueueSender<TaskOutput>(name, type, numberThreads),
      CoreSlot(name, type, numberThreads),
      CoreReceiver<TaskInputs>(name, type, numberThreads)...,
      CoreTask<TaskOutput, TaskInputs...>(name, numberThreads, type, task, automaticStart),
      DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...>(name, numberThreads, type, task, automaticStart)...{
  }

  /// @brief CoreDefaultTask default destructor
  virtual ~CoreDefaultTask() = default;

  /// @brief Clone overload for CoreDefaultTask
  /// @return The clone of CoreDefaultTask (this)
  std::shared_ptr<CoreNode> clone() override {
    return this->createCopyFromThis()->core();
  }

  /// @brief Defines what a CoreDefaultTask does before the execute loop
  void preRun() override {
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->startRangeInitializing();
#endif
    HLOG_SELF(0, "Initialize Memory manager for the task " << this->name() << " / " << this->id())
    // Call User-defined initialize
    this->task()->initialize();

    // Define the memory manager
    if (this->task()->memoryManager() != nullptr) {
      this->task()->memoryManager()->profiler(this->nvtxProfiler());
      this->task()->memoryManager()->deviceId(this->deviceId());
      this->task()->memoryManager()->initialize();
    }
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->endRangeInitializing();
#endif
  }

  /// @brief Defines what a CoreDefaultTask does after the execute loop
  void postRun() override {
    this->isActive(false);
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->startRangeShuttingDown();
#endif
    // Call User-defined shutdown
    this->task()->shutdown();
#ifndef HH_DISABLE_NVTX_PROFILE
    this->nvtxProfiler()->endRangeShuttingDown();
#endif
    // Notify all linked node, the node (this) is terminated
    this->notifyAllTerminated();
  }
};

}
#endif //HEDGEHOG_CORE_DEFAULT_TASK_H
