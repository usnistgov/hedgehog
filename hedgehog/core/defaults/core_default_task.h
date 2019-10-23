//
// Created by anb22 on 6/10/19.
//

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
    this->nvtxProfiler()->startRangeExecuting();
    static_cast<behavior::Execute<TaskInput> *>(this->task())->execute(data);
    this->nvtxProfiler()->endRangeExecuting();
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
    this->nvtxProfiler()->startRangeInitializing();

    HLOG_SELF(0, "Initialize Memory manager for the task " << this->name() << " / " << this->id())
    // Call User-defined initialize
    this->task()->initialize();

    // Define the memory manager
    if (this->task()->memoryManager() != nullptr) {
      this->task()->memoryManager()->profiler(this->nvtxProfiler());
      this->task()->memoryManager()->deviceId(this->deviceId());
      this->task()->memoryManager()->initialize();
    }

    this->nvtxProfiler()->endRangeInitializing();
  }

  /// @brief Defines what a CoreDefaultTask does after the execute loop
  void postRun() override {
    this->isActive(false);
    this->nvtxProfiler()->startRangeShuttingDown();
    // Call User-defined shutdown
    this->task()->shutdown();
    this->nvtxProfiler()->endRangeShuttingDown();
    // Notify all linked node, the node (this) is terminated
    this->notifyAllTerminated();
  }
};

}
#endif //HEDGEHOG_CORE_DEFAULT_TASK_H
