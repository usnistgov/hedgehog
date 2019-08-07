//
// Created by anb22 on 6/10/19.
//

#ifndef HEDGEHOG_CORE_DEFAULT_TASK_H
#define HEDGEHOG_CORE_DEFAULT_TASK_H

#include "../node/core_task.h"

// Have to add -Woverloaded-virtual for clang because execute hides overloaded virtual function
#if defined (__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif //__clang//
template<class TaskInput, class TaskOutput, class ...TaskInputs>
class DefaultCoreTaskExecute : public virtual CoreTask<TaskOutput, TaskInputs...> {
 protected:
 public:
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

  void callExecute(std::shared_ptr<TaskInput> data) final {
    HLOG_SELF(2, "Call execute")
    this->nvtxProfiler()->startRangeExecuting();
    static_cast<Execute<TaskInput> *>(this->task())->execute(data);
    this->nvtxProfiler()->endRangeExecuting();
  }
};
#if defined (__clang__)
#pragma clang diagnostic pop
#endif //__clang//

template<class TaskOutput, class ...TaskInputs>
class CoreDefaultTask
    : public DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...> ... {
 public:
  using DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...>::callExecute...;

  CoreDefaultTask(std::string_view const &name,
                  size_t const numberThreads,
                  NodeType const type,
                  AbstractTask<TaskOutput, TaskInputs...> *task,
                  bool automaticStart)
      :
      CoreNode(name, type, numberThreads),
      CoreNotifier(name, type, numberThreads),
      CoreQueueNotifier(name, type, numberThreads),
      CoreQueueSender<TaskOutput>(name, type, numberThreads),
      CoreSlot(name, type, numberThreads),
      CoreReceiver<TaskInputs>(name, type, numberThreads)...,
      CoreTask<TaskOutput, TaskInputs...>(name, numberThreads, type, task, automaticStart),
      DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...>(name, numberThreads, type, task, automaticStart)
  ...{}

  virtual ~CoreDefaultTask() = default;

  std::shared_ptr<CoreNode> clone() override {
    return this->createCopyFromThis()->core();
  }

  void preRun() override {
    this->nvtxProfiler()->startRangeInitializing();

    HLOG_SELF(0, "Initialize Memory manager for the task " << this->name() << " / " << this->id())
    this->task()->initialize();

    if (this->task()->memoryManager() != nullptr) {
      this->task()->memoryManager()->profiler(this->nvtxProfiler());
      this->task()->memoryManager()->deviceId(this->deviceId());
      this->task()->memoryManager()->initialize();
    }

    this->nvtxProfiler()->endRangeInitializing();
  }

  void postRun() override {
    this->isActive(false);
    this->nvtxProfiler()->startRangeShuttingDown();
    this->task()->shutdown();
    this->nvtxProfiler()->endRangeShuttingDown();
    this->notifyAllTerminated();
  }
};

#endif //HEDGEHOG_CORE_DEFAULT_TASK_H
