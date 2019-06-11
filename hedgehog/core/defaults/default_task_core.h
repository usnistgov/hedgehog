//
// Created by anb22 on 6/10/19.
//

#ifndef HEDGEHOG_DEFAULT_TASK_CORE_H
#define HEDGEHOG_DEFAULT_TASK_CORE_H

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
    static_cast<Execute<TaskInput> *>(this->task())->execute(data);
  }
};
#if defined (__clang__)
#pragma clang diagnostic pop
#endif //__clang//

template<class TaskOutput, class ...TaskInputs>
class DefaultTaskCore
    : public DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...> ... {
 public:
  using DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...>::callExecute...;

  DefaultTaskCore(std::string_view const &name,
                  size_t const numberThreads,
                  NodeType const type,
                  AbstractTask<TaskOutput, TaskInputs...> *task,
                  bool automaticStart)
      : CoreNode(name, type, numberThreads),
        CoreNotifier(name, type, numberThreads),
        CoreQueueSender<TaskOutput>(name, type, numberThreads),
        CoreSlot(name, type, numberThreads),
        CoreReceiver<TaskInputs>(name, type, numberThreads)...,
      CoreTask<TaskOutput, TaskInputs...>(name,
                                          numberThreads,
                                          type,
                                          task,
                                          automaticStart),
      DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...>(name, numberThreads, type, task, automaticStart)
  ...{}

  void postRun() override {
    this->task()->shutdown();
    this->notifyAllTerminated();
  }

  void preRun() override {
    this->task()->initialize();
  }
};

#endif //HEDGEHOG_DEFAULT_TASK_CORE_H
