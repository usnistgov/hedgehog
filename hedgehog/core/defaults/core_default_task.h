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
    static_cast<Execute<TaskInput> *>(this->task())->execute(data);
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
  ...
  {}

  virtual ~CoreDefaultTask() = default;

//  CoreDefaultTask(CoreDefaultTask<TaskOutput, TaskInputs...> const &  rhs ) :
//  CoreNode(rhs.name(), rhs.type(), rhs.numberThreads()),
//  CoreNotifier(rhs.name(), rhs.type(), rhs.numberThreads()),
//  CoreQueueNotifier(rhs.name(), rhs.type(), rhs.numberThreads()),
//  CoreQueueSender<TaskOutput>(rhs.name(), rhs.type(), rhs.numberThreads()),
//  CoreSlot(rhs.name(), rhs.type(), rhs.numberThreads()),
//  CoreReceiver<TaskInputs>(rhs.name(), rhs.type(), rhs.numberThreads())...,
//  CoreTask<TaskOutput, TaskInputs...>(rhs.name(), rhs.numberThreads(), rhs.type(), rhs.task(), rhs.automaticStart()),
//    DefaultCoreTaskExecute<TaskInputs, TaskOutput, TaskInputs...>(rhs.name(), rhs.numberThreads(), rhs.type(), rhs.task(), rhs.automaticStart())...{
//    HLOG_SELF(0,
//              "Copy construction information from " << rhs.name() << "(" << rhs.id() << ") << and task " << rhs.name() << "("
//                                            << rhs.id() << ")")
//
//
//    this->isInside(true);
//    if(rhs.isInCluster()){this->setInCluster();}
//    this->numberThreads(rhs.numberThreads());
//
//  }

  std::shared_ptr<CoreNode> clone() override {
    return this->createCopyFromThis()->core();
//    return std::make_shared<CoreDefaultTask<TaskOutput, TaskInputs...>>(*this);
  }

  void preRun() override { this->task()->initialize(); }

  void postRun() override {
    this->isActive(false);
    this->task()->shutdown();
    this->notifyAllTerminated();
  }
};

#endif //HEDGEHOG_CORE_DEFAULT_TASK_H
