//
// Created by 775backup on 2019-04-08.
//

#ifndef HEDGEHOG_CORE_TASK_H
#define HEDGEHOG_CORE_TASK_H

#include <variant>

#include "../../tools/traits.h"
#include "../io/task/receiver/core_task_multi_receivers.h"
#include "../io/task/sender/core_task_sender.h"

#include "../../behaviour/execute.h"

#include "../../tools/logger.h"

template<class TaskOutput, class ...TaskInputs>
class AbstractTask;

template<class TaskOutput, class ...TaskInputs>
class TaskCore : public CoreTaskSender<TaskOutput>, public CoreTaskMultiReceivers<TaskInputs...> {
 private:
  AbstractTask<TaskOutput, TaskInputs...> *task_ = nullptr;
  bool automaticStart_ = false;
 public:
  TaskCore(std::string_view const &name,
           size_t const numberThreads,
           NodeType const type,
           AbstractTask<TaskOutput, TaskInputs...> *task,
           bool automaticStart) : CoreNode(name, type, numberThreads),
                                  CoreNotifier(name, type, numberThreads),
                                  CoreSlot(name, type, numberThreads),
                                  CoreReceiver<TaskInputs>(name, type, numberThreads)...,
  CoreTaskSender<TaskOutput>(name, type, numberThreads),
  CoreTaskMultiReceivers<TaskInputs...>(name, type, numberThreads), task_(task),
  automaticStart_(automaticStart) {
    HLOG_SELF(0, "Creating TaskCore with task: " << task << " type: " << (int) type << " and name: " << name)
  }

  TaskCore(TaskCore<TaskOutput, TaskInputs...> *const rhs, AbstractTask<TaskOutput, TaskInputs...> *task) :
      CoreNode(rhs->name(), rhs->type(), rhs->numberThreads()),
      CoreNotifier(rhs->name(), rhs->type(), rhs->numberThreads()),
      CoreSlot(rhs->name(), rhs->type(), rhs->numberThreads()),
      CoreReceiver<TaskInputs>(rhs->name(), rhs->type(), rhs->numberThreads())...,
      CoreTaskSender<TaskOutput>(rhs
  ->
  name(), rhs
  ->
  type(), rhs
  ->
  numberThreads()
  ),
  CoreTaskMultiReceivers<TaskInputs...>(rhs
  ->
  name(), rhs
  ->
  type(), rhs
  ->
  numberThreads()
  ),
  task_(task), automaticStart_(rhs
  ->automaticStart_){
    HLOG_SELF(0,
              "Duplicate information from " << rhs->name() << "(" << rhs->id() << ") << and task " << rhs->name() << "("
                                            << rhs->id() << ")")
  }

  ~TaskCore() override {
    HLOG_SELF(0, "Destructing TaskCore")
    task_ = nullptr;
  }

  bool automaticStart() const { return automaticStart_; }

  virtual Node *getNode() override {
    return task_;
  }

  AbstractTask<TaskOutput, TaskInputs...> *task() const {
    return task_;
  }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
      CoreTaskSender < TaskOutput > ::visit(printer);
    }
  }

  void copyInnerStructure(TaskCore<TaskOutput, TaskInputs...> *rhs) {
    HLOG_SELF(0, "Duplicate CoreTask information from " << rhs->name() << "(" << rhs->id() << ")")
    CoreTaskMultiReceivers < TaskInputs...>::copyInnerStructure(rhs);
    CoreTaskSender < TaskOutput > ::copyInnerStructure(rhs);
    CoreNode::copyInnerStructure(rhs);
  }

  void copyWholeNode(std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>> &insideNodesGraph) override {
    auto name = this->name();
    if (this->numberThreads() > 1) { this->setInCluster(); }
    for (size_t threadId = 1; threadId < this->numberThreads(); threadId++) {
      HLOG_SELF(0, "Copy Whole Task")
      std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> sharedAbstractTaskCopy = this->task()->copy();

      if (sharedAbstractTaskCopy == nullptr) {
        HLOG_SELF(0,
                  "A copy for the task " << name
                                         << " has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and return a valid object.")
        std::cerr << "A copy for the task " << name
                  << " has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and return a valid object."
                  << std::endl;
        exit(42);
      }

      auto coreCopy = dynamic_cast<TaskCore<TaskOutput, TaskInputs...> *>(sharedAbstractTaskCopy->getCore());

      if (coreCopy == nullptr) {
        HLOG_SELF(0,
                  "A copy for the task " << name
                                         << " has been invoked but the AbstractTask constructor has not been called. To fix this error, call the  AbstractTask constructor in your specialized task constructor.")
        std::cerr << "A copy for the task " << name
                  << " has been invoked but the AbstractTask constructor has not been called. To fix this error, call the  AbstractTask constructor in your specialized task constructor."
                  << std::endl;
        exit(42);
      }

      coreCopy->automaticStart_ = this->automaticStart();
      coreCopy->threadId(threadId);
      coreCopy->copyInnerStructure(this);
      coreCopy->setInCluster();
      coreCopy->clusterId(this->clusterId());

      insideNodesGraph->insert({this->id(), std::static_pointer_cast<Node>(sharedAbstractTaskCopy)});
    }
  }

  void run() override {
    std::chrono::time_point<std::chrono::high_resolution_clock>
        start,
        finish;

    HLOG_SELF(2, "Run")

    this->task()->initialize();

    if (this->automaticStart()) {
      start = std::chrono::high_resolution_clock::now();
      (this->executionCall<TaskInputs>(nullptr), ...);
      finish = std::chrono::high_resolution_clock::now();
      this->incrementExecutionDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
    }

    while (!this->callCanTerminate(true)) {

      start = std::chrono::high_resolution_clock::now();
      this->waitForNotification();
      finish = std::chrono::high_resolution_clock::now();
      this->incrementWaitDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));

      start = std::chrono::high_resolution_clock::now();
      (this->operateReceivers<TaskInputs>(), ...);
      finish = std::chrono::high_resolution_clock::now();
      this->incrementExecutionDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
    }

    this->task_->shutdown();
    this->notifyAllTerminated();
    this->wakeUp();
  }

  bool callCanTerminate(bool lock) {
    if (lock) {
      this->lockUniqueMutex();
    }
    bool result = this->task()->canTerminate() && this->receiversEmpty();
    HLOG_SELF(2, "callCanTerminate: " << std::boolalpha << result)
    if (lock) {
      this->unlockUniqueMutex();
    }
    return result;
  };

  template<class Input>
  void executionCall(std::shared_ptr<Input> data) {
    HLOG_SELF(2, "Call execute")
    static_cast<Execute<Input> *>(this->task())->execute(data);
  }

  template<class Input>
  void operateReceivers() {
    HLOG_SELF(2, "Operate receivers")
    this->lockUniqueMutex();
    auto receiver = static_cast<CoreTaskReceiver<Input> *>(this);
    if (!receiver->receiverEmpty()) {
      std::shared_ptr<Input> data = receiver->popFront();
      this->unlockUniqueMutex();
      this->executionCall<Input>(data);
    } else {
      this->unlockUniqueMutex();
    }
  }

  void waitForNotification() override {
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

  }

};

#endif //HEDGEHOG_CORE_TASK_H
