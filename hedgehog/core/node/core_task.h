//
// Created by 775backup on 2019-04-08.
//

#ifndef HEDGEHOG_CORE_TASK_H
#define HEDGEHOG_CORE_TASK_H

#include <variant>

#include "../../tools/traits.h"
#include "../io/queue/receiver/core_queue_multi_receivers.h"
#include "../io/queue/sender/core_queue_sender.h"

#include "../../behaviour/execute.h"

#include "../../tools/logger.h"
#include "../behaviour/core_execute.h"

template<class TaskOutput, class ...TaskInputs>
class AbstractTask;

template<class TaskOutput, class ...TaskInputs>
class CoreTask
    : public virtual CoreQueueSender<TaskOutput>,
      public CoreQueueMultiReceivers<TaskInputs...>,
      public virtual CoreExecute<TaskInputs> ... {
 private:
  AbstractTask<TaskOutput, TaskInputs...> *task_ = nullptr;
  bool automaticStart_ = false;
 public:
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
  }

  CoreTask(CoreTask<TaskOutput, TaskInputs...> *const rhs, AbstractTask<TaskOutput, TaskInputs...> *task) :
      CoreNode(rhs->name(), rhs->type(), rhs->numberThreads()),
      CoreNotifier(rhs->name(), rhs->type(), rhs->numberThreads()),
      CoreSlot(rhs->name(), rhs->type(), rhs->numberThreads()),
      CoreReceiver<TaskInputs>(rhs->name(), rhs->type(), rhs->numberThreads())...,
      CoreQueueSender<TaskOutput>(rhs
  ->
  name(), rhs
  ->
  type(), rhs
  ->
  numberThreads()
  ),
  CoreQueueMultiReceivers<TaskInputs...>(rhs
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

  ~CoreTask() override {
    HLOG_SELF(0, "Destructing CoreTask")
    task_ = nullptr;
  }

  bool automaticStart() const { return automaticStart_; }

  Node *node() override {
    return task_;
  }

  AbstractTask<TaskOutput, TaskInputs...> *task() const {
    return task_;
  }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
      CoreQueueSender < TaskOutput > ::visit(printer);
    }
  }

  void copyInnerStructure(CoreTask<TaskOutput, TaskInputs...> *rhs) {
    HLOG_SELF(0, "Duplicate CoreTask information from " << rhs->name() << "(" << rhs->id() << ")")
    CoreQueueMultiReceivers < TaskInputs...>::copyInnerStructure(rhs);
    CoreQueueSender < TaskOutput > ::copyInnerStructure(rhs);
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
                  "A copy for the queue " << name
                                          << " has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and return a valid object.")
        std::cerr << "A copy for the queue " << name
                  << " has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and return a valid object."
                  << std::endl;
        exit(42);
      }

      auto coreCopy = dynamic_cast<CoreTask<TaskOutput, TaskInputs...> *>(sharedAbstractTaskCopy->core());

      if (coreCopy == nullptr) {
        HLOG_SELF(0,
                  "A copy for the queue " << name
                                          << " has been invoked but the AbstractTask constructor has not been called. To fix this error, call the  AbstractTask constructor in your specialized queue constructor.")
        std::cerr << "A copy for the queue " << name
                  << " has been invoked but the AbstractTask constructor has not been called. To fix this error, call the  AbstractTask constructor in your specialized queue constructor."
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

    this->preRun();
    if (this->automaticStart()) {
      start = std::chrono::high_resolution_clock::now();
      (static_cast<CoreExecute<TaskInputs> *>(this)->callExecute(nullptr), ...);
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
    this->postRun();

    this->wakeUp();
  }
 public:

  bool callCanTerminate(bool lock) {
    if (lock) {
      this->lockUniqueMutex();
    }
    bool result = this->node()->canTerminate() && this->receiversEmpty();
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
    auto receiver = static_cast<CoreQueueReceiver<Input> *>(this);
    if (!receiver->receiverEmpty()) {
      std::shared_ptr<Input> data = receiver->popFront();
      this->unlockUniqueMutex();
      static_cast<CoreExecute<Input> *>(this)->callExecute(data);
//      this->executionCall<Input>(data);
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
