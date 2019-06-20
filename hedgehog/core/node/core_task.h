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
  //Store all AbstractTask clones
  std::vector<std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>>> clusterAbstractTask_;

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
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    std::cout << "->->Constructing Task " << this->name() << " / " << this->id() << " with the AbstractTask: "  << this->task() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
    HLOG_SELF(0, "Copy cluster CoreTask information from " << rhs->name() << "(" << rhs->id() << ")")
    CoreQueueMultiReceivers < TaskInputs...>::copyInnerStructure(rhs);
    CoreQueueSender < TaskOutput > ::copyInnerStructure(rhs);
    CoreNode::copyInnerStructure(rhs);
  }

  void createCluster(std::shared_ptr<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>> &insideNodesGraph) override {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    std::cout << std::endl;
//    std::cout << __PRETTY_FUNCTION__ << std::endl;
//    std::cout << "Creating cluster for: " << this->name() << " / " << this->id() << std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (this->numberThreads() > 1) { this->setInCluster(); }

//    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    std::cout << "copy whole node: " << this->id() << " gid: " << this->graphId() << std::endl;
//    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for (size_t threadId = 1; threadId < this->numberThreads(); threadId++) {
      auto taskCopy = createCopyFromThis();
      auto coreCopy = std::dynamic_pointer_cast<CoreTask<TaskOutput, TaskInputs...>>(taskCopy->core());
      coreCopy->threadId(threadId);
      coreCopy->coreClusterNode(this);
      coreCopy->copyInnerStructure(this);
      coreCopy->setInCluster();
      insideNodesGraph->insert({this, coreCopy});
    }
  }

  void run() override {
    std::chrono::time_point<std::chrono::high_resolution_clock>
        start,
        finish;

    HLOG_SELF(2, "Run")

    this->isActive(true);

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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    mutex.lock();
//    std::cout << "|||||||||||||||||||||||||| Task: " << this->name() << " / " << this->id() << " is dying." << std::endl;
//    mutex.unlock();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    this->postRun();

    this->wakeUp();
  }

  bool callCanTerminate(bool lock) {
    if (lock) {
      this->lockUniqueMutex();
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    mutex.lock();
//    std::cout
//    << this->name() << " id:" << this->id() << " gid:" << this->graphId()
//    << " canTerminate: " << std::boolalpha << this->node()->canTerminate()
//    << " receiversEmpty: " << this->receiversEmpty() << std::endl;
//    mutex.unlock();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

  std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> createCopyFromThis() {
    HLOG_SELF(0, "Copy Whole Task")
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    std::cout << "->->Create a copy of task : " << this->name() << " / " << this->id() << " with as AbstractTask: " << this->task() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> sharedAbstractTaskCopy = this->task()->copy();
    if (sharedAbstractTaskCopy == nullptr) {
      HLOG_SELF(0,
                "A copy for the queue " << this->name()
                                        << " has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and return a valid object.")
      std::cerr << "A copy for the queue " << this->name()
                << " has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and return a valid object."
                << std::endl;
      exit(42);
    }

    this->clusterAbstractTask_.push_back(sharedAbstractTaskCopy);

    auto coreCopy = std::dynamic_pointer_cast<CoreTask<TaskOutput, TaskInputs...>>(sharedAbstractTaskCopy->core());

    if (coreCopy == nullptr) {
      HLOG_SELF(0,
                "A copy for the queue " << this->name()
                                        << " has been invoked but the AbstractTask constructor has not been called. To fix this error, call the  AbstractTask constructor in your specialized queue constructor.")
      std::cerr << "A copy for the queue " << this->name()
                << " has been invoked but the AbstractTask constructor has not been called. To fix this error, call the  AbstractTask constructor in your specialized queue constructor."
                << std::endl;
      exit(42);
    }

    coreCopy->automaticStart_ = this->automaticStart();
    coreCopy->threadId(this->threadId());

    coreCopy->isInside(true);
    if (this->isInCluster()) { coreCopy->setInCluster(); }
    coreCopy->numberThreads(this->numberThreads());

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    std::cout << "New copy in cluster made: " << std::endl;
//    std::cout << "\t\tCore: " << coreCopy->name() << " / " << coreCopy->id() << std::endl;
//    std::cout << "->->Copy Created : " << coreCopy->name() << " / " << coreCopy->id() << " with as AbstractTask: " << coreCopy->task() << std::endl;
//
//    std::cout << "->->sharedAbstractTaskCopy: " << sharedAbstractTaskCopy << std::endl;
//    std::cout << "->->sharedAbstractTaskCopy->core(): " << sharedAbstractTaskCopy->core() << std::endl;
//    std::cout << "->->sharedAbstractTaskCopy->name(): " << sharedAbstractTaskCopy->name() << std::endl;
//    std::cout << "->->sharedAbstractTaskCopy->copy(): " << sharedAbstractTaskCopy->copy() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return sharedAbstractTaskCopy;
  }

};

#endif //HEDGEHOG_CORE_TASK_H
