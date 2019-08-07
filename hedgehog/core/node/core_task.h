//
// Created by 775backup on 2019-04-08.
//

#ifndef HEDGEHOG_CORE_TASK_H
#define HEDGEHOG_CORE_TASK_H

#include <variant>
#include <string_view>


#include "../../tools/traits.h"
#include "../io/queue/receiver/core_queue_multi_receivers.h"
#include "../io/queue/sender/core_queue_sender.h"

#include "../../behaviour/execute.h"

#include "../../tools/logger.h"
#include "../../tools/nvtx_profiler.h"
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
  std::vector<std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>>> clusterAbstractTask_ = {};

  std::shared_ptr<NvtxProfiler> nvtxProfiler_ = nullptr;

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
    nvtxProfiler_ = std::make_shared<NvtxProfiler>(this->name());
  }

  ~CoreTask() override {
    HLOG_SELF(0, "Destructing CoreTask")
    task_ = nullptr;
  }

  bool automaticStart() const { return automaticStart_; }
  Node *node() override { return task_; }
  AbstractTask<TaskOutput, TaskInputs...> *task() const { return task_; }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
      CoreQueueSender<TaskOutput>::visit(printer);
    }
  }

  void copyInnerStructure(CoreTask<TaskOutput, TaskInputs...> *rhs) {
    HLOG_SELF(0, "Copy cluster CoreTask information from " << rhs->name() << "(" << rhs->id() << ")")
    CoreQueueMultiReceivers < TaskInputs...>::copyInnerStructure(rhs);
    CoreQueueSender<TaskOutput>::copyInnerStructure(rhs);
    CoreNode::copyInnerStructure(rhs);
  }

  void createCluster(std::shared_ptr<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>> &insideNodesGraph) override {
    auto mm = this->task()->memoryManager();
    if (this->numberThreads() > 1) { this->setInCluster(); }

    for (size_t threadId = 1; threadId < this->numberThreads(); threadId++) {
      auto taskCopy = createCopyFromThis();
      if (mm) {
        HLOG_SELF(1, "Copy the MM pointer " << mm
                                            << " from: " << this->name() << " / " << this->id()
                                            << " to: " << taskCopy->core()->name() << " / " << taskCopy->core()->id())
        taskCopy->connectMemoryManager(mm);
      }
      auto coreCopy = std::dynamic_pointer_cast<CoreTask<TaskOutput, TaskInputs...>>(taskCopy->core());
      coreCopy->threadId(threadId);
      coreCopy->coreClusterNode(this);
      coreCopy->copyInnerStructure(this);
      coreCopy->setInCluster();
      insideNodesGraph->insert({this, coreCopy});
    }
  }

  void run() override {
    HLOG_SELF(2, "Run")
    std::chrono::time_point<std::chrono::high_resolution_clock>
        start,
        finish;

    this->isActive(true);
    this->nvtxProfiler()->initialize(this->threadId());
    this->preRun();

    if (this->automaticStart()) {
      start = std::chrono::high_resolution_clock::now();
      (static_cast<CoreExecute<TaskInputs> *>(this)->callExecute(nullptr), ...);
      finish = std::chrono::high_resolution_clock::now();
      this->incrementExecutionDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
    }
    while (!this->callCanTerminate(true)) {
      start = std::chrono::high_resolution_clock::now();
      volatile bool canTerminate = this->waitForNotification();
      finish = std::chrono::high_resolution_clock::now();
      this->incrementWaitDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
      if (canTerminate) { break; }
      start = std::chrono::high_resolution_clock::now();
      (this->operateReceivers<TaskInputs>(), ...);
      finish = std::chrono::high_resolution_clock::now();
      this->incrementExecutionDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
    }

    this->postRun();
    this->wakeUp();
  }

  virtual bool callCanTerminate(bool lock) {
    bool result;
    if (lock) { this->lockUniqueMutex(); }
    result = this->node()->canTerminate();
    HLOG_SELF(2, "callCanTerminate: " << std::boolalpha << result)
    if (lock) { this->unlockUniqueMutex(); }
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

  bool waitForNotification() override {
    this->nvtxProfiler()->startRangeWaiting(this->totalQueueSize());
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
	assert(lock.owns_lock());

    this->nvtxProfiler()->endRangeWaiting();
    return callCanTerminate(false);
  }

  std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> createCopyFromThis() {
    HLOG_SELF(0, "Copy Whole Task")

    std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> sharedAbstractTaskCopy = this->task()->copy();
    if (sharedAbstractTaskCopy == nullptr) {
      HLOG_SELF(0,
                "A copy for the task \"" << this->name()
                                         << "\" has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and return a valid object.")
      std::cerr << "A copy for the task \"" << this->name()
                << "\" has been invoked but return nullptr. To fix this error, overload the AbstractTask::copy function and return a valid object."
                << std::endl;
      exit(42);
    }

    this->clusterAbstractTask_.push_back(sharedAbstractTaskCopy);

    auto coreCopy = std::dynamic_pointer_cast<CoreTask<TaskOutput, TaskInputs...>>(sharedAbstractTaskCopy->core());

    if (coreCopy == nullptr) {
      std::ostringstream oss;
      oss << "A copy for the queue " << this->name()
          << " has been invoked but the AbstractTask constructor has not been called. To fix this error, call the "
             "AbstractTask constructor in your specialized queue constructor.";
      HLOG_SELF(0, oss.str())
      std::cerr << oss.str() << std::endl;
      exit(42);
    }

    coreCopy->automaticStart_ = this->automaticStart();
    coreCopy->threadId(this->threadId());

    coreCopy->isInside(true);
    if (this->isInCluster()) { coreCopy->setInCluster(); }
    coreCopy->numberThreads(this->numberThreads());

    if (this->task()->memoryManager()) {
      auto copyMemoryManager = this->task()->memoryManager()->copy();
      if (!copyMemoryManager) {
        std::ostringstream oss;
        oss << "An execution pipeline has been created with a graph that hold a task named \""
            << this->name()
            << "\" connected to a memory manager. Or the memory manager does not have a compliant copy method. "
               "Please implement it.";
        HLOG_SELF(0, oss.str())
        std::cerr << oss.str() << std::endl;
        exit(42);
      }
      HLOG_SELF(0, "Copy the MM " << this->task()->memoryManager() << " to: " << copyMemoryManager
                                  << " and set it to the task: " << coreCopy->name() << " / " << coreCopy->id())
      sharedAbstractTaskCopy->connectMemoryManager(copyMemoryManager);
    }
    return sharedAbstractTaskCopy;
  }

  std::shared_ptr<NvtxProfiler> nvtxProfiler() {
    return nvtxProfiler_;
  }
};

#endif //HEDGEHOG_CORE_TASK_H
