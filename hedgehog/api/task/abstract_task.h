//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_TASK_H
#define HEDGEHOG_TASK_H

#include "../../behaviour/io/multi_receivers.h"
#include "../../behaviour/io/sender.h"
#include "../../behaviour/execute.h"
#include "../../behaviour/threadable.h"
#include "../../behaviour/memory_manager/abstract_memory_manager.h"
#include "../../core/node/core_node.h"
#include "../../core/defaults/core_default_task.h"

template<class TaskOutput, class ...TaskInputs>
class AbstractTask :
    public MultiReceivers<TaskInputs...>,
    public Sender<TaskOutput>,
    public virtual Node,
    public Execute<TaskInputs> ... {
 protected:
  std::shared_ptr<CoreTask<TaskOutput, TaskInputs...>> taskCore_ = nullptr;
  std::shared_ptr<AbstractMemoryManager<TaskOutput>> mm_ = nullptr;

  using Execute<TaskInputs>::execute...;

 public:
  AbstractTask() {
    taskCore_ = std::make_shared<CoreDefaultTask<TaskOutput, TaskInputs...>>("Task", 1, NodeType::Task, this, false);
  }

  explicit AbstractTask(std::string_view const &name) {
    taskCore_ =
        std::make_shared<CoreDefaultTask<TaskOutput, TaskInputs...>>(name,
                                                                     1,
                                                                     NodeType::Task,
                                                                     this,
                                                                     false);
  }

  explicit AbstractTask(std::string_view const &name, size_t numberThreads) {
    taskCore_ =
        std::make_shared<CoreDefaultTask<TaskOutput, TaskInputs...>>(name,
                                                                     numberThreads,
                                                                     NodeType::Task,
                                                                     this,
                                                                     false);
  }

  explicit AbstractTask(std::string_view const &name, size_t numberThreads, bool automaticStart) {
    taskCore_ =
        std::make_shared<CoreDefaultTask<TaskOutput, TaskInputs...>>(name,
                                                                     numberThreads,
                                                                     NodeType::Task,
                                                                     this,
                                                                     automaticStart);
  }

  AbstractTask(std::string_view const name, size_t numberThreads, NodeType nodeType, bool automaticStart) {
    taskCore_ = std::make_shared<CoreDefaultTask<TaskOutput, TaskInputs...>>(name,
                                                                             numberThreads,
                                                                             nodeType,
                                                                             this,
                                                                             automaticStart);
  }

  explicit AbstractTask(AbstractTask<TaskOutput, TaskInputs ...> *rhs) {
    taskCore_ = std::make_shared<CoreDefaultTask<TaskOutput, TaskInputs...>>(rhs->name(),
                                                                             rhs->numberThreads(),
                                                                             rhs->nodeType(),
                                                                             this,
                                                                             rhs->automaticStart());
  }

  ~AbstractTask() override = default;

  template<class Input, typename std::enable_if_t<HedgehogTraits::contains_v<Input, TaskInputs...>>>
  void pushData(std::shared_ptr<Input> &data) { this->taskCore_->pushData(data); }
  void addResult(std::shared_ptr<TaskOutput> output) { this->taskCore_->sendAndNotify(output); }

  std::string_view name() { return this->taskCore_->name(); }
  size_t numberThreads() { return this->taskCore_->numberThreads(); }
  bool automaticStart() { return this->taskCore_->automaticStart(); }
  NodeType nodeType() { return this->taskCore_->type(); }
  int deviceId() { return this->taskCore_->deviceId(); }
  std::shared_ptr<CoreNode> core() final { return taskCore_; }
  std::shared_ptr<AbstractMemoryManager<TaskOutput>> const &memoryManager() const { return mm_; }

  virtual std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> copy() { return nullptr; }
  virtual void initialize() {}

  void connectMemoryManager(std::shared_ptr<AbstractMemoryManager<TaskOutput>> mm) { mm_ = mm; }

  std::shared_ptr<TaskOutput> getManagedMemory() {
    if (mm_ == nullptr) {
      std::cerr
          << "For the task:\"" << this->name()
          << "\"To get managed memory, you need first to connect a memory manager to the task via \"connectMemoryManager()\""
          << std::endl;
      exit(42);
    }
    return mm_->getData();
  }

  bool canTerminate() override { return !this->taskCore_->hasNotifierConnected() && this->taskCore_->receiversEmpty(); }
  virtual void shutdown() {}

};

#endif //HEDGEHOG_TASK_H
