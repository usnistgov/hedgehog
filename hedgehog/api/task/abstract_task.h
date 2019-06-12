//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_TASK_H
#define HEDGEHOG_TASK_H

#include "../../behaviour/io/multi_receivers.h"
#include "../../behaviour/io/sender.h"
#include "../../behaviour/execute.h"
#include "../../behaviour/threadable.h"
#include "../../core/node/core_node.h"
#include "../../core/defaults/default_task_core.h"

template<class TaskOutput, class ...TaskInputs>
class AbstractTask :
    public MultiReceivers<TaskInputs...>,
    public Sender<TaskOutput>,
    public virtual Node,
    public Execute<TaskInputs> ... {
 protected:
  CoreTask<TaskOutput, TaskInputs...> *taskCore_ = nullptr;
  using Execute<TaskInputs>::execute...;

 public:
  AbstractTask() {
    taskCore_ = new DefaultTaskCore<TaskOutput, TaskInputs...>("Task", 1, NodeType::Task, this, false);
  }

  explicit AbstractTask(std::string_view const &name, size_t numberThreads = 1, bool automaticStart = false) {
    taskCore_ =
        new DefaultTaskCore<TaskOutput, TaskInputs...>(name, numberThreads, NodeType::Task, this, automaticStart);
  }

  AbstractTask(std::string_view const name, size_t numberThreads, NodeType nodeType, bool automaticStart) {
    taskCore_ = new DefaultTaskCore<TaskOutput, TaskInputs...>(name, numberThreads, nodeType, this, automaticStart);
  }

  explicit AbstractTask(AbstractTask<TaskOutput, TaskInputs ...> *rhs) {
    taskCore_ = new DefaultTaskCore<TaskOutput, TaskInputs...>(rhs->name(),
                                                               rhs->numberThreads(),
                                                               rhs->nodeType(),
                                                               this,
                                                               rhs->automaticStart());
  }

  ~AbstractTask() override { delete taskCore_; }

  std::string_view name() { return this->taskCore_->name(); }
  size_t numberThreads() { return this->taskCore_->numberThreads(); }
  bool automaticStart() { return this->taskCore_->automaticStart(); }
  NodeType nodeType() { return this->taskCore_->type(); }

  CoreNode *core() final { return taskCore_; }

  virtual std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> copy() { return nullptr; }
  virtual void initialize() {}
  bool canTerminate() override {
    return !this->taskCore_->hasNotifierConnected();
  }

  virtual void shutdown() {}

  template<class Input, typename std::enable_if_t<HedgehogTraits::contains_v<Input, TaskInputs...>>>
  void pushData(std::shared_ptr<Input> &data) { this->taskCore_->pushData(data); }
  void addResult(std::shared_ptr<TaskOutput> output) { this->taskCore_->sendAndNotify(output); }
};

#endif //HEDGEHOG_TASK_H
