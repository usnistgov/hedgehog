//
// Created by anb22 on 5/28/19.
//

#ifndef HEDGEHOG_MANAGED_MEMORY_ABSTRACT_TASK_H
#define HEDGEHOG_MANAGED_MEMORY_ABSTRACT_TASK_H

#include "../memory_manager/abstract_memory_manager.h"

template<class TaskOutput, class ...TaskInputs>
class ManagedMemoryAbstractTask : public AbstractTask<ManagedMemory<TaskOutput>, TaskInputs...> {
 private:
  std::shared_ptr<AbstractMemoryManager<TaskOutput>>
      memoryManager_ = nullptr;

 public:
  ManagedMemoryAbstractTask() = delete;

  explicit ManagedMemoryAbstractTask(std::shared_ptr<AbstractMemoryManager<TaskOutput>> const &mm)
      : memoryManager_(mm) {}

  ManagedMemoryAbstractTask(std::shared_ptr<AbstractMemoryManager<TaskOutput>> const &mm,
                            std::string_view const &name,
                            size_t numberThreads = 1,
                            bool automaticStart = false)
      : AbstractTask<ManagedMemory<TaskOutput>, TaskInputs...>(name, numberThreads, automaticStart) {
    this->memoryManager_ = mm;
  }

  explicit ManagedMemoryAbstractTask(ManagedMemoryAbstractTask<TaskOutput, TaskInputs...> *rhs)
      : AbstractTask<ManagedMemory<TaskOutput>, TaskInputs...>(rhs), memoryManager_(rhs->memoryManager_) {}

  std::shared_ptr<AbstractMemoryManager<TaskOutput>> const &memoryManager() const { return memoryManager_; }

  std::shared_ptr<ManagedMemory<TaskOutput>> getMemory(std::unique_ptr<AbstractReleaseRule<TaskOutput>> rr,
                                                       int numberElements) {
    assert(memoryManager_ != nullptr);
    return this->memoryManager_->getMemory(std::move(rr), numberElements);
  }

  void release(std::shared_ptr<ManagedMemory<TaskOutput>> managedMemory) {
    assert(memoryManager_ != nullptr);
    this->memoryManager_->release(managedMemory);
  }

};

#endif //HEDGEHOG_MANAGED_MEMORY_ABSTRACT_TASK_H
