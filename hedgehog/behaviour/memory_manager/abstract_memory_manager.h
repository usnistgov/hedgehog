//
// Created by anb22 on 5/24/19.
//

#ifndef HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H
#define HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H

#include <deque>
#include <memory>
#include <condition_variable>

#include "abstract_allocator.h"
#include "../../api/memory_manager/abstract_release_rule.h"
#include "../../tools/logger.h"
#include "../../tools/data_structure/pool.h"

template<class Data>
class AbstractMemoryManager {
 private:
  std::unique_ptr<Pool<Data>>
      pool_;

  std::unique_ptr<AbstractAllocator<Data>>
      allocator_;

  std::unique_ptr<std::condition_variable>
      conditionVariable_;

 public:
  AbstractMemoryManager() = delete;
  template<
      class UserDefinedAllocator,
      class IsAllocator = typename std::enable_if<
          std::is_base_of_v<AbstractAllocator<Data>, UserDefinedAllocator>
      >::type
  >
  AbstractMemoryManager(size_t poolSize, std::unique_ptr<UserDefinedAllocator> allocator)
      : conditionVariable_(std::make_unique<std::condition_variable>()) {
    this->pool_ = std::make_unique<Pool<Data>>(poolSize, this);
    this->allocator_ =
        std::unique_ptr<AbstractAllocator<Data>>{
            static_cast<AbstractAllocator<Data> *>(std::move(allocator).release())};
    this->allocator_->initialize();
  }

  virtual ~AbstractMemoryManager() {
    HLOG(4, "AbstractMemoryManager Destruction")
    for (std::shared_ptr<ManagedMemory<Data>> mem : this->pool()->queue()) {
      HLOG(4, "AbstractMemoryManager Deallocating: " << mem)
      this->allocator()->deallocate(mem->data());
    }
  }

  virtual std::shared_ptr<ManagedMemory<Data>> getMemory(std::unique_ptr<AbstractReleaseRule<Data>>, size_t) = 0;
  virtual void release(std::shared_ptr<ManagedMemory<Data>>) = 0;

 protected:
  std::unique_ptr<Pool<Data>> const &pool() const { return pool_; }
  std::unique_ptr<AbstractAllocator<Data>> const &allocator() const { return allocator_; }
  std::unique_ptr<std::condition_variable> const &conditionVariable() const { return conditionVariable_; }
};

#endif //HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H
