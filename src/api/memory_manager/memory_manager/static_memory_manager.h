//
// Created by anb22 on 5/24/19.
//

#ifndef HEDGEHOG_STATIC_MEMORY_MANAGER_H
#define HEDGEHOG_STATIC_MEMORY_MANAGER_H

#include "abstract_memory_manager.h"
template<class Data>
class StaticMemoryManager : public AbstractMemoryManager<Data> {
 public:
  StaticMemoryManager() = delete;
  template<
      class UserDefinedAllocator,
      class IsAllocator = typename std::enable_if<
          std::is_base_of_v<
              AbstractAllocator<Data>, UserDefinedAllocator
          >
      >::type
  >
  StaticMemoryManager(size_t poolSize, size_t numElements, std::unique_ptr<UserDefinedAllocator> allocator)
      : AbstractMemoryManager<Data>(poolSize, std::move(allocator)) {
    for (std::shared_ptr<ManagedMemory<Data>> mem : this->pool()->queue()) {
      HLOG(4, "StaticMemoryManager Allocating: " << mem)
      mem->data(this->allocator()->allocate(numElements));
    }
  }

  virtual ~StaticMemoryManager() = default;

  std::shared_ptr<ManagedMemory<Data>> getMemory(std::unique_ptr<AbstractReleaseRule<Data>> rule,
                                                 [[maybe_unused]]size_t numElements = 1) override {
    std::shared_ptr<ManagedMemory<Data>> managedMemory;
    std::unique_lock<std::mutex> lock(this->pool()->mutexQueue());
    HLOG(4, "StaticMemoryManager memory pool size = " << this->pool()->queue().size())
    this->conditionVariable()->wait(lock, [this]() { return !this->pool()->empty(); });
    managedMemory = this->pool()->popFront();
    HLOG(4,
         "StaticMemoryManager After waiting: received: " << managedMemory << " pSize: " << this->pool()->queue().size())
    managedMemory->rule(std::move(rule));
    return managedMemory;
  }

  void release(std::shared_ptr<ManagedMemory<Data>> managedMemory) override {
    managedMemory->lock();
    managedMemory->used();
    if (managedMemory->canRelease()) {
      managedMemory->unlock();
      this->pool()->lockPool();
      this->pool()->pushBack(managedMemory);
      HLOG(4,
           "StaticMemoryManager Received memory: " << managedMemory << " Release -- memory pool size = "
                                                   << this->pool()->queue().size())
      this->pool()->unlockPool();
      this->conditionVariable()->notify_one();
    } else {
      managedMemory->unlock();
    }
  }

};

#endif //HEDGEHOG_STATIC_MEMORY_MANAGER_H
