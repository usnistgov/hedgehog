//
// Created by anb22 on 5/24/19.
//

#ifndef HEDGEHOG_DYNAMIC_MEMORY_MANAGER_H
#define HEDGEHOG_DYNAMIC_MEMORY_MANAGER_H

#include "abstract_memory_manager.h"

template<class Data>
class DynamicMemoryManager : public AbstractMemoryManager<Data> {

 public:
  //Cstr
  DynamicMemoryManager() = delete;
  template<
      class UserDefinedAllocator,
      class IsAllocator = typename std::enable_if<
          std::is_base_of_v<
              AbstractAllocator<Data>, UserDefinedAllocator
          >
      >::type
  >
  DynamicMemoryManager(size_t poolSize, std::unique_ptr<UserDefinedAllocator> allocator)
      : AbstractMemoryManager<Data>(poolSize,
                                    std::static_pointer_cast<AbstractAllocator<Data>>(std::move(allocator))) {}
  virtual ~DynamicMemoryManager() = default;

  std::shared_ptr<ManagedMemory<Data>> getMemory(std::unique_ptr<AbstractReleaseRule<Data>> rule,
                                                 size_t numElements) override {
    std::shared_ptr<ManagedMemory<Data>> managedMemory;
    std::unique_lock<std::mutex> lock(this->pool()->mutexQueue());
    HLOG(4, "DynamicMemoryManager memory pool size = " << this->pool()->queue().size())
    this->conditionVariable()->wait(lock, [this]() { return !this->pool()->empty(); });
    managedMemory = this->pool()->popFront();
    HLOG(4, "DynamicMemoryManager Getting: " << managedMemory)
    managedMemory->data(this->allocator()->allocate(numElements));
    HLOG(4, "DynamicMemoryManager Allocating: " << managedMemory)
    managedMemory->rule(std::move(rule));
    return managedMemory;
  }

  void release(std::shared_ptr<ManagedMemory<Data>> managedMemory) override {
    managedMemory->lock();

    managedMemory->used();
    if (managedMemory->canRelease()) {

      auto data = managedMemory->data();
      this->allocator()->deallocate(data);
      managedMemory->unlock();
      this->pool()->lockPool();
      this->pool()->pushBack(managedMemory);
      HLOG(4,
           "DynamicMemoryManager Received memory: " << managedMemory << " Release -- memory pool size = "
                                                    << this->pool()->queue().size())
      this->pool()->unlockPool();
      this->conditionVariable()->notify_one();

    } else {
      managedMemory->unlock();
    }
  }
};

#endif //HEDGEHOG_DYNAMIC_MEMORY_MANAGER_H
