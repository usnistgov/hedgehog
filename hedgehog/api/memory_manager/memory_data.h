//
// Created by anb22 on 6/21/19.
//

#ifndef HEDGEHOG_MEMORY_DATA_H
#define HEDGEHOG_MEMORY_DATA_H

#include <memory>

template<class MANAGEDMEMORY>
class AbstractMemoryManager;

template<class MANAGEDMEMORY>
class MemoryData : public std::enable_shared_from_this<MemoryData<MANAGEDMEMORY>> {
 private:
  AbstractMemoryManager<MANAGEDMEMORY> *memoryManager_ = nullptr;
  std::mutex mutexMemData_ = {};
 public:
  MemoryData() = default;
  virtual ~MemoryData() = default;

  AbstractMemoryManager<MANAGEDMEMORY> *memoryManager() const { return memoryManager_; }
  void memoryManager(AbstractMemoryManager<MANAGEDMEMORY> *memoryManager) { memoryManager_ = memoryManager; }

  void returnToMemoryManager() {
    this->memoryManager_->releaseData(this->shared_from_this());
  }

  virtual void recycle() {};
  void lock() { mutexMemData_.lock(); }
  void unlock() { mutexMemData_.unlock(); }
};

#endif //HEDGEHOG_MEMORY_DATA_H
