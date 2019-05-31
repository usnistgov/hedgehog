//
// Created by anb22 on 5/24/19.
//

#ifndef HEDGEHOG_MANAGED_MEMORY_H
#define HEDGEHOG_MANAGED_MEMORY_H

#include <memory>
#include <mutex>

#include "../memory_manager/abstract_memory_manager.h"
#include "abstract_release_rule.h"

template<class Data>
class ManagedMemory : public std::enable_shared_from_this<ManagedMemory<Data>> {
 private:
  Data *data_ = nullptr;

  AbstractMemoryManager<Data> *
      memoryManager_ = nullptr;

  std::mutex
      mutexMemory_ = {};

  std::unique_ptr<AbstractReleaseRule<Data>>
      releaseRule_ = nullptr;

 public:
  //Cstr
  ManagedMemory() = delete;
  explicit ManagedMemory(AbstractMemoryManager<Data> *memoryManager) : memoryManager_(memoryManager) {}

  //Dstr
  virtual ~ManagedMemory() = default;

  // Getter
  Data *data() const { return data_; }
  void memoryManager(AbstractMemoryManager<Data> *const &memoryManager) { memoryManager_ = memoryManager; }

  // Setter
  void data(Data *data) { ManagedMemory::data_ = data; }
  void rule(std::unique_ptr<AbstractReleaseRule<Data>> &&rule) { releaseRule_ = std::move(rule); }

  // Redirect calls
  void used() { releaseRule_->used(); }
  bool canRelease() { return this->releaseRule_->canRelease(); }
  void release() { this->memoryManager_->release(std::enable_shared_from_this<ManagedMemory<Data>>::shared_from_this()); }

  // Mutex call
  void lock() { this->mutexMemory_.lock(); }
  void unlock() { this->mutexMemory_.unlock(); }
};

#endif //HEDGEHOG_MANAGED_MEMORY_H
