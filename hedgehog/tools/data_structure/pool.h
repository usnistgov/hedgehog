//
// Created by anb22 on 5/24/19.
//

#ifndef HEDGEHOG_POOL_H
#define HEDGEHOG_POOL_H

#include <deque>
#include <memory>
#include <mutex>

template<class Data>
class AbstractMemoryManager;

template<class Data>
class ManagedMemory;

template<typename Data>
class Pool {

 private:
  std::deque<std::shared_ptr<ManagedMemory<Data>>>
      queue_ = {};

  std::mutex
      mutexQueue_ = {};

  std::size_t
      poolSize_ = 0;

 public:
  Pool() = delete;
  explicit Pool(size_t poolSize, AbstractMemoryManager<Data> *memoryManager) {
    poolSize_ = poolSize == 0 ? 1 : poolSize;
    for (size_t i = 0; i < poolSize_; ++i) {
      this->queue_.push_back(std::make_shared<ManagedMemory<Data>>(memoryManager));
    }

  }

  std::deque<std::shared_ptr<ManagedMemory<Data>>> const &queue() const {
    return queue_;
  }

  std::mutex &mutexQueue() { return mutexQueue_; }

  bool empty() { return this->queue_.empty(); }
  void pushBack(std::shared_ptr<ManagedMemory<Data>> data) {
    this->queue_.push_back(data);
    assert(this->queue_.size() <= poolSize_);
  }
  std::shared_ptr<ManagedMemory<Data>> popFront() {
    std::shared_ptr<ManagedMemory<Data>> ret = nullptr;
    if (!this->queue_.empty()) {
      ret = this->queue_.front();
      this->queue_.pop_front();
    }
    return ret;
  }

  void lockPool() { this->mutexQueue_.lock(); }
  void unlockPool() { this->mutexQueue_.unlock(); }
};

#endif //HEDGEHOG_POOL_H
