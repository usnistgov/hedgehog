//
// Created by anb22 on 6/21/19.
//

#ifndef HEDGEHOG_POOL_H
#define HEDGEHOG_POOL_H

#include <memory>
#include <deque>
#include <algorithm>
#include <cassert>

template<class MANAGEDDATA>
class Pool {
 private:
  size_t poolSize_ = 1;
  std::deque<std::shared_ptr<MANAGEDDATA>> queue_ = {};
  std::mutex mutex_ = {};
  std::unique_ptr<std::condition_variable> conditionVariable_ = std::make_unique<std::condition_variable>();

 public:
  Pool() = default;
  virtual ~Pool() = default;

  void initialize(size_t poolSize) {
//    std::cout << "--> Initialize Pool" << std::endl;
    poolSize_ = poolSize == 0 ? 1 : poolSize;
    poolSize_ = poolSize;
    queue_ = std::deque<std::shared_ptr<MANAGEDDATA>>(poolSize);
    std::for_each(
        queue_.begin(), queue_.end(),
        [](std::shared_ptr<MANAGEDDATA> &emptyShared) {
          emptyShared = std::make_shared<MANAGEDDATA>();
//          std::cout << " Create " <<  emptyShared << " in pool " << this << " ." << std::endl;
        }
    );
  }

//  std::mutex &mutex() const { return mutex_; }
  std::deque<std::shared_ptr<MANAGEDDATA>> const &queue() const { return queue_; }
  typename std::deque<std::shared_ptr<MANAGEDDATA>>::iterator begin() { return this->queue_.begin(); }
  typename std::deque<std::shared_ptr<MANAGEDDATA>>::iterator end() { return this->queue_.end(); }
  size_t size() { return queue_.size(); }
  bool empty() { return queue_.empty(); }

  void push_back(std::shared_ptr<MANAGEDDATA> const &data) {
    std::lock_guard<std::mutex> lock(mutex_);
    this->queue_.push_back(data);
    assert(this->queue_.size() <= poolSize_);
    conditionVariable_->notify_one();
  }

  std::shared_ptr<MANAGEDDATA> pop_front() {
    std::unique_lock<std::mutex> lock(mutex_);
    std::shared_ptr<MANAGEDDATA> ret = nullptr;
    conditionVariable_->wait(lock, [this]() { return !queue_.empty(); });
    ret = queue_.front();
    queue_.pop_front();
    conditionVariable_->notify_one();
    return ret;
  }

};

#endif //HEDGEHOG_POOL_H
