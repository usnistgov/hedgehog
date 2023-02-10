

// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.

#ifndef HEDGEHOG_POOL_H_
#define HEDGEHOG_POOL_H_

#pragma once

#include <deque>
#include <memory>
#include <sstream>
#include <condition_variable>

/// Hedgehog main namespace
namespace hh {
/// Hedgehog tool namespace
namespace tool {

/// @brief Pool of data used by the memory manager
/// @tparam T Type stored in the pool
template<class T>
class Pool {
 private:
  size_t const capacity_ = 1; ///< Capacity of the pool
  std::deque<std::shared_ptr<T>> queue_ = {}; ///< Actual storage used by the pool
  std::mutex mutex_ = {}; ///< Mutex used to protect the queue
  std::unique_ptr<std::condition_variable>
      conditionVariable_ = std::make_unique<std::condition_variable>(); ///< Condition variable to wake up a thread
                                                                        ///< waiting for data
 public:
  /// @brief Create a pool with a certain capacity
  /// @param capacity Pool's capacity
  explicit Pool(size_t const &capacity) : capacity_(capacity == 0 ? 1 : capacity) {
    queue_ = std::deque<std::shared_ptr<T >>(capacity_);
  }

  /// @brief Getter to the iterator to the beginning of the Pool
  /// @return Iterator to the beginning of the Pool
  typename std::deque<std::shared_ptr<T>>::iterator begin() { return this->queue_.begin(); }

  /// @brief Getter to the iterator to the end of the Pool
  /// @return Iterator to the end of the Pool
  typename std::deque<std::shared_ptr<T>>::iterator end() { return this->queue_.end(); }

  /// @brief Getter to the pool's size
  /// @return Pool size
  size_t size() {
    mutex_.lock();
    auto s = queue_.size();
    mutex_.unlock();
    return s;
  }

  /// @brief Returns true if the Pool is empty. (Thus begin() would equal end()).
  /// @return Returns true if the Pool is empty. (Thus begin() would equal end()).
  bool empty() {
    mutex_.lock();
    auto e = queue_.empty();
    mutex_.unlock();
    return e;
  }

  /// @brief Getter to the pool's capacity
  /// @return Pool capacity
  [[nodiscard]] size_t capacity() const { return capacity_; }

  /// @brief The function creates an element at the end of the pool and assigns the given data to it. Once inserted one
  /// waiting thread is woken up
  /// @param data Data to insert
  /// @throw std::runtime_error if the queue is overflowing
  void push_back(std::shared_ptr<T> const &data) {
    mutex_.lock();
    this->queue_.push_back(data);

    if (this->queue_.size() > capacity_) {
      std::ostringstream oss;
      oss << "The queue is overflowing, the same data " << data
          << " has been returned to the memory manager too many times: " << __FUNCTION__;
      throw (std::runtime_error(oss.str()));
    }
    mutex_.unlock();
    conditionVariable_->notify_one();
  }

  /// @brief Extract an element from the queue. If none is available wait until one become available
  /// @return Element from the queue
  std::shared_ptr<T> pop_front() {
    std::unique_lock<std::mutex> lock(mutex_);
    std::shared_ptr<T> ret = nullptr;
    conditionVariable_->wait(lock, [this]() { return !queue_.empty(); });
    ret = queue_.front();
    queue_.pop_front();
    return ret;
  }
};
}
}
#endif //HEDGEHOG_POOL_H_
