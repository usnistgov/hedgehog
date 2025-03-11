//  NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
//  software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
//  derivative works of the software or any portion of the software, and you may copy and distribute such modifications
//  or works. Modified works should carry a notice stating that you changed the software and should note the date and
//  nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
//  source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
//  EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
//  WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
//  CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
//  THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
//  are solely responsible for determining the appropriateness of using and distributing the software and you assume
//  all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
//  with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
//  operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
//  damage to property. The software developed by NIST employees is not subject to copyright protection within the
//  United States.

// MIT License
//
// Copyright (c) 2019 Maxim Egorushkin
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#ifndef HEDGEHOG_LIMITED_ATOMIC_QUEUE_RECEIVER_H
#define HEDGEHOG_LIMITED_ATOMIC_QUEUE_RECEIVER_H

#include <execution>
// #include "../../tools/intrinsics.h"

#include "../../implementor/implementor_receiver.h"
#include "../../../abstractions/base/input_output/receiver_abstraction.h"
#include "../../../../../constants.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Hedgehog core namespace
namespace core {

/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Concrete implementation of the receiver core abstraction for a type using a limited queue using atomics
/// @details This queue is heavily inspired by Maxim Egorushkin work (https://github.com/max0x7ba/atomic_queue)
/// published under MIT license. This class only follow the same license.
/// The queue uses a fixed size ring-buffer of atomic elements. It can store at maximum MaxCapacity elements. During its
/// lifetime it can stores at most std::numeric_limits<long long>::max() elements.
/// @tparam Input Type of inputs stored in the queue
/// @tparam MaxCapacity Queue maximum capacity
template<class Input, long long MaxCapacity>
class LimitedAtomicQueueReceiver : public ImplementorReceiver<Input> {
 private:
  alignas(CACHE_LINE_SIZE) std::atomic<long long>
      head_{0}, ///< Head index
  tail_{0}; ///< Tail index
  alignas(CACHE_LINE_SIZE) std::atomic<size_t>
      maxSize_{0}; ///< Maximum filling size during its lifetime

  alignas(CACHE_LINE_SIZE) std::atomic<std::shared_ptr<Input>>
      data_[MaxCapacity] = {}; ///< Ring buffer of atomic values

  alignas(CACHE_LINE_SIZE) std::unique_ptr<std::set<abstraction::SenderAbstraction<Input> *>> const
      senders_ = nullptr; ///< List of senders attached to this receiver

  alignas(CACHE_LINE_SIZE)
  std::atomic_flag senderAccessFlag_{}; ///< Flag to protect access to the list of senders

 public:
  /// @brief Default constructor
  LimitedAtomicQueueReceiver() : senders_(std::make_unique<std::set<abstraction::SenderAbstraction<Input> *>>()) {}

  /// @brief Accessor of the connected senders
  /// @return Set of connected senders
  std::set<abstraction::SenderAbstraction<Input> *> const &connectedSenders() const override { return *senders_; }

  /// @brief Add a sender to the set of connected senders
  /// @param sender Sender to add to the connected senders
  void addSender(abstraction::SenderAbstraction<Input> *sender) override {
    while (senderAccessFlag_.test_and_set(std::memory_order_acquire)) { cross_platform_yield(); }
    senders_->insert(sender);
    senderAccessFlag_.clear(std::memory_order_release);
  }

  /// @brief Remove a sender to the set of connected senders
  /// @param sender Sender to remove from the connected senders
  void removeSender(abstraction::SenderAbstraction<Input> *sender) override {
    while (senderAccessFlag_.test_and_set(std::memory_order_acquire)) { cross_platform_yield(); }
    senders_->erase(sender);
    senderAccessFlag_.clear(std::memory_order_release);
  }

  /// @brief Receive a piece of data to store it in the limited queue
  /// @warning The storage may fail, the function returns then false
  /// @param data Data to store
  /// @return True if the data has been stored, else false
  bool receive(std::shared_ptr<Input> data) override {
    assert(data != nullptr);
    auto head = head_.load(std::memory_order_relaxed);
    do { if (head - tail_.load(std::memory_order_relaxed) >= MaxCapacity) { return false; }}
    while (!head_.compare_exchange_strong(head, head + 1, std::memory_order_relaxed, std::memory_order_relaxed));
    auto &refData = data_[head % MaxCapacity];
    std::shared_ptr<Input> expected = nullptr;

    while (!refData.compare_exchange_strong(expected, data, std::memory_order_relaxed, std::memory_order_relaxed)) {
      expected = nullptr;
      do { cross_platform_yield(); } while (refData.load(std::memory_order_relaxed) != nullptr);
    }

    auto diff = head - tail_.load(std::memory_order_relaxed);;
    if (diff > 0) {
      auto const prevSize = static_cast<size_t>(diff);
      if (prevSize > maxSize_.load(std::memory_order_acquire)) { maxSize_.store(prevSize, std::memory_order_release); }
    }
    return true;
  }

  /// @brief Get a piece of data from the limited queue
  /// @warning The operation may fail, the function returns then false
  /// @param data Reference uses to return the piece of data
  /// @return True if the data has been returned, else false
  bool getInputData(std::shared_ptr<Input> &data) override {
    auto tail = tail_.load(std::memory_order_relaxed);
    do { if (head_.load(std::memory_order_relaxed) - tail <= 0) { return false; }}
    while (!tail_.compare_exchange_strong(tail, tail + 1, std::memory_order_relaxed, std::memory_order_relaxed));

    auto &refData = data_[tail % MaxCapacity];
    while (true) {
      data = refData.exchange(nullptr, std::memory_order_acquire);
      if (data != nullptr) { return true; }
      do { cross_platform_yield(); } while (refData.load(std::memory_order_relaxed) == nullptr);
    }
  }

  /// @brief Get the "current size" of the queue
  /// @details Current size is deduced from the head and tail pointers, both atomic values.
  /// @return Current queue size
  [[nodiscard]] size_t numberElementsReceived() override {
    auto h = head_.load(std::memory_order_relaxed);
    auto t = tail_.load(std::memory_order_relaxed);
    if (h > t) { return static_cast<size_t>(h - t); }
    else {
      if (h == t) {
        // If the queue is empty, retry to verify that the queue is *really* empty
        auto h2 = head_.load(std::memory_order_relaxed);
        auto t2 = tail_.load(std::memory_order_relaxed);
        if (h == h2 && t == t2) { return 0; }
      }
    }
    // Error case where both loads of the head and tail give an impossible size (negative value for example)
    return 1;
  }

  /// @brief Accessor to the maximum filling size during the queue lifetime
  /// @return Maximum filling size during the queue lifetime
  [[nodiscard]] size_t maxNumberElementsReceived() const override { return maxSize_.load(std::memory_order_relaxed); }

  /// @brief Test if the receiver is empty or not
  /// @return True if the receiver is empty, else false
  [[nodiscard]] bool empty() override { return numberElementsReceived() == 0; }
};

/// @brief Concrete implementation of the LimitedAtomicQueueReceivers for multiple types
/// @tparam MaxCapacity Queue maximum capacity
/// @tparam Inputs List of input types
template<long long MaxCapacity, class ...Inputs>
class MultiLimitedAtomicQueueReceivers : public LimitedAtomicQueueReceiver<Inputs, MaxCapacity> ... {
 public:
  /// @brief Default constructor
  explicit MultiLimitedAtomicQueueReceivers() : LimitedAtomicQueueReceiver<Inputs, MaxCapacity>()... {}

  /// Default destructor
  virtual ~MultiLimitedAtomicQueueReceivers() = default;
};

/// @brief Base definition of the type deducer for MultiLimitedAtomicQueueReceivers
/// @tparam MaxCapacity Queue maximum capacity
/// @tparam Inputs Input types as tuple
template<long long MaxCapacity, class Inputs>
struct MultiLimitedAtomicQueueReceiversTypeDeducer;

/// @brief Definition of the type deducer for MultiLimitedAtomicQueueReceivers
/// @tparam MaxCapacity Queue maximum capacity
/// @tparam Inputs Variadic of types
template<long long MaxCapacity, class ...Inputs>
struct MultiLimitedAtomicQueueReceiversTypeDeducer<MaxCapacity, std::tuple<Inputs...>> {
  using type = core::implementor::MultiLimitedAtomicQueueReceivers<MaxCapacity, Inputs...>; ///< Type accessor
};

/// @brief Helper to the deducer for MultiLimitedAtomicQueueReceivers
/// @tparam MaxCapacity Queue maximum capacity
/// @tparam TupleInputs Tuple of input types
template<long long MaxCapacity, class TupleInputs>
using MultiLimitedAtomicQueueReceiversTypeDeducer_t
= typename MultiLimitedAtomicQueueReceiversTypeDeducer<MaxCapacity, TupleInputs>::type;

/// @brief Helper to the deducer for MultiLimitedAtomicQueueReceivers from the nodes template parameters
/// @tparam MaxCapacity Queue maximum capacity
/// @tparam Separator Separator of node template arg
/// @tparam AllTypes All types of node template arg
template<long long MaxCapacity, size_t Separator, class ...AllTypes>
using MLAQR = MultiLimitedAtomicQueueReceiversTypeDeducer_t<MaxCapacity, hh::tool::Inputs<Separator, AllTypes...>>;

}
}
}

#endif //HEDGEHOG_LIMITED_ATOMIC_QUEUE_RECEIVER_H
