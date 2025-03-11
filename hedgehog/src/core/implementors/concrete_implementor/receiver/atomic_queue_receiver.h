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

#ifndef HEDGEHOG_ATOMIC_QUEUE_RECEIVER_H
#define HEDGEHOG_ATOMIC_QUEUE_RECEIVER_H

#include "../../implementor/implementor_receiver.h"
#include "../../../abstractions/base/input_output/receiver_abstraction.h"
#include "../../../../../constants.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Hedgehog core namespace
namespace core {

/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Concrete implementation of the receiver core abstraction for a type using a queue using atomics
/// @warning The AtomicQueueReceiver can not store nullptr data !
/// @details This queue uses a simple linked list of nodes to store the data. It uses two atomics flags (one for the
/// producers and one of the consumers) to protect the queue access. It can be used by only one consumer and one
/// producer at a time.
/// @tparam Input Type of inputs stored in the queue
template<class Input>
class AtomicQueueReceiver : public ImplementorReceiver<Input> {
 private:
  /// @brief Definition of a node of the linked list
  struct alignas(CACHE_LINE_SIZE) Node {

    std::shared_ptr<Input> data_ = nullptr; ///< Data stored
    alignas(CACHE_LINE_SIZE) std::atomic<Node *> next_ = nullptr;  ///< Link to the next node

    /// @brief Constructor using a data to create a node
    /// @param data Data to use to create a node
    explicit Node(std::shared_ptr<Input> const &data) : data_(data), next_(nullptr) {}
  };

  Node
      *head_ = nullptr,///< Head of the list
  *tail_ = nullptr;///< Tail of the list

  alignas(CACHE_LINE_SIZE) std::atomic<bool>
      producerLock_{false}, ///< Atomic lock used to protect multi-producers concurrent access
  consumerLock_{false};///< Atomic lock used to protect multi-consumers concurrent access

  alignas(CACHE_LINE_SIZE) std::atomic<long long>
      queueSize_{0}, ///< Current queue size
  maxQueueSize_{0}; ///< Maximum filling size during its lifetime

  alignas(CACHE_LINE_SIZE) std::unique_ptr<std::set<abstraction::SenderAbstraction<Input> *>> const
  senders_ = nullptr; ///< List of senders attached to this receiver

  alignas(CACHE_LINE_SIZE)
  std::atomic_flag senderAccessFlag_{}; ///< Flag to protect access to the list of senders

 public:
  /// @brief Default constructor
  /// @details Initialize the queue with a default node with no data (nullptr)
  AtomicQueueReceiver() : senders_(std::make_unique<std::set<abstraction::SenderAbstraction<Input> *>>()) {
    head_ = tail_ = new Node(nullptr);
    producerLock_.store(false);
    consumerLock_.store(false);
    queueSize_.store(0);
    maxQueueSize_.store(0);
  }

  /// @brief Default destructor
  virtual ~AtomicQueueReceiver() {
    while (auto next = head_->next_.load()) {
      auto oldFirst = head_;
      head_ = next;
      delete oldFirst;
      oldFirst = nullptr;
    }
    delete head_;
    head_ = tail_ = nullptr;
  }

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

  /// @brief Store a piece of data in the atomic queue
  /// @param data Data to store
  /// @return True
  bool receive(std::shared_ptr<Input> data) override {
    assert(data != nullptr);
    auto newNode = new Node(data);

    while (producerLock_.exchange(true, std::memory_order_acquire));

    tail_->next_.store(newNode);
    tail_ = newNode;

    auto oldQS = queueSize_.fetch_add(1, std::memory_order_relaxed);
    maxQueueSize_.compare_exchange_strong(oldQS, oldQS + 1, std::memory_order_relaxed);

    producerLock_.store(false, std::memory_order_release);

    return true;
  }

  /// @brief Get a piece of data from the atomic queue
  /// @param data Reference uses to return the piece of data
  /// @return True
  bool getInputData(std::shared_ptr<Input> &data) override {
    data = nullptr;

    while (consumerLock_.exchange(true, std::memory_order_acquire));

    auto next = head_->next_.load();
    [[likely]] if (next != nullptr) {
      auto oldFirst = head_;
      head_ = next;
      std::swap(data, head_->data_);

      queueSize_.fetch_sub(1, std::memory_order_relaxed);
      consumerLock_.store(false, std::memory_order_release);

      delete oldFirst;
      return true;
    }
    consumerLock_.store(false, std::memory_order_release);
    return false;
  }

  /// @brief Get the "current size" of the queue
  /// @return Current queue size
  size_t numberElementsReceived() override {
    auto s = queueSize_.load();
    return s <= 0 ? 0 : static_cast<size_t>(s);
  }
  /// @brief Accessor to the maximum filling size during the queue lifetime
  /// @return Maximum filling size during the queue lifetime
  [[nodiscard]] size_t maxNumberElementsReceived() const override {
    return static_cast<size_t>(maxQueueSize_.load());
  }

  /// @brief Test if the receiver is empty or not
  /// @return True if the receiver is empty, else false
  bool empty() override { return numberElementsReceived() == 0; }
};

/// @brief Concrete implementation of the receiver core abstraction for multiple types using AtomicQueueReceiver
/// @tparam Inputs List of input types
template<class ...Inputs>
class MultiAtomicQueueReceivers : public AtomicQueueReceiver<Inputs> ... {
 public:
  /// @brief Default constructor
  explicit MultiAtomicQueueReceivers() : AtomicQueueReceiver<Inputs>()... {}

  /// Default destructor
  virtual ~MultiAtomicQueueReceivers() = default;
};

/// @brief Base definition of the type deducer for MultiAtomicQueueReceivers
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct MultiAtomicQueueReceiversTypeDeducer;

/// @brief Definition of the type deducer for MultiAtomicQueueReceivers
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct MultiAtomicQueueReceiversTypeDeducer<std::tuple<Inputs...>> {
  using type = core::implementor::MultiAtomicQueueReceivers<Inputs...>; ///< Type accessor
};

/// @brief Helper to the deducer for MultiAtomicQueueReceivers
/// @tparam TupleInputs Tuple of input types
template<class TupleInputs>
using MultiAtomicQueueReceiversTypeDeducer_t = typename MultiAtomicQueueReceiversTypeDeducer<TupleInputs>::type;

/// @brief Helper to the deducer for MultiAtomicQueueReceivers from the nodes template parameters
/// @tparam Separator Separator of node template arg
/// @tparam AllTypes All types of node template arg
template<size_t Separator, class ...AllTypes>
using MAQR = MultiAtomicQueueReceiversTypeDeducer_t<hh::tool::Inputs<Separator, AllTypes...>>;

}
}
}

#endif //HEDGEHOG_ATOMIC_QUEUE_RECEIVER_H
