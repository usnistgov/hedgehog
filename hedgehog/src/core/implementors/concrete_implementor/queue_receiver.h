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



#ifndef HEDGEHOG_QUEUE_RECEIVER_H
#define HEDGEHOG_QUEUE_RECEIVER_H

#include <iostream>
#include <cassert>
#include <queue>
#include <mutex>
#include <set>
#include <utility>

#include "../../abstractions/base/input_output/receiver_abstraction.h"
#include "../implementor/implementor_receiver.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

/// @brief Hedgehog abstraction namespace
namespace abstraction {
template<class Output>
class SenderAbstraction;
}

/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Concrete implementation of the receiver core abstraction for a type using a std::queue
/// @tparam Input Input data type
template<class Input>
class QueueReceiver : public ImplementorReceiver<Input> {
 private:
  std::unique_ptr<std::queue<std::shared_ptr<Input>>> const
      queue_ = nullptr; ///< Queue storing to be processed data

  std::unique_ptr<std::set<abstraction::SenderAbstraction<Input> *>> const
      senders_ = nullptr; ///< List of senders attached to this receiver

  size_t
      maxSize_ = 0; ///< Maximum size attained by the queue

 public:
  /// @brief Default constructor
  explicit QueueReceiver()
  : queue_(std::make_unique<std::queue<std::shared_ptr<Input>>>()),
      senders_(std::make_unique<std::set<abstraction::SenderAbstraction<Input> *>>()) {}

  /// @brief Default destructor
  virtual ~QueueReceiver() = default;

  /// @brief Receive a data and store it in the queue
  /// @param data Data to store
  void receive(std::shared_ptr<Input> const data) final {
    queue_->push(data);
    maxSize_ = std::max(queue_->size(), maxSize_);
  }

  /// @brief Get a data from the queue
  /// @attention The queue should not be empty
  /// @return The data in front of the queue
  [[nodiscard]] std::shared_ptr<Input> getInputData() override {
    assert(!queue_->empty());
    auto front = queue_->front();
    queue_->pop();
    return front;
  }

  /// @brief Accessor to the current size of the queue
  /// @return Current size of the queue
  [[nodiscard]] size_t numberElementsReceived() const override { return queue_->size(); }

  /// @brief Accessor to the maximum queue size
  /// @return Maximum queue size
  [[nodiscard]] size_t maxNumberElementsReceived() const override { return maxSize_; }

  /// @brief Test if the queue is empty
  /// @return True if the queue is empty, else false
  [[nodiscard]] bool empty() const override { return queue_->empty(); }

  /// @brief Accessor to the set of connected senders
  /// @return Set of connected senders
  [[nodiscard]] std::set<abstraction::SenderAbstraction<Input> *> const &connectedSenders() const override {
    return *senders_;
  }

  /// @brief Add a sender to the set of connected senders
  /// @param sender Sender to add
  void addSender(abstraction::SenderAbstraction<Input> *const sender) override { senders_->insert(sender); }

  /// @brief Remove a sender to the set of connected senders
  /// @param sender Sender to remove
  void removeSender(abstraction::SenderAbstraction<Input> *const sender) override { senders_->erase(sender); }

};

}
}
}
#endif //HEDGEHOG_QUEUE_RECEIVER_H
