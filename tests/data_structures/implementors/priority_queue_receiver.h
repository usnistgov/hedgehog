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

#ifndef HEDGEHOG_PRIORITY_QUEUE_RECEIVER_H
#define HEDGEHOG_PRIORITY_QUEUE_RECEIVER_H

#include "../../../hedgehog/hedgehog.h"

template<class Input>
class PriorityQueueReceiver : public hh::core::implementor::ImplementorReceiver<Input> {
 private:

  struct CustomSharedInputGreater {bool operator()(auto const & lhs, auto const & rhs) { return *lhs > *rhs;} } ;
  using QueueType = std::priority_queue<std::shared_ptr<Input>, std::vector<std::shared_ptr<Input>>, CustomSharedInputGreater>;

  std::unique_ptr<QueueType> const
      queue_ = nullptr; ///< Queue storing to be processed data

  std::unique_ptr<std::set<hh::core::abstraction::SenderAbstraction<Input> *>> const
      senders_ = nullptr; ///< List of senders attached to this receiver

  size_t
      maxSize_ = 0; ///< Maximum size attained by the queue

  std::mutex queueMutex_{}, sendersMutex_{};
 public:
  explicit PriorityQueueReceiver() : queue_(std::make_unique<QueueType>()),
  senders_(std::make_unique<std::set<hh::core::abstraction::SenderAbstraction<Input> *>>()) {}
  virtual ~PriorityQueueReceiver() = default;
  bool receive(std::shared_ptr<Input> data) final {
    std::lock_guard<std::mutex> lck(queueMutex_);
    queue_->push(data);
    maxSize_ = std::max(queue_->size(), maxSize_);
    return true;
  }
  [[nodiscard]] bool getInputData(std::shared_ptr<Input> &data) override {
    std::lock_guard<std::mutex> lck(queueMutex_);
    assert(!queue_->empty());
    data = queue_->top();
    queue_->pop();
    return true;
  }
  [[nodiscard]] size_t numberElementsReceived() override {
    std::lock_guard<std::mutex> lck(queueMutex_);
    return queue_->size();
  }
  [[nodiscard]] size_t maxNumberElementsReceived() const override { return maxSize_; }

  [[nodiscard]] bool empty() override {
    std::lock_guard<std::mutex> lck(queueMutex_);
    return queue_->empty();
  }

  [[nodiscard]] std::set<hh::core::abstraction::SenderAbstraction<Input> *> const &connectedSenders() const override {
    return *senders_;
  }
  void addSender(hh::core::abstraction::SenderAbstraction<Input> *const sender) override {
    std::lock_guard<std::mutex> lck(sendersMutex_);
    senders_->insert(sender); }
  void removeSender(hh::core::abstraction::SenderAbstraction<Input> *const sender) override {
    std::lock_guard<std::mutex> lck(sendersMutex_);
    senders_->erase(sender); }
};

#endif //HEDGEHOG_PRIORITY_QUEUE_RECEIVER_H
