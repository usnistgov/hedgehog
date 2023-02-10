
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

#ifndef HEDGEHOG_PRIORITY_QUEUE_RECEIVER_H_
#define HEDGEHOG_PRIORITY_QUEUE_RECEIVER_H_

#include "../../../hedgehog/hedgehog.h"

template<class Input>
class PriorityQueueReceiver : public hh::core::implementor::ImplementorReceiver<Input> {
 private:
  std::unique_ptr<std::priority_queue<std::shared_ptr<Input>>> const
      queue_ = nullptr; ///< Queue storing to be processed data

  std::unique_ptr<std::set<hh::core::abstraction::SenderAbstraction<Input> *>> const
      senders_ = nullptr; ///< List of senders attached to this receiver

  size_t
      maxSize_ = 0; ///< Maximum size attained by the queue

 public:
  explicit PriorityQueueReceiver()
      : queue_(std::make_unique<std::priority_queue<std::shared_ptr<Input>>>()),
  senders_(std::make_unique<std::set<hh::core::abstraction::SenderAbstraction<Input> *>>()) {}
  virtual ~PriorityQueueReceiver() = default;
  void receive(std::shared_ptr<Input> const data) final {
    queue_->push(data);
    maxSize_ = std::max(queue_->size(), maxSize_);
  }
  [[nodiscard]] std::shared_ptr<Input> getInputData() override {
    assert(!queue_->empty());
    auto front = queue_->top();
    queue_->pop();
    return front;
  }
  [[nodiscard]] size_t numberElementsReceived() const override { return queue_->size(); }
  [[nodiscard]] size_t maxNumberElementsReceived() const override { return maxSize_; }
  [[nodiscard]] bool empty() const override { return queue_->empty(); }
  [[nodiscard]] std::set<hh::core::abstraction::SenderAbstraction<Input> *> const &connectedSenders() const override {
    return *senders_;
  }
  void addSender(hh::core::abstraction::SenderAbstraction<Input> *const sender) override { senders_->insert(sender); }
  void removeSender(hh::core::abstraction::SenderAbstraction<Input> *const sender) override { senders_->erase(sender); }
};

#endif //HEDGEHOG_PRIORITY_QUEUE_RECEIVER_H_
