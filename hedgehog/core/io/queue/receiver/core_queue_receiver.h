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


#ifndef HEDGEHOG_CORE_QUEUE_RECEIVER_H
#define HEDGEHOG_CORE_QUEUE_RECEIVER_H

#include "../../base/receiver/core_receiver.h"
#include "core_queue_slot.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Receiver for nodes possessing a queue of data
/// @tparam NodeInput Node input types
template<class NodeInput>
class CoreQueueReceiver : public virtual CoreReceiver<NodeInput> {
 private:
  std::shared_ptr<std::queue<std::shared_ptr<NodeInput>>> queue_ = nullptr;  ///< Waiting list of data to be processed
  std::shared_ptr<std::set<CoreSender < NodeInput> *>> senders_ = nullptr;     ///< Senders connected to this receiver
  size_t maxQueueSize_ = 0;                                                  ///< Maximum queue size registered

 public:
  /// @brief CoreQueueReceiver constructor
  /// @param name Node's name
  /// @param type Node's type
  /// @param numberThreads Node's number of thread
  CoreQueueReceiver(std::string_view const &name, NodeType const type, size_t const numberThreads) : CoreReceiver<
      NodeInput>(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueReceiver with type: " << (int) type << " and name: " << name)
    queue_ = std::make_shared<std::queue<std::shared_ptr<NodeInput>>>();
    senders_ = std::make_shared<std::set<CoreSender<NodeInput> *>>();
  }

  /// @brief CoreQueueReceiver default destructor
  ~CoreQueueReceiver() override {HLOG_SELF(0, "Destructing CoreQueueReceiver")}

  //Virtual
  /// @brief Queue slot accessor possessed by a CoreQueueMultiReceiver
  /// @return Queue slot possessed by a CoreQueueMultiReceiver
  virtual CoreQueueSlot *queueSlot() = 0;

  /// @brief Return the current waiting data queue size
  /// @return The current waiting data queue size
  size_t queueSize() override { return this->queue_->size(); }

  /// @brief Return the maximum current waiting data queue size registered
  /// @return The maximum current waiting data queue size registered
  size_t maxQueueSize() { return this->maxQueueSize_; }

  /// @brief Add a CoreSender to this
  /// @param sender CoreSender to add
  void addSender(CoreSender <NodeInput> *sender) final {
    HLOG_SELF(0, "Adding sender " << sender->name() << "(" << sender->id() << ")")
    this->senders_->insert(sender);
  }

  /// @brief Remove a CoreSender from this
  /// @param sender CoreSender to remove
  void removeSender(CoreSender <NodeInput> *sender) final {
    HLOG_SELF(0, "Remove sender " << sender->name() << "(" << sender->id() << ")")
    this->senders_->erase(sender);
  }

  /// @brief Receive a data from a CoreQueueSender, and store it into the waiting queue
  /// @attention Thread safe
  /// @param data Data to store into the queue
  void receive(std::shared_ptr<NodeInput> data) final {
    this->queueSlot()->lockUniqueMutex();
    this->queue_->push(data);
    HLOG_SELF(2, "Receives data new queue Size " << this->queueSize())
    if (this->queueSize() > this->maxQueueSize_) { this->maxQueueSize_ = this->queueSize(); }
    this->queueSlot()->unlockUniqueMutex();
  }

  /// @brief Test emptiness on the queue
  /// @attention Not thread safe
  /// @return  True if the queue is empty, else False
  bool receiverEmpty() final {
    HLOG_SELF(2, "Test queue emptiness")
    return this->queue_->empty();
  }

  /// @brief Receivers accessor
  /// @return {this}
  std::set<CoreReceiver < NodeInput> *>
  receivers() override { return {this}; }

  /// @brief Return the front element of the queue and return it
  /// @attention Not thread safe
  /// @return The queue front element
  std::shared_ptr<NodeInput> popFront() {
    HLOG_SELF(2, "Pop & front from queue")
    auto element = queue_->front();
    assert(element != nullptr);
    queue_->pop();
    return element;
  }

  // Suppress wrong static analysis  Used but syntax analysis didn't find it out
  /// @brief Copy the CoreQueueReceiver inner structure (queue and senders list) from rhs to this
  /// @param rhs CoreQueueReceiver to copy
  void copyInnerStructure(CoreQueueReceiver<NodeInput> *rhs) {
    HLOG_SELF(0, "Copy Cluster CoreQueueReceiver information from " << rhs->name() << "(" << rhs->id() << ")")
    this->queue_ = rhs->queue_;
    this->senders_ = rhs->senders_;
  }
};

}
#endif //HEDGEHOG_CORE_QUEUE_RECEIVER_H
