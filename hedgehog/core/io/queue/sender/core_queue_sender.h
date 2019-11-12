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


#ifndef HEDGEHOG_CORE_QUEUE_SENDER_H
#define HEDGEHOG_CORE_QUEUE_SENDER_H

#include "core_queue_notifier.h"
#include "../receiver/core_queue_receiver.h"
#include "../../base/sender/core_sender.h"
#include "../../../../tools/traits.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Sender for nodes possessing a queue of data
/// @tparam NodeOutput Node output type
template<class NodeOutput>
class CoreQueueSender : public CoreSender<NodeOutput>, public virtual CoreQueueNotifier {
  std::shared_ptr<std::set<CoreQueueReceiver<NodeOutput> *>> destinations_ = nullptr; ///< Set of receivers linked

 public:
  /// @brief CoreQueueSender constructor
  /// @param name Node's name
  /// @param type Node's type
  /// @param numberThreads Node's number of thread
  CoreQueueSender(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreQueueNotifier(name, type, numberThreads),
        CoreSender<NodeOutput>(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueSender with type: " << (int) type << " and name: " << name)
    destinations_ = std::make_shared<std::set<CoreQueueReceiver<NodeOutput> *>>();
  }

  /// @brief CoreQueueSender destructor
  ~CoreQueueSender() override {HLOG_SELF(0, "Destructing CoreQueueSender")}

  /// @brief Destination accessor
  /// @return Set of CoreQueueReceiver
  virtual std::shared_ptr<std::set<CoreQueueReceiver<NodeOutput> *>> const &destinations() const {
    return destinations_;
  }

  /// @brief Add a receiver to the set of receivers
  /// @param receiver CoreReceiver to add
  void addReceiver(CoreReceiver<NodeOutput> *receiver) override {
    HLOG_SELF(0, "Add receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (auto queueReceiver: receiver->receivers()) {
      if (auto r = dynamic_cast<CoreQueueReceiver<NodeOutput> *>(queueReceiver)) {
        this->destinations_->insert(r);
      } else {
        std::ostringstream oss;
        oss
            << "Internal error, CoreQueueSender connected to a node which is not a CoreQueueReceiver: "
            << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }
  }

  /// @brief Remove a receiver from the set of receivers
  /// @param receiver CoreReceiver to remove
  void removeReceiver(CoreReceiver<NodeOutput> *receiver) override {
    HLOG_SELF(0, "Remove receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (auto queueReceiver: receiver->receivers()) {
      if (auto r = dynamic_cast<CoreQueueReceiver<NodeOutput> *>(queueReceiver)) {
        this->destinations_->erase(r);
      } else {
        std::ostringstream oss;
        oss
            << "Internal error, CoreQueueSender connected to a node which is not a CoreQueueReceiver: "
            << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }
  }

  /// @brief Send a data to the list of destinations, and notify them
  /// @param ptr Data send to all CoreQueueReceiver
  void sendAndNotify(std::shared_ptr<NodeOutput> ptr) final {
    for (CoreQueueReceiver<NodeOutput> *receiver : *(this->destinations_)) {
      HLOG_SELF(2, "Send data to " << receiver->name() << "(" << receiver->id() << ")")
      receiver->receive(ptr);
      HLOG_SELF(2, "Wake up " << receiver->name() << "(" << receiver->id() << ")")
      receiver->queueSlot()->wakeUp();
    }
  }

  /// @brief Special visit method for a CoreQueueSender, printing an edge
  /// @param printer Printer visitor to print the CoreQueueSender
  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    for (CoreQueueReceiver<NodeOutput> *receiver : *(this->destinations())) {
      if (receiver->type() != NodeType::Switch || receiver->type() != NodeType::ExecutionPipeline) {
        printer->printEdge(this,
                           receiver,
                           traits::type_name<NodeOutput>(),
                           receiver->queueSize(),
                           receiver->maxQueueSize(),
                           traits::is_managed_memory_v<NodeOutput>);
      }
    }
  }

  /// @brief Get inner CoreSender i.e. this
  /// @return This as CoreSender
  std::set<CoreSender<NodeOutput> *> getSenders() override { return {this}; }

  /// @brief Copy the inner structure of a CoreQueueSender (destinations, and notifier)
  /// @param rhs CoreQueueSender to copy to this
  void copyInnerStructure(CoreQueueSender<NodeOutput> *rhs) {
    HLOG_SELF(0, "Copy Cluster CoreQueueSender information from " << rhs->name() << "(" << rhs->id() << ")")
    this->destinations_ = rhs->destinations_;
    CoreQueueNotifier::copyInnerStructure(rhs);
  }

 protected:
  /// @brief Duplicate all the edges from this to it's copy duplicateNode
  /// @param duplicateNode Node to connect
  /// @param correspondenceMap Correspondence  map from base node to copy
  void duplicateEdge(CoreNode *duplicateNode,
                     std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
    for (CoreQueueReceiver<NodeOutput> *originalReceiver : *(this->destinations())) {
      if (auto coreNode = dynamic_cast<CoreNode *>(originalReceiver)) {
        auto nodeReceiverFound = correspondenceMap.find(coreNode);
        if (nodeReceiverFound != correspondenceMap.end()) {
          if (nodeReceiverFound->second->id() == this->id()) {
            std::ostringstream oss;
            oss
                << "Internal error, Receiver found is the same as the original Receiver, copy failed "
                << __FUNCTION__;
            HLOG_SELF(0, oss.str())
            throw (std::runtime_error(oss.str()));
          }
          connectSenderToReceiverDuplication(duplicateNode, nodeReceiverFound->second.get());
        }
      } else {
        std::ostringstream oss;
        oss
            << "Internal error, The Receiver is not a CoreNode, copy failed "
            << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }
  }

 private:
  /// @brief Connect coreNodeSender to a coreNodeReceiver, connect for data transfer and signal handling
  /// @param coreNodeSender the sender
  /// @param coreNodeReceiver the receiver
  void connectSenderToReceiverDuplication(CoreNode *coreNodeSender, CoreNode *coreNodeReceiver) {
    auto coreReceiver = dynamic_cast<CoreReceiver<NodeOutput> *>(coreNodeReceiver);
    auto coreSlot = dynamic_cast<CoreSlot *>(coreNodeReceiver);
    auto coreNotifier = dynamic_cast<CoreNotifier *>(coreNodeSender);

    // Do the data connection
    if (coreReceiver) {
      for (auto r : coreReceiver->receivers()) {
        dynamic_cast<CoreQueueSender<NodeOutput> *>(coreNodeSender)->addReceiver(r);
      }
    } else {
      std::ostringstream oss;
      oss << "Internal error, during edge duplication" << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }

    //Do the signal connection
    if (coreSlot && coreNotifier) {
      for (CoreSlot *slot : coreSlot->getSlots()) { coreNotifier->addSlot(slot); }
      for (auto s : dynamic_cast<CoreQueueSender<NodeOutput> *>(coreNodeSender)->getSenders()) {
        coreReceiver->addSender(s);
        coreSlot->addNotifier(s);
      }
    } else {
      std::ostringstream oss;
      oss << "Internal error, during edge duplication" << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
  }
};

}
#endif //HEDGEHOG_CORE_QUEUE_SENDER_H
