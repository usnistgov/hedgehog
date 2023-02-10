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



#ifndef HEDGEHOG_RECEIVER_ABSTRACTION_H
#define HEDGEHOG_RECEIVER_ABSTRACTION_H

#include <memory>
#include <utility>
#include <ostream>

#include "sender_abstraction.h"
#include "../any_groupable_abstraction.h"
#include "../node/node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog implementor namespace
namespace implementor {
/// @brief Implementor to Receiver
/// @tparam Input Type received by implementor
template<class Input>
class ImplementorReceiver;
}
#endif //DOXYGEN_SHOULD_SKIP_THIS


/// @brief Hedgehog abstraction namespace
namespace abstraction {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Abstraction of the core sender interface
/// @tparam Output Output type sent by the interface
template<class Output>
class SenderAbstraction;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Core's abstraction to receive a piece of data
/// @tparam Input Type of data received by the abstraction
template<class Input>
class ReceiverAbstraction {
 private:
  std::shared_ptr<implementor::ImplementorReceiver<Input>>
      concreteReceiver_ = nullptr; ///< Concrete implementation of the interface

  std::shared_ptr<std::mutex> const
      slotMutex_ = nullptr; ///< Mutex shared from the slot abstraction

 public:
  /// @brief Constructor using a concrete implementation of a receiver implementor, and the mutex from the slot
  /// @param concreteReceiver Concrete implementation of the ImplementorReceiver
  /// @param slotMutex Mutex from the slot
  explicit ReceiverAbstraction(
      std::shared_ptr<implementor::ImplementorReceiver<Input>> concreteReceiver,
      std::shared_ptr<std::mutex> slotMutex) :
      concreteReceiver_(std::move(concreteReceiver)),
      slotMutex_(std::move(slotMutex)) {
    concreteReceiver_->initialize(this);
  }

  /// @brief Default destructor
  virtual ~ReceiverAbstraction() = default;

  /// Const accessor to receivers
  /// @brief Present the receivers linked to this abstraction, usually one, may be multiple for the graph presenting all
  /// of the input node receivers
  /// @return Const reference to receivers
  [[nodiscard]] std::set<ReceiverAbstraction *> const &receivers() const { return concreteReceiver_->receivers(); }

  /// Accessor to receivers
  /// @brief Present the receivers linked to this abstraction, usually one, may be multiple for the graph presenting all
  /// of the input node receivers
  /// @return Reference to receivers
  [[nodiscard]] std::set<ReceiverAbstraction *> &receivers() { return concreteReceiver_->receivers(); }

  /// @brief Accessor to the senders attached to this receiver
  /// @return The SenderAbstraction attached to this receiver
  [[nodiscard]] std::set<SenderAbstraction<Input> *> const &connectedSenders() const {
    return concreteReceiver_->connectedSenders();
  }

  /// @brief Receive a piece of data
  /// @details Receive a piece of data and transmit it to the concrete receiver implementation. The call to the concrete
  /// receiver is protected via the mutex
  /// @param inputData Data to transmit to the implementation
  void receive(std::shared_ptr<Input> const inputData) {
    slotMutex_->lock();
    concreteReceiver_->receive(inputData);
    slotMutex_->unlock();
  }

  /// @brief Get an input data from the concrete receiver implementation
  /// @return An input data
  std::shared_ptr<Input> getInputData() { return concreteReceiver_->getInputData(); }

  /// @brief Accessor to the current number of input data received and waiting to be processed
  /// @return The current number of input data received and waiting to be processed
  [[nodiscard]] size_t numberElementsReceived() const { return concreteReceiver_->numberElementsReceived(); }

  /// @brief Accessor to the maximum number of input data received and waiting to be processed
  /// @return The maximum number of input data received and waiting to be processed
  [[nodiscard]] size_t maxNumberElementsReceived() const { return concreteReceiver_->maxNumberElementsReceived(); }

  /// @brief Test if the receiver is empty
  /// @return True if the receiver is empty, else false
  [[nodiscard]] bool empty() const { return concreteReceiver_->empty(); }

  /// @brief Add a SenderAbstraction to the concrete receiver implementation
  /// @param sender SenderAbstraction to add
  void addSender(SenderAbstraction<Input> *const sender) { concreteReceiver_->addSender(sender); }

  /// @brief Remove a SenderAbstraction to the concrete receiver implementation
  /// @param sender SenderAbstraction to remove
  void removeSender(SenderAbstraction<Input> *const sender) { concreteReceiver_->removeSender(sender); }

  /// @brief Add to the printer the edge information
  /// @param printer Printer used to gather information
  /// @throw std::runtime_error If the current node is not a NodeAbstraction, if a sender is not a NodeAbstraction
  void printEdgeInformation(Printer *printer) {
    auto nodeReceiver = dynamic_cast<NodeAbstraction const *>(this);
    if (nodeReceiver == nullptr) {
      throw std::runtime_error("To print an edge, a receiver should be a NodeAbstraction.");
    }

    for (auto receiver : ReceiverAbstraction<Input>::receivers()) {
      for (SenderAbstraction<Input> *connectedSender : receiver->connectedSenders()) {
        for (auto &s : connectedSender->senders()) {
          auto nodeSender = dynamic_cast<NodeAbstraction const *>(s);
          if (nodeSender == nullptr) {
            throw std::runtime_error("To print an edge, a sender should be a NodeAbstraction.");
          }
          printer->printEdge(
              nodeSender, nodeReceiver,
              tool::typeToStr<Input>(),
              receiver->numberElementsReceived(), receiver->maxNumberElementsReceived());
        }
      }
    }
  }

 protected:
  /// @brief Copy inner structure of the receiver to this one
  /// @param copyableCore ReceiverAbstraction to copy into this
  void copyInnerStructure(ReceiverAbstraction<Input> *copyableCore) {
    this->concreteReceiver_ = copyableCore->concreteReceiver_;
  }
};
}
}
}
#endif //HEDGEHOG_RECEIVER_ABSTRACTION_H
