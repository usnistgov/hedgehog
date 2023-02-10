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



#ifndef HEDGEHOG_SENDER_ABSTRACTION_H
#define HEDGEHOG_SENDER_ABSTRACTION_H

#include <memory>
#include <utility>
#include <ostream>

#include "../../../implementors/implementor/implementor_notifier.h"
#include "../../../implementors/implementor/implementor_sender.h"
#include "../clonable_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

/// @brief Hedgehog abstraction namespace
namespace abstraction {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class Input>
class ReceiverAbstraction;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Core abstraction to send data
/// @tparam Output Data type to send
template<class Output>
class SenderAbstraction {
 private:
  std::shared_ptr<implementor::ImplementorSender<Output>>
      concreteSender_ = nullptr; ///< Concrete implementation of the sender used in the node

 public:
  /// @brief Constructor using the concrete implementation
  /// @param concreteSender Sender concrete implementation
  explicit SenderAbstraction(std::shared_ptr<implementor::ImplementorSender<Output>> concreteSender)
  : concreteSender_(std::move(concreteSender)) {
    concreteSender_->initialize(this);
  }

  /// @brief Default destructor
  virtual ~SenderAbstraction() = default;

  /// Const accessor to senders
  /// @brief Present the senders linked to this abstraction, usually one, may be multiple for the graph presenting all
  /// of the output node senders
  /// @return Const reference to senders
  [[nodiscard]] std::set<SenderAbstraction<Output> *> const &senders() const { return concreteSender_->senders(); }

  /// Accessor to senders
  /// @brief Present the senders linked to this abstraction, usually one, may be multiple for the graph presenting all
  /// of the output node senders
  /// @return Reference to senders
  [[nodiscard]] std::set<SenderAbstraction<Output> *> &senders() { return concreteSender_->senders(); }

  /// @brief Accessor to the receivers attached to this SenderAbstraction
  /// @return The ReceiverAbstraction attached to this SenderAbstraction
  [[nodiscard]] std::set<ReceiverAbstraction<Output> *> const &connectedReceivers() const {
    return concreteSender_->connectedReceivers();
  }

  /// @brief Add a ReceiverAbstraction
  /// @param receiver ReceiverAbstraction to add
  void addReceiver(ReceiverAbstraction<Output> *const receiver) { concreteSender_->addReceiver(receiver); }

  /// @brief Remove a ReceiverAbstraction
  /// @param receiver ReceiverAbstraction to remove
  void removeReceiver(ReceiverAbstraction<Output> *const receiver) { concreteSender_->removeReceiver(receiver); }

  /// @brief Send a data as output of the node
  /// @param data Data to send to successor node or output of the graph
  void send(std::shared_ptr<Output> data) { concreteSender_->send(data); }

 protected:
  /// @brief Copy inner structure of the sender to this one
  /// @param copyableCore SenderAbstraction to copy into this
  void copyInnerStructure(SenderAbstraction<Output> *copyableCore) {
    this->concreteSender_ = copyableCore->concreteSender_;
  }

  /// @brief Duplicate edges of the current sender to receiver to clone in map
  /// @param mapping Map of the nodes -> clone
  /// @throw throw std::runtime_error if the current node is not mapped to its clone, if the clone is not a
  /// ReceiverAbstraction, if a slot is not a node
  void duplicateEdgeSender(std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping) {
    std::shared_ptr<NodeAbstraction> duplicateReceiver;
    auto senderAsNode = dynamic_cast<abstraction::NodeAbstraction *>(this);
    if (!mapping.contains(senderAsNode)) {
      throw std::runtime_error("A node that we are trying to connect is not mapped yet.");
    }
    auto mappedSender = std::dynamic_pointer_cast<SenderAbstraction < Output>>
    (mapping.at(senderAsNode));
    if (mappedSender == nullptr) {
      std::ostringstream oss;
      oss << "The mapped type of a node is not of the right type: Sender<" << hh::tool::typeToStr<Output>() << ">.";
      throw std::runtime_error(oss.str());
    }

    for (auto &sender : this->senders()) {
      for (auto &receiver : sender->connectedReceivers()) {
        for (auto &r : receiver->receivers()) {
          if (auto receiverAsNode = dynamic_cast<abstraction::NodeAbstraction *>(r)) {
            if (mapping.contains(receiverAsNode)) {
              auto mappedReceiver = std::dynamic_pointer_cast<ReceiverAbstraction<Output>>(mapping.at(receiverAsNode));

              if (mappedReceiver == nullptr) {
                std::ostringstream oss;
                oss
                    << "The mapped type of a node is not of the right type: ReceiverAbstraction<"
                    << hh::tool::typeToStr<Output>() << ">.";
                throw std::runtime_error(oss.str());
              }

              mappedSender->addReceiver(mappedReceiver.get());
              mappedReceiver->addSender(mappedSender.get());
            }
          } else {
            throw std::runtime_error("A receiver is not a node when duplicating edges.");
          }
        }
      }

    }

  }
};
}
}
}
#endif //HEDGEHOG_SENDER_ABSTRACTION_H
