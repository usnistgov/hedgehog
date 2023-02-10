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



#ifndef HEDGEHOG_IMPLEMENTOR_SENDER_H
#define HEDGEHOG_IMPLEMENTOR_SENDER_H

#include <memory>
#include <set>

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of ImplementorSender
/// @tparam Input Input data type
template<class Input>
class ImplementorReceiver;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Implementor for the SenderAbstraction
/// @tparam Output Type of output data
template<class Output>
class ImplementorSender {
 protected:
  std::unique_ptr<std::set<abstraction::SenderAbstraction<Output> *>>
      abstractSenders_ = nullptr; ///< Linked SenderAbstraction

 public:
  /// @brief Default constructor
  explicit ImplementorSender()
      : abstractSenders_(std::make_unique<std::set<abstraction::SenderAbstraction<Output> *>>()) {}

  /// @brief Default destructor
  virtual ~ImplementorSender() = default;

  /// @brief Accessor to the SenderAbstraction
  /// @return Set of SenderAbstraction
  [[nodiscard]] std::set<abstraction::SenderAbstraction<Output> *> &senders() { return *abstractSenders_; }

  /// @brief Initialize the implementor Sender by setting the corresponding abstraction
  /// @param senderAbstraction Sending abstraction to set
  virtual void initialize(abstraction::SenderAbstraction<Output> *senderAbstraction) {
    abstractSenders_->insert(senderAbstraction);
  }

  /// @brief Accessor to the connected receivers
  /// @return Set of connected receivers
  [[nodiscard]] virtual std::set<abstraction::ReceiverAbstraction<Output> *> const &connectedReceivers() const = 0;

  /// @brief Add a receiver to the set of connected receivers
  /// @param receiver Receiver to add
  virtual void addReceiver(abstraction::ReceiverAbstraction<Output> *receiver) = 0;

  /// @brief Remove a receiver to the set of connected receivers
  /// @param receiver Receiver to remove
  virtual void removeReceiver(abstraction::ReceiverAbstraction<Output> *receiver) = 0;

  /// @brief Send a data to successor node
  /// @param data Data to send
  virtual void send(std::shared_ptr<Output> data) = 0;
};
}
}
}

#endif //HEDGEHOG_IMPLEMENTOR_SENDER_H
