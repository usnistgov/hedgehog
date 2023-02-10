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



#ifndef HEDGEHOG_IMPLEMENTOR_RECEIVER_H
#define HEDGEHOG_IMPLEMENTOR_RECEIVER_H

#include <set>

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of ImplementorSender
/// @tparam Output Data type sent
template<class Output>
class ImplementorSender;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Implementor for the ReceiverAbstraction
/// @tparam Input Data type received
template<class Input>
class ImplementorReceiver {
 protected:
  std::unique_ptr<std::set<abstraction::ReceiverAbstraction<Input> *>>
      abstractReceivers_ = nullptr; ///< Set of linked ReceiverAbstraction

 public:
  /// @brief Default constructor
  explicit ImplementorReceiver() :
      abstractReceivers_(std::make_unique<std::set<abstraction::ReceiverAbstraction<Input> *>>()) {}

  /// @brief Default destructor
  virtual ~ImplementorReceiver() = default;

  /// @brief Accessor to the linked ReceiverAbstraction
  /// @return Set of ReceiverAbstraction
  [[nodiscard]] std::set<abstraction::ReceiverAbstraction<Input> *> &receivers() { return *abstractReceivers_; }

  /// @brief Initialize the implementor Receiver by setting the corresponding abstraction
  /// @param receiverAbstraction Receiver abstraction to set
  virtual void initialize(abstraction::ReceiverAbstraction<Input> *receiverAbstraction) {
    abstractReceivers_->insert(receiverAbstraction);
  }

  /// @brief Accessor to the connected senders
  /// @return A set of connected sender
  [[nodiscard]] virtual std::set<abstraction::SenderAbstraction<Input> *> const &connectedSenders() const = 0;

  /// @brief Add a SenderAbstraction
  /// @param sender SenderAbstraction to add
  virtual void addSender(abstraction::SenderAbstraction<Input> *sender) = 0;

  /// @brief Remove a SenderAbstraction
  /// @param sender SenderAbstraction to remove
  virtual void removeSender(abstraction::SenderAbstraction<Input> *sender) = 0;

  /// @brief Receive data interface
  /// @param data Data received by the core
  virtual void receive(std::shared_ptr<Input> data) = 0;

  /// @brief Accessor to a received data
  /// @return Data previously received
  [[nodiscard]] virtual std::shared_ptr<Input> getInputData() = 0;

  /// @brief Accessor to number of data waiting to be processed in the queue
  /// @return Number of data waiting to be processed in the queue
  [[nodiscard]] virtual size_t numberElementsReceived() const = 0;

  /// @brief Accessor to the maximum number of data waiting to be processed in the queue during the whole execution
  /// @return Maximum number of data waiting to be processed in the queue during the whole execution
  [[nodiscard]] virtual size_t maxNumberElementsReceived() const = 0;

  /// @brief Test if the receiver is empty or not
  /// @return True if the receiver is empty, else false
  [[nodiscard]] virtual bool empty() const = 0;
};
}
}
}
#endif //HEDGEHOG_IMPLEMENTOR_RECEIVER_H
