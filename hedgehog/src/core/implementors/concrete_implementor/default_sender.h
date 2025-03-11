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

#ifndef HEDGEHOG_DEFAULT_SENDER_H
#define HEDGEHOG_DEFAULT_SENDER_H

#include <execution>
#include "../../../tools/intrinsics.h"
#include "../implementor/implementor_sender.h"
#include "../implementor/implementor_receiver.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of sender abstraction
/// @tparam Output Data type to send
template<class Output>
class DefaultSender : public ImplementorSender<Output> {
 private:
  std::unique_ptr<std::set<abstraction::ReceiverAbstraction<Output> *>> const
      receivers_{}; ///< List of receivers

  std::mutex mutex_{}; ///< Mutex used to protect the list of receivers

 public:
  /// @brief Default constructor
  explicit DefaultSender()
      : receivers_(std::make_unique<std::set<abstraction::ReceiverAbstraction<Output> *>>()) {}

  /// @brief Default destructor
  virtual ~DefaultSender() = default;

  /// @brief Accessor to the connected receivers
  /// @return Connected receivers
  [[nodiscard]] std::set<abstraction::ReceiverAbstraction<Output> *> const &connectedReceivers() const override {
    return *receivers_;
  }

  /// @brief Add a receiver to the list
  /// @param receiver Receiver to add
  void addReceiver(abstraction::ReceiverAbstraction<Output> *receiver) override {
    std::lock_guard<std::mutex> lck(mutex_);
    receivers_->insert(receiver);
  }

  /// @brief Remove a receiver to the list
  /// @param receiver Receiver to remove
  void removeReceiver(abstraction::ReceiverAbstraction<Output> *receiver) override {
    std::lock_guard<std::mutex> lck(mutex_);
    receivers_->erase(receiver);
  }

  /// @brief Send a piece of data to all connected receivers
  /// @param data Data to send
  void send(std::shared_ptr<Output> data) override {
    std::lock_guard<std::mutex> lck(mutex_);
    for (auto const &receiver : *receivers_) {
      while(!receiver->receive(data)){ cross_platform_yield(); }
    }
  }
};
}
}
}

#endif //HEDGEHOG_DEFAULT_SENDER_H
