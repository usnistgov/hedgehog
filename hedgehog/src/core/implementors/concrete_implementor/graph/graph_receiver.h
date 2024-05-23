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

#ifndef HEDGEHOG_GRAPH_RECEIVER_H
#define HEDGEHOG_GRAPH_RECEIVER_H

#include <memory>
#include <mutex>
#include <execution>
#include <emmintrin.h>

#include "../../implementor/implementor_receiver.h"
#include "../../../abstractions/base/input_output/receiver_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of the receiver abstraction for the graph core
/// @tparam Input Input type
template<class Input>
class GraphReceiver : public ImplementorReceiver<Input> {
 public:

  /// @brief Default constructor
  explicit GraphReceiver() = default;

  /// @brief Default destructor
  virtual ~GraphReceiver() = default;

  /// @brief Redefine the implementor to do nothing, the graph do nothing by itself
  /// @param receiverAbstraction Abstraction not used
  void initialize([[maybe_unused]]abstraction::ReceiverAbstraction<Input> *receiverAbstraction) override {}

  /// @brief Do nothing, throw an error, a graph does not receive data, its input nodes do
  /// @return Nothing, throw a std::runtime_error
  /// @throw std::runtime_error It is not possible to get the number of input data from the graph receiver as it is only
  /// used to transfer data to input nodes.
  [[nodiscard]] size_t numberElementsReceived() override {
    throw std::runtime_error("It is not possible to get the number of input data from the graph receiver as it is only "
                             "used to transfer data to input nodes.");
  }

  /// @brief Do nothing, throw an error, a graph does not receive data, its input nodes do
  /// @return Nothing, throw a std::runtime_error
  /// @throw std::runtime_error It is not possible to get the number of input data from the graph receiver as it is only
  /// used to transfer data to input nodes.
  [[nodiscard]] size_t maxNumberElementsReceived() const override {
    throw std::runtime_error("It is not possible to get the number of input data from the graph receiver as it is only "
                             "used to transfer data to input nodes.");
  }

  /// @brief Do nothing, throw an error, a graph does not receive data, its input nodes do
  /// @return Nothing, throw a std::runtime_error
  /// @throw std::runtime_error It is not possible to get the number of input data from the graph receiver as it is only
  /// used to transfer data to input nodes.
  [[nodiscard]] bool empty() override {
    throw std::runtime_error("It is not possible to test if there is input data from the graph receiver as it is only "
                             "used to transfer data to input nodes.");
  }

  /// @brief Do nothing, throw an error, a graph does not receive data, its input nodes do
  /// @param data Not used
  /// @return Nothing, throw a std::runtime_error
  /// @throw std::runtime_error It is not possible to get the number of input data from the graph receiver as it is only
  /// used to transfer data to input nodes.
  bool getInputData([[maybe_unused]]std::shared_ptr<Input> &data) override {
    throw std::runtime_error("It is not possible to get input data from the graph receiver as it is only used to "
                             "transfer data to input nodes.");
  }

  /// @brief Do nothing, throw an error, a graph is not really connected to other nodes
  /// @return Nothing, throw a std::runtime_error
  /// @throw std::runtime_error A graph is not connected to any senders
  std::set<abstraction::SenderAbstraction<Input> *> const &connectedSenders() const override {
    throw std::runtime_error("A graph is not connected to any senders");
  }

  /// @brief Receive a data and transmit to its input nodes, wait until it is transmitted to all of its input nodes
  /// @param data Dat ato transmit to input nodes
  /// @return True
  /// @note Returns always true, this receiver send data to all input nodes. If the node cannot receive the data, it is
  /// retried until the data go through.
  bool receive(std::shared_ptr<Input> data) override {
    std::for_each(
        this->abstractReceivers_->begin(), this->abstractReceivers_->end(),
        [&data](abstraction::ReceiverAbstraction<Input> *receiver) {
          while(!receiver->receive(data)) { _mm_pause(); }
        }
    );
    return true;
  }

  /// @brief Add a sender to add to the graph input nodes
  /// @param sender Sender to add
  void addSender(abstraction::SenderAbstraction<Input> *const sender) override {
    std::for_each(
        this->abstractReceivers_->begin(), this->abstractReceivers_->end(),
        [&sender](abstraction::ReceiverAbstraction<Input> *receiver) {  receiver->addSender(sender); }
    );
  }

  /// @brief Remove a sender to add to the graph input nodes
  /// @param sender Sender to remove
  void removeSender(abstraction::SenderAbstraction<Input> *const sender) override {
    std::for_each(
        this->abstractReceivers_->begin(), this->abstractReceivers_->end(),
        [&sender](abstraction::ReceiverAbstraction<Input> *receiver) { receiver->removeSender(sender); }
    );
  }

};
}
}
}

#endif //HEDGEHOG_GRAPH_RECEIVER_H
