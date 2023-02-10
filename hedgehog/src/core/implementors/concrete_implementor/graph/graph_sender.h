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

#ifndef HEDGEHOG_GRAPH_SENDER_H
#define HEDGEHOG_GRAPH_SENDER_H

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default concrete implementation of the sender abstraction for the graph core
/// @tparam Output Output type
template<class Output>
class GraphSender : public ImplementorSender<Output> {
 public:

  /// @brief Default constructor
  explicit GraphSender() = default;

  /// @brief Default destructor
  virtual ~GraphSender() = default;

  /// @brief Redefine the implementor to do nothing, the graph do nothing by itself
  /// @param senderAbstraction Abstraction not used
  void initialize([[maybe_unused]]abstraction::SenderAbstraction<Output> *senderAbstraction) override {}

  /// @brief Do nothing, throw an error, a graph is not really connected to other nodes
  /// @throw std::runtime_error A graph has no connected ReceiverAbstraction by itself.
  /// @return Nothing, throw an error
  [[nodiscard]] std::set<abstraction::ReceiverAbstraction < Output> *> const &connectedReceivers() const override {
    throw std::runtime_error("A graph has no connected ReceiverAbstraction by itself.");
  }

  /// @brief Add receiver node to graph's output node
  /// @param receiver Receiver node to add to graph's output node
  void addReceiver(abstraction::ReceiverAbstraction<Output> *receiver) override {
    for (auto sender : *this->abstractSenders_) { sender->addReceiver(receiver); }
  }

  /// @brief Remove receiver node to graph's output node
  /// @param receiver Receiver node to remove to graph's output node
  void removeReceiver(abstraction::ReceiverAbstraction<Output> *receiver) override {
    for (auto sender : *this->abstractSenders_) { sender->removeReceiver(receiver); }
  }

  /// @brief Do nothing, throw an error, a graph is not really connected to other nodes
  /// @param data not used data
  /// @throw std::runtime_error A graph has no connected ReceiverAbstraction by itself.
  void send([[maybe_unused]]std::shared_ptr<Output> data) override {
    throw std::runtime_error("A graph has no connected ReceiverAbstraction by itself");
  }
};
}
}
}
#endif //HEDGEHOG_GRAPH_SENDER_H
