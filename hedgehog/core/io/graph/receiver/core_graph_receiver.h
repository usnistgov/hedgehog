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


#ifndef HEDGEHOG_CORE_GRAPH_RECEIVER_H
#define HEDGEHOG_CORE_GRAPH_RECEIVER_H

#include "../../base/receiver/core_receiver.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Graph Receiver for a type GraphInput
/// @tparam GraphInput Type received by the graph
template<class GraphInput>
class CoreGraphReceiver : public virtual CoreReceiver<GraphInput> {
 private:
  std::unique_ptr<std::set<CoreReceiver < GraphInput> *>> graphReceiverInputs_ = nullptr; ///< Graph receivers
  ///< (Input node receivers)

 public:
  /// @brief CoreGraphReceiver constructor
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Node number of threads
  CoreGraphReceiver(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads),
      CoreReceiver<GraphInput>(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreGraphReceiver with type: " << (int) type << " and name: " << name)
    this->graphReceiverInputs_ = std::make_unique<std::set<CoreReceiver<GraphInput> *>>();
  }

  /// @brief CoreGraphReceiver destructor
  virtual ~CoreGraphReceiver() {
    HLOG_SELF(0, "Destructing CoreGraphReceiver")
  }

  /// @brief Add a CoreSender to the graph
  /// @details Add a CoreSender to all input nodes
  /// @param sender CoreSender to add to the graph
  void addSender(CoreSender <GraphInput> *sender) final {
    HLOG_SELF(0, "Add sender " << sender->name() << "(" << sender->id() << ")")
    for (CoreReceiver<GraphInput> *inputNode: *(this->graphReceiverInputs_)) {
      inputNode->addSender(sender);
    }
  }

  /// @brief Remove a CoreSender from the graph
  /// @details CoreSender to remove from all input nodes
  /// @param sender CoreSender to remove from the graph
  void removeSender(CoreSender <GraphInput> *sender) final {
    HLOG_SELF(0, "Remove sender " << sender->name() << "(" << sender->id() << ")")
    for (CoreReceiver<GraphInput> *inputNode: *(this->graphReceiverInputs_)) {
      inputNode->removeSender(sender);
    }
  }

  /// @brief Define how the graph receives data for a specific type and sends the data to all input nodes
  /// @param ptr Data received by the graph
  void receive(std::shared_ptr<GraphInput> ptr) final {
    HLOG_SELF(2, "Receive data")
    for (CoreReceiver<GraphInput> *inputNode: *(this->graphReceiverInputs_)) { inputNode->receive(ptr); }
  }

  /// @brief Register a CoreReceiver from an input node
  /// @param receiver CoreReceiver from an input node to register
  void addGraphReceiverInput(CoreReceiver <GraphInput> *receiver) {
    HLOG_SELF(0, "Add Graph Receiver Input " << receiver->name() << "(" << receiver->id() << ")")
    this->graphReceiverInputs_->insert(receiver);
  }

  /// @brief Test emptiness in all graph receivers
  /// @return True if all graph receivers are empty, else False
  bool receiverEmpty() final {
    auto result = true;
    for (CoreReceiver<GraphInput> *inputNode: *(this->graphReceiverInputs_)) {
      result &= inputNode->receiverEmpty();
    }
    HLOG_SELF(2, "Test receiver empty " << "(" << result << ")")
    return result;
  }

  /// @brief Get a set of CoreReceiver built from the input nodes
  /// @return Set of CoreReceiver build from the input nodes
  std::set<CoreReceiver < GraphInput> *>
  receivers() override {
    std::set<CoreReceiver<GraphInput> *> receivers{};
    std::set<CoreReceiver<GraphInput> *> tempReceivers;
    for (CoreReceiver<GraphInput> *receiver : *(this->graphReceiverInputs_)) {
      tempReceivers = receiver->receivers();
      receivers.insert(tempReceivers.begin(), tempReceivers.end());
    }
    return receivers;
  }

  /// @brief Get a node from the graph, that does not exist, throw an error in every case
  /// @exception std::runtime_error A graph does not have node
  /// @return nothing, fail with a std::runtime_error
  [[noreturn]] behavior::Node *node() override {
    std::ostringstream oss;
    oss << "Internal error, should not be called, graph does not have a node: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }
};

}
#endif //HEDGEHOG_CORE_GRAPH_RECEIVER_H
