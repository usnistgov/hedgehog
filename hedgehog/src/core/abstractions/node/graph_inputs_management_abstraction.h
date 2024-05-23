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

#ifndef HEDGEHOG_GRAPH_INPUTS_MANAGEMENT_ABSTRACTION_H
#define HEDGEHOG_GRAPH_INPUTS_MANAGEMENT_ABSTRACTION_H

#include <sstream>
#include <ostream>

#include "../../implementors/concrete_implementor/graph/graph_receiver.h"
#include "../../implementors/concrete_implementor/graph/graph_slot.h"

#include "../../parts/graph_source.h"

#include "../base/input_output/slot_abstraction.h"
#include "../base/input_output/sender_abstraction.h"
#include "../base/input_output/notifier_abstraction.h"
#include "../base/input_output/receiver_abstraction.h"

#include "../../../tools/concepts.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Input management abstraction for the graph
/// @tparam Inputs Types of input data
template<class ...Inputs>
class GraphInputsManagementAbstraction :
    public SlotAbstraction,
    public ReceiverAbstraction<Inputs> ... {
 private:
  std::unique_ptr<GraphSource < Inputs...>> source_ = nullptr; ///< Graph's source

 public:
  using inputs_t = std::tuple<Inputs...>; ///< Accessor to the graph's inputs

  /// @brief Default constructor
  GraphInputsManagementAbstraction() :
      SlotAbstraction(
          std::static_pointer_cast<implementor::ImplementorSlot>(std::make_shared<implementor::GraphSlot>())
      ),
      ReceiverAbstraction<Inputs>(std::make_shared<implementor::GraphReceiver<Inputs>>())...,
      source_(std::make_unique<GraphSource < Inputs...>>()) {

  }

  /// @brief Default destructor
  ~GraphInputsManagementAbstraction() override = default;

  /// @brief Wait for the graph, should not be called a graph does not wait !
  /// @return nothing, throw in all cases
  /// @throw std::runtime_error Il all cases, a graph can not wait
  bool wait() { throw std::runtime_error("A graph can not wait"); }

 protected:
  /// @brief Source accessor
  /// @return Graph's source
  std::unique_ptr<GraphSource < Inputs...>> const &
  source() const { return source_; }

  /// @brief Terminate a source, notify all input nodes to terminate
  void terminateSource() { this->source_->notifyAllTerminated(); }

  /// @brief Add an input node to a graph
  /// @tparam InputDataType Input data type
  /// @param inputNode Node to add as input of the graph for the type InputDataType
  template<class InputDataType>
  void addInputNodeToGraph(NodeAbstraction *const inputNode) {
    static_assert(
        tool::isContainedIn_v<InputDataType, Inputs...>,
        "The input type (InputDataType) should be part of the list of input types (Inputs)");
    auto inputAsSlot = dynamic_cast<SlotAbstraction *>(inputNode);
    auto inputAsReceiver = dynamic_cast<ReceiverAbstraction<InputDataType> *>(inputNode);
    assert(inputAsSlot != nullptr && inputAsReceiver != nullptr);

    if (source_ != nullptr) {
      auto sourceAsSender = static_cast<SenderAbstraction<InputDataType> *> (source_.get());
      auto sourceAsNotifier = static_cast<NotifierAbstraction *> (source_.get());

      for (auto sender : sourceAsSender->senders()) {
        for (auto receiver : inputAsReceiver->receivers()) {
          sender->addReceiver(receiver);
          receiver->addSender(sender);
        }
      }

      for (auto slot : inputAsSlot->slots()) {
        for (auto notifier : sourceAsNotifier->notifiers()) {
          slot->addNotifier(notifier);
          notifier->addSlot(slot);
        }
      }

    }
  }

  /// @brief Disconnect the source when the graph is set as inside
  void disconnectSource() {
    if (source_) {
      auto sourceAsNotifier = static_cast<NotifierAbstraction *> (source_.get());
      for (auto notifier : sourceAsNotifier->notifiers()) {
        for (auto inputSlot : notifier->connectedSlots()) {
          for (auto slot : inputSlot->slots()) {
            this->slots().insert(slot);
            slot->removeNotifier(notifier);
          }
        }
      }
      (disconnectSourceFromReceiver<Inputs>(), ...);
      source_ = nullptr;
    }
  }

  /// @brief Send a data to the source
  /// @tparam Input Type of the input data
  /// @param data Data of type Input
  template<class Input>
  void sendInputDataToSource(std::shared_ptr<Input> data) { source_->sendAndNotifyAllInputs(data); }

  /// @brief Gather source information
  /// @param printer Visitor printer gathering source information
  void printSource(Printer *printer) {
    if (this->source_) {
      this->source_->print(printer);
    }
  }

  /// @brief Duplicate source edges
  /// @param rhs Graph input to copy the source edges from
  /// @param mapping Map from node to its clone
  void duplicateSourceEdges(GraphInputsManagementAbstraction<Inputs...> const &rhs,
                            std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping) {
    (duplicateSourceEdge<Inputs>(rhs, mapping), ...);
  }

 private:
  /// @brief So nothing and should not be called
  /// @throw std::runtime_error because the graph is not attached to a thread
  /// @return Nothing throw an error
  [[nodiscard]] bool waitTerminationCondition() override {
    throw std::runtime_error("A graph is not attached to a thread it cannot wait for termination.");
  }

  /// @brief So nothing and should not be called
  /// @throw std::runtime_error because the graph is not attached to a thread
  /// @return Nothing throw an error
  [[nodiscard]] bool canTerminate() override {
    throw std::runtime_error("A graph is not attached to a thread it cannot terminate.");
  }
  /// @brief Disconnect the source from the input node receivers
  /// @tparam Input Type of input node receivers to disconnect from the source
  template<class Input>
  void disconnectSourceFromReceiver() {
    auto sourceAsSender = static_cast<SenderAbstraction<Input> *>(source_.get());
    for (auto &sender : sourceAsSender->senders()) {
      for (auto &inputReceiver : sender->connectedReceivers()) {
        for (auto &receiver : inputReceiver->receivers()) {
          ReceiverAbstraction<Input>::receivers().insert(receiver);
          receiver->removeSender(sender);
        }
      }
    }
  }

  /// @brief Duplicate source edge for a type
  /// @tparam Input Type of input node receivers to duplicate
  /// @param rhs Graph input to copy the source edges from
  /// @param correspondenceMap Map from node to its clone
  template<class Input>
  void duplicateSourceEdge(
      GraphInputsManagementAbstraction<Inputs...> const &rhs,
      std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &correspondenceMap) {
    for (auto &receiver : ((ReceiverAbstraction<Input> *) &rhs)->receivers()) {
      if (auto receiverAsClonable = dynamic_cast<abstraction::ClonableAbstraction *>(receiver)) {
        if (auto receiverAsNode = dynamic_cast<abstraction::NodeAbstraction *>(receiver)) {
          if (!correspondenceMap.contains(receiverAsNode)) {
            auto duplicate = receiverAsClonable->clone(correspondenceMap);
            dynamic_cast<hh::core::abstraction::GraphNodeAbstraction *>(this)->registerNodeInsideGraph(duplicate.get());
            correspondenceMap.insert({receiverAsNode, duplicate});
          }

          auto inputNode = correspondenceMap.at(receiverAsNode).get();

          auto inputAsSlot = dynamic_cast<SlotAbstraction *>(inputNode);
          auto inputAsReceiver = dynamic_cast<ReceiverAbstraction<Input> *>(inputNode);

          assert(inputAsSlot != nullptr && inputAsReceiver != nullptr);

          for (auto &inputSlot : inputAsSlot->slots()) {
            this->slots().insert(inputSlot);
          }

          for (auto &inputReceiver : inputAsReceiver->receivers()) {
            ReceiverAbstraction<Input>::receivers().insert(inputReceiver);
          }

        } else {
          throw std::runtime_error("An input node core is not a NodeAbstraction.");
        }
      }
    }
  }
};
}
}
}

#endif //HEDGEHOG_GRAPH_INPUTS_MANAGEMENT_ABSTRACTION_H
