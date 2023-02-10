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



#ifndef HEDGEHOG_GRAPH_OUTPUTS_MANAGEMENT_ABSTRACTION_H
#define HEDGEHOG_GRAPH_OUTPUTS_MANAGEMENT_ABSTRACTION_H

#include <ostream>
#include "../../parts/graph_sink.h"

#include "../base/input_output/slot_abstraction.h"
#include "../base/input_output/sender_abstraction.h"
#include "../base/input_output/notifier_abstraction.h"
#include "../base/input_output/receiver_abstraction.h"

#include "../../../tools/concepts.h"
#include "../../implementors/concrete_implementor/graph/graph_notifier.h"
#include "../../implementors/concrete_implementor/graph/graph_sender.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Output management abstraction for the grpah
/// @tparam Outputs Types of output data
template<class ...Outputs>
class GraphOutputsManagementAbstraction :
    public NotifierAbstraction,
    public SenderAbstraction<Outputs> ... {
 private:
  std::unique_ptr<GraphSink < Outputs...>> sink_ = nullptr; ///< Graph's sink

 public:
  using outputs_t = std::tuple<Outputs...>; ///< Accessor to the output types

  /// @brief Default constructor
  GraphOutputsManagementAbstraction() :
      NotifierAbstraction(std::make_shared<implementor::GraphNotifier>()),
      SenderAbstraction<Outputs>(std::make_shared<implementor::GraphSender<Outputs>>())...,
      sink_(std::make_unique<GraphSink < Outputs...>>
  ()) {}

  /// @brief Default destructor
  ~GraphOutputsManagementAbstraction() override = default;

  /// @brief Get a blocking result for the outer graph from the sink
  /// @return An output data
  auto getBlockingResult() { return sink_->getBlockingResult(); }

 protected:
  /// @brief Accessor to the graph's sink
  /// @return The sink
  std::unique_ptr<GraphSink < Outputs...>> const & sink() const { return sink_; }

  /// @brief Add an output node in the graph
  /// @tparam OutputDataType Type used to connect a node
  /// @param outputNode Node to connect as output of the graph for the type OutputDataType
  template<class OutputDataType>
  void addOutputNodeToGraph(NodeAbstraction *const outputNode) {
    auto outputAsNotifier = dynamic_cast<NotifierAbstraction *>(outputNode);
    auto outputAsSender = dynamic_cast<SenderAbstraction<OutputDataType> *>(outputNode);
    assert(outputAsNotifier != nullptr && outputAsSender != nullptr);

    if (sink_ != nullptr) {
      auto sinkAsReceiver = static_cast<ReceiverAbstraction<OutputDataType> *> (sink_.get());
      auto sinkAsSlot = static_cast<SlotAbstraction *> (sink_.get());

      for (auto receiver : sinkAsReceiver->receivers()) {
        for (auto sender : outputAsSender->senders()) {
          sender->addReceiver(receiver);
          receiver->addSender(sender);
        }
      }

      for (auto slot : sinkAsSlot->slots()) {
        for (auto notifier : outputAsNotifier->notifiers()) {
          slot->addNotifier(notifier);
          notifier->addSlot(slot);
        }
      }
    }
  }

  /// @brief Disconnect the sink
  void disconnectSink() {
    if (sink_) {
      auto sinkAsSlot = static_cast<SlotAbstraction *> (sink_.get());
      for (auto slot : sinkAsSlot->slots()) {
        for (auto outputNotifier : sink_->connectedNotifiers()) {
          for (auto notifier : outputNotifier->notifiers()) {
            this->notifiers().insert(notifier);
            notifier->removeSlot(slot);
          }
        }
      }
      (disconnectSinkFromSender<Outputs>(), ...);
      sink_ = nullptr;
    }
  }

  /// @brief Visit the sink
  /// @param printer Printer visits gathering data from the nodes
  void printSink(Printer *printer) {
    if (this->sink_) {
      this->sink_->print(printer);
    }
  }

  /// @brief Duplicate sink edges
  /// @param rhs Graph to copy edges from
  /// @param mapping Map from the nodes and their clones
  void duplicateSinkEdges(GraphOutputsManagementAbstraction<Outputs...> const &rhs,
                          std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping) {
    (duplicateSinkEdge<Outputs>(rhs, mapping), ...);
  }

  /// @brief Duplicate output edges
  /// @param mapping Map from the nodes and their clones
  void duplicateOutputEdges(std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping){
    (SenderAbstraction<Outputs>::duplicateEdgeSender(mapping), ...);
    this->duplicateEdgeNotifier(mapping);
  }

 private:
  /// @brief Disconnect the sink from the graph's output senders
  /// @tparam Output Type to disconnect from
  template<class Output>
  void disconnectSinkFromSender() {
    auto sourceAsReceiver = static_cast<ReceiverAbstraction<Output> *>(sink_.get());
    for (auto receiver : sourceAsReceiver->receivers()) {
      for (auto outputSender : receiver->connectedSenders()) {
        for (auto sender : outputSender->senders()) {
          SenderAbstraction<Output>::senders().insert(sender);
          sender->removeReceiver(receiver);
        }
      }
    }
  }


  /// @brief Duplicate sink edge
  /// @tparam Output Type to duplicate
  /// @param rhs Graph to copy edges from
  /// @param correspondenceMap Map from the nodes and their clones
  template<class Output>
  void duplicateSinkEdge(GraphOutputsManagementAbstraction<Outputs...> const &rhs,
                         std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &correspondenceMap) {
    for (auto &sender : ((SenderAbstraction<Output>*)&rhs)->senders()) {
      if (auto senderAsClonable = dynamic_cast<abstraction::ClonableAbstraction *>(sender)) {
        if (auto senderAsNode = dynamic_cast<abstraction::NodeAbstraction *>(sender)) {
          if (!correspondenceMap.contains(senderAsNode)) {
            auto duplicate = senderAsClonable->clone(correspondenceMap);
            dynamic_cast<hh::core::abstraction::GraphNodeAbstraction *>(this)->registerNodeInsideGraph(duplicate.get());
            correspondenceMap.insert({senderAsNode, duplicate});
          }

          auto outputNode = correspondenceMap.at(senderAsNode).get();

          auto outputAsNotifier = dynamic_cast<NotifierAbstraction *>(outputNode);
          auto outputAsSender = dynamic_cast<SenderAbstraction<Output> *>(outputNode);

          assert(outputAsNotifier != nullptr && outputAsSender != nullptr);

          for (auto & outputNotifier : outputAsNotifier->notifiers()) {
            this->notifiers().insert(outputNotifier);
          }

        for (auto & outputSender : outputAsSender->senders()) {
            SenderAbstraction<Output>::senders().insert(outputSender);
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
#endif //HEDGEHOG_GRAPH_OUTPUTS_MANAGEMENT_ABSTRACTION_H
