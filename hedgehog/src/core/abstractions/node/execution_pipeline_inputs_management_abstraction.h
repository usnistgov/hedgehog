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

#ifndef HEDGEHOG_EXECUTION_PIPELINE_INPUTS_MANAGEMENT_ABSTRACTION_H
#define HEDGEHOG_EXECUTION_PIPELINE_INPUTS_MANAGEMENT_ABSTRACTION_H

#include "../../parts/core_switch.h"

#include "../base/input_output/slot_abstraction.h"
#include "../base/input_output/receiver_abstraction.h"

#include "../../implementors/concrete_implementor/slot/default_slot.h"
#include "../../implementors/concrete_implementor/receiver/queue_receiver.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Input management abstraction for the execution pipeline
/// @tparam Inputs Types of input data
template<class ...Inputs>
class ExecutionPipelineInputsManagementAbstraction :
    public SlotAbstraction,
    public ReceiverAbstraction<Inputs> ... {
 private:
  std::unique_ptr<CoreSwitch < Inputs...>> coreSwitch_ = nullptr; ///< Core switch used to call user-defined rules
 protected:
  using inputs_t = std::tuple<Inputs...>; ///<  Accessor to inputs data
 public:

  /// @brief Construct an Input management abstraction for the execution pipeline from an implementation of
  /// MultiSwitchRules
  /// @tparam ExecutionPipelineImplementation Type of data deriving from MultiSwitchRules
  /// @param executionPipeline Implementation of the execution pipeline
  template<class ExecutionPipelineImplementation> requires std::is_base_of_v<behavior::MultiSwitchRules<Inputs...>,
                                                                             ExecutionPipelineImplementation>
  explicit ExecutionPipelineInputsManagementAbstraction(ExecutionPipelineImplementation *const executionPipeline)
      : SlotAbstraction(std::make_shared<implementor::DefaultSlot>()),
        ReceiverAbstraction<Inputs>(std::make_shared<implementor::QueueReceiver<Inputs>>())...,
      coreSwitch_(std::make_unique<CoreSwitch < Inputs...>>
  (static_cast<behavior::MultiSwitchRules<Inputs...> * >(executionPipeline))){}

  /// @brief Default destructor
  ~ExecutionPipelineInputsManagementAbstraction() override = default;

 protected:
  /// @brief Accessor to the switch core
  /// @return Pointer to the switch core
  CoreSwitch<Inputs...> *coreSwitch() const {
    return coreSwitch_.get();
  }

  /// @brief Interface to the user-defined switch rule
  /// @tparam Input Data type to send to the graphs (should be part of the exec pipeline types)
  /// @param data Data of type Input
  /// @param graphId Graph id
  /// @return True if the data needs to be sent to the graph of id graphId
  template<tool::ContainsConcept<Inputs...> Input>
  bool callSendToGraph(std::shared_ptr<Input> &data, size_t const &graphId) {
    return this->coreSwitch_->callSendToGraph(data, graphId);
  }

  /// @brief Test if the receivers are empty
  /// @return True if the receivers queue are empty, else false
  [[nodiscard]] bool receiversEmpty() { return (ReceiverAbstraction<Inputs>::empty() && ...); }

  /// @brief Connect a graph to the switch
  /// @param graph Graph to connect to the switch
  void connectGraphToSwitch(std::shared_ptr<GraphInputsManagementAbstraction<Inputs...>> const graph) {
    auto switchAsNotifier = static_cast<NotifierAbstraction *>(this->coreSwitch_.get());
    (connectInputGraphForAType<Inputs>(graph), ...);
    for (auto slot : std::static_pointer_cast<SlotAbstraction>(graph)->slots()) {
      switchAsNotifier->addSlot(slot);
      slot->addNotifier(switchAsNotifier);
    }
  }

  /// @brief Disconnect the switch from all of the graphs
  void disconnectSwitch() {
    auto switchAsNotifier = static_cast<NotifierAbstraction *>(this->coreSwitch_.get());
    for (auto &slot : switchAsNotifier->connectedSlots()) {
      for (auto &s : slot->slots()) {
        s->removeNotifier(switchAsNotifier);
        s->wakeUp();
      }
    }
  }

  /// @brief The thread wait termination condition
  /// @return The thread should not wait if there is data available or if the node should terminate
  [[nodiscard]] bool waitTerminationCondition() final {
    return !this->receiversEmpty() || canTerminate();
  }

  /// @brief The thread wait termination condition
  /// @return The thread should not wait if there is data available or if the node should terminate
  [[nodiscard]] bool canTerminate() override {
    return !this->hasNotifierConnected() && this->receiversEmpty();
  }

 protected:
  /// @brief Visitor for the execution pipeline edge
  /// @param printer Printer gathering data information
  void printEdgesInformation(Printer *printer) {
    (ReceiverAbstraction<Inputs>::printEdgeInformation(printer), ...);
  }

 private:
  /// @brief Connect a graph to the switch for the type Input
  /// @tparam Input Type of input data
  /// @param graph Graph to connect to the switch
  template<class Input>
  void connectInputGraphForAType(std::shared_ptr<GraphInputsManagementAbstraction<Inputs...>> const graph) {
    auto switchAsSender = static_cast<SenderAbstraction<Input> *>(this->coreSwitch_.get());
    for (auto const receiver : std::static_pointer_cast<ReceiverAbstraction<Input>>(graph)->receivers()) {
      switchAsSender->addReceiver(receiver);
      receiver->addSender(switchAsSender);
    }
  }

};
}
}
}

#endif //HEDGEHOG_EXECUTION_PIPELINE_INPUTS_MANAGEMENT_ABSTRACTION_H
