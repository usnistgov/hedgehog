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

#ifndef HEDGEHOG_GRAPH_SINK_H
#define HEDGEHOG_GRAPH_SINK_H

#include <variant>
#include "../abstractions/base/input_output/slot_abstraction.h"
#include "../abstractions/base/node/node_abstraction.h"
#include "../abstractions/base/input_output/receiver_abstraction.h"

#include "../implementors/concrete_implementor/slot/default_slot.h"
#include "../implementors/concrete_implementor/receiver/queue_receiver.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

/// @brief Sink of the graph, only used in an outer graph
/// @tparam Outputs Output list types of the graph
template<class ...Outputs>
class GraphSink :
    public abstraction::NodeAbstraction,
    public abstraction::SlotAbstraction,
    public abstraction::ReceiverAbstraction<Outputs> ... {
 public:
  using ResultType_t = std::shared_ptr<std::variant<std::shared_ptr<Outputs>...>>; ///< Accessor to the output types

  /// @brief Default constructor
  GraphSink() :
      NodeAbstraction("Sink"),
      SlotAbstraction(std::static_pointer_cast<implementor::ImplementorSlot>(std::make_shared<implementor::DefaultSlot>())),
      abstraction::ReceiverAbstraction<Outputs>(
          std::make_shared<implementor::QueueReceiver<Outputs>>())... {}

  /// @brief Default destructor
  ~GraphSink() override = default;

  /// @brief Get a result from the sink, if none is available wait for it (block the current thread)
  /// @return A variant of shared pointers containing an output data
  ResultType_t getBlockingResult() {
    ResultType_t res = nullptr;
    if (!sleep()) {
      res = std::make_shared<std::variant<std::shared_ptr<Outputs>...>>();
      bool outputFound = false;
      (getOneAvailableResultForAType<Outputs>(outputFound, res), ...);
    }
    return res;
  }

  /// @brief Gather sink information
  /// @param printer Printer visitor gathering information on nodes
  void print(Printer *printer) {
    printer->printSink(this);
    (abstraction::ReceiverAbstraction<Outputs>::printEdgeInformation(printer), ...);
  }

  /// @brief Node ids [nodeId, nodeGroupId] accessor
  /// @return  Node ids [nodeId, nodeGroupId]
  [[nodiscard]] std::vector<std::pair<std::string const, std::string const>> ids() const override {
    return {{this->id(), this->id()}};
  }

  /// @brief Test if the sink can leave its wait state or not
  /// @return Yes if it can leave its wait state, else false
  [[nodiscard]] bool waitTerminationCondition() override {
    return !this->receiversEmpty() || !this->hasNotifierConnected();
  }

  /// @brief Test if the sink can terminate or not
  /// @return Yes if it can terminate, else false
  [[nodiscard]] bool canTerminate() override { return !this->hasNotifierConnected() && this->receiversEmpty(); }

 private:
  /// @brief Test if there is an available output data for a data
  /// @details If none as been previously found (outputFound == true) and if the receiver for this type is not empty,
  /// get a data from the receiver
  /// @tparam Output Type of the output data
  /// @param outputFound Flag to avoid getting multiple output data in one run
  /// @param res Output data
  template<class Output>
  void getOneAvailableResultForAType(bool &outputFound, ResultType_t &res) {
    if (!outputFound) {
      std::shared_ptr<Output> data = nullptr;
      outputFound = abstraction::ReceiverAbstraction<Output>::getInputData(data);
      if (outputFound) { *res = data; }
    }
  }

  /// @brief Test if the receivers for all types are empty
  /// @return True if the receivers for all types are empty, else false
  [[nodiscard]] inline bool receiversEmpty() { return (abstraction::ReceiverAbstraction<Outputs>::empty() && ...); }

  /// @brief Getter to the node counterpart
  /// @return Nothing, throw an error because there is no Node attached to the core
  /// @throw std::runtime_error because there is no Node attached to the core
  [[nodiscard]] behavior::Node *node() const override {
    throw std::runtime_error("Try to get a node out of a core switch while there is none.");
  }
};
}
}

#endif //HEDGEHOG_GRAPH_SINK_H
