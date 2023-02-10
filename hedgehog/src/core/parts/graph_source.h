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



#ifndef HEDGEHOG_GRAPH_SOURCE_H
#define HEDGEHOG_GRAPH_SOURCE_H

#include <ostream>

#include "../abstractions/base/input_output/sender_abstraction.h"

#include "../abstractions/base/input_output/notifier_abstraction.h"
#include "../abstractions/base/node/node_abstraction.h"
#include "../implementors/concrete_implementor/default_sender.h"

#include "../implementors/concrete_implementor/default_notifier.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

/// @brief Source of the graph, only used in an outer graph
/// @tparam Inputs Input list types of the graph
template<class ...Inputs>
class GraphSource :
    public abstraction::NodeAbstraction,
    public abstraction::NotifierAbstraction,
    public abstraction::SenderAbstraction<Inputs> ... {

 public:

  /// @brief Default constructor
  GraphSource() :
      NodeAbstraction("Source"),
      NotifierAbstraction(std::make_shared<implementor::DefaultNotifier>()),
      abstraction::SenderAbstraction<Inputs>(std::make_shared<implementor::DefaultSender < Inputs>>())... {}

  /// @brief Default destructor
  ~GraphSource() override = default;


  /// @brief Send a piece of data to all input nodes and notify them
  /// @tparam Input Input data type
  /// @param data Input data
  template<class Input>
  void sendAndNotifyAllInputs(std::shared_ptr<Input> &data) {
    static_cast<abstraction::SenderAbstraction<Input> *>(this)->send(data);
    static_cast<NotifierAbstraction *>(this)->notify();
  }

  /// @brief Gather source information
  /// @param printer Printer visitor gathering information on nodes
  void print(Printer *printer) {
    printer->printSource(this);
  }

  /// @brief Node ids [nodeId, nodeGroupId] accessor
  /// @return  Node ids [nodeId, nodeGroupId]
  [[nodiscard]] std::vector<std::pair<std::string const, std::string const>> ids() const override {
    return {{this->id(), this->id()}};
  }

  /// @brief Getter to the node counterpart
  /// @return Nothing, throw an error because there is no Node attached to the core
  /// @throw std::runtime_error because there is no Node attached to the core
  [[nodiscard]] behavior::Node *node() const override {
    throw std::runtime_error("Try to get a node out of a core switch while there is none.");
  }

};
}
}
#endif //HEDGEHOG_GRAPH_SOURCE_H
