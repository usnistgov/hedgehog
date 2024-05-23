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

#ifndef HEDGEHOG_EXECUTION_PIPELINE_OUTPUTS_MANAGEMENT_ABSTRACTION_H
#define HEDGEHOG_EXECUTION_PIPELINE_OUTPUTS_MANAGEMENT_ABSTRACTION_H

#include "../base/input_output/notifier_abstraction.h"
#include "../base/input_output/sender_abstraction.h"
#include "../../implementors/concrete_implementor/graph/graph_notifier.h"
#include "../../implementors/concrete_implementor/graph/graph_sender.h"
#include "graph_outputs_management_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Output management abstraction for the execution pipeline
/// @tparam Outputs Types of output data
template<class ...Outputs>
class ExecutionPipelineOutputsManagementAbstraction
    : public NotifierAbstraction,
      public SenderAbstraction<Outputs> ...{
 public:

  /// @brief Default constructor
  explicit ExecutionPipelineOutputsManagementAbstraction()
      : NotifierAbstraction(std::make_shared<implementor::GraphNotifier>()),
        SenderAbstraction<Outputs>(std::make_shared<implementor::GraphSender<Outputs>>())...{}

  /// @brief Default destructor
  ~ExecutionPipelineOutputsManagementAbstraction() override = default;

  /// @brief Register output node as output of the execution pipeline
  /// @param coreGraph Core graph to register as output
  void registerGraphOutputNodes(std::shared_ptr<GraphOutputsManagementAbstraction<Outputs...>> coreGraph){
    (this->registerGraphSender<Outputs>(std::static_pointer_cast<SenderAbstraction<Outputs>>(coreGraph)), ...);
    for(auto outputNodeNotifier : coreGraph->notifiers()){
      this->notifiers().insert(outputNodeNotifier);
    }
  }

  /// @brief Duplicate output edges
  /// @param mapping Map from a node and its clone
  void duplicateOutputEdges(std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping){
    (SenderAbstraction<Outputs>::duplicateEdgeSender(mapping), ...);
    this->duplicateEdgeNotifier(mapping);
  }

 private:
  /// @brief Register output node as output of the execution pipeline for a type
  /// @tparam Output Type of the graph output to register
  /// @param coreGraphSender Core graph to register as output
  template <class Output>
  void registerGraphSender(std::shared_ptr<SenderAbstraction<Output>> coreGraphSender){
    for(auto outputNodeSender : coreGraphSender->senders()){
      (static_cast<SenderAbstraction<Output>*>(this))->senders().insert(outputNodeSender);
    }
  }

};
}
}
}

#endif //HEDGEHOG_EXECUTION_PIPELINE_OUTPUTS_MANAGEMENT_ABSTRACTION_H
