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

#ifndef HEDGEHOG_TASK_OUTPUTS_MANAGEMENT_ABSTRACTION_H
#define HEDGEHOG_TASK_OUTPUTS_MANAGEMENT_ABSTRACTION_H

#include <ostream>

#include "../../../api/printer/printer.h"
#include "../base/input_output/notifier_abstraction.h"
#include "../base/input_output/receiver_abstraction.h"

#include "../../implementors/concrete_implementor/default_notifier.h"
#include "../../implementors/concrete_implementor/default_sender.h"

#include "../../../tools/concepts.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Output management abstraction for the task
/// @tparam Outputs Types of output data
template<class ...Outputs>
class TaskOutputsManagementAbstraction : public NotifierAbstraction, public SenderAbstraction<Outputs> ... {
 public:
  using outputs_t = std::tuple<Outputs...>; ///< Accessor to the graph outputs type

  /// @brief Default constructor
  TaskOutputsManagementAbstraction()
      : NotifierAbstraction(std::make_shared<implementor::DefaultNotifier>()),
        SenderAbstraction<Outputs>(std::make_shared<implementor::DefaultSender<Outputs>>())... {}

  /// @brief Constructor using concrete implementation of the possible core abstraction
  /// @tparam ConcreteMultiSenders Type of the concrete implementation of the senders abstraction
  /// @param concreteNotifier Concrete implementation of the notifier abstraction
  /// @param concreteMultiSenders Concrete implementation of the senders abstraction
  template<hh::tool::ConcreteMultiSenderImplementation<Outputs...> ConcreteMultiSenders>
  TaskOutputsManagementAbstraction(
      std::shared_ptr<implementor::ImplementorNotifier> concreteNotifier,
      std::shared_ptr<ConcreteMultiSenders> concreteMultiSenders)
      : NotifierAbstraction(concreteNotifier),
        SenderAbstraction<Outputs>(concreteMultiSenders)... {}

  /// @brief Default destructor
  ~TaskOutputsManagementAbstraction() override = default;

  /// @brief Send a piece of data and notify the successors
  /// @tparam OutputDataType Type of data
  /// @param data Data to send
  template<tool::ContainsConcept<Outputs...> OutputDataType>
  void sendAndNotify(std::shared_ptr<OutputDataType> data) {
    static_cast<SenderAbstraction<OutputDataType> *>(this)->send(data);
    static_cast<NotifierAbstraction *>(this)->notify();
  }


 protected:
  /// @brief Copy the inner structure from another task
  /// @param copyableCore Core task to copy inner structure from
  void copyInnerStructure(TaskOutputsManagementAbstraction<Outputs...> *copyableCore) {
    (SenderAbstraction<Outputs>::copyInnerStructure(copyableCore), ...);
  }

  /// @brief Duplicate the output edges for the node clone
  /// @param mapping Map of the nodes and their clones
  void duplicateOutputEdges(std::map<abstraction::NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping){
    (SenderAbstraction<Outputs>::duplicateEdgeSender(mapping), ...);
    this->duplicateEdgeNotifier(mapping);
  }
};
}
}
}

#endif //HEDGEHOG_TASK_OUTPUTS_MANAGEMENT_ABSTRACTION_H
