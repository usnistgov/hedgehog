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


#ifndef HEDGEHOG_CORE_GRAPH_MULTI_RECEIVERS_H
#define HEDGEHOG_CORE_GRAPH_MULTI_RECEIVERS_H

#include "../../base/receiver/core_slot.h"
#include "../../base/receiver/core_multi_receivers.h"
#include "core_graph_receiver.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Graph multi receiver
/// @tparam GraphInputs Graph's input types
template<class ...GraphInputs>
class CoreGraphMultiReceivers
    : public CoreMultiReceivers<GraphInputs...>, public CoreGraphReceiver<GraphInputs> ... {
 public:
  /// @brief CoreGraphMultiReceivers constructor
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Node number of threads
  CoreGraphMultiReceivers(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreSlot(name, type, numberThreads),
        CoreMultiReceivers<GraphInputs...>(name, type, numberThreads),
        CoreGraphReceiver<GraphInputs>(name, type, numberThreads)... {
    HLOG_SELF(0, "Creating CoreGraphMultiReceivers with type: " << (int) type << " and name: " << name)
  }

  /// @brief CoreGraphMultiReceivers destructor
  ~CoreGraphMultiReceivers() override {HLOG_SELF(0, "Destructing CoreGraphMultiReceivers")}

  /// @brief Test if all the CoreGraphMultiReceivers of the graphs (its input nodes CoreMultiReceivers) are empty
  /// @return True, if all graph's CoreMultiReceivers are empty, else False
  bool receiversEmpty() final { return (static_cast<CoreGraphReceiver<GraphInputs> *>(this)->receiverEmpty() && ...); }

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
#endif //HEDGEHOG_CORE_GRAPH_MULTI_RECEIVERS_H