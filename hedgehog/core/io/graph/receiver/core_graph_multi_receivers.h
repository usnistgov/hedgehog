//
// Created by anb22 on 5/8/19.
//

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