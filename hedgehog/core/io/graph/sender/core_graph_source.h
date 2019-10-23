//
// Created by anb22 on 5/9/19.
//

#ifndef HEDGEHOG_CORE_GRAPH_SOURCE_H
#define HEDGEHOG_CORE_GRAPH_SOURCE_H

#include "../../queue/sender/core_queue_sender.h"
#include "../../../../behavior/node.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Part of the outer graph that sends data from the outside to the input nodes
/// @tparam GraphInputs Graph's data inputs
template<class ...GraphInputs>
class CoreGraphSource : public CoreQueueSender<GraphInputs> ... {
 public:
  /// @brief CoreGraphSource default constructor
  CoreGraphSource() :
      CoreNode("Source", NodeType::Source, 1),
      CoreNotifier("Source", NodeType::Source, 1),
      CoreQueueNotifier("Source", NodeType::Source, 1),
      CoreQueueSender<GraphInputs>("Source", NodeType::Source, 1)... {
    HLOG_SELF(0, "Creating CoreGraphSource")
  }

  /// @brief CoreGraphSource default destructor
  ~CoreGraphSource() override {HLOG_SELF(0, "Destructing CoreGraphSource")}

  /// @brief Special visit method for graph's source
  /// @param printer Printer visitor to print the graph's source
  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
      (CoreQueueSender<GraphInputs>::visit(printer), ...);
    }
  }

  /// @brief Add slot to all inputs node possessing a CoreQueueSender
  /// @param slot CoreSlot to add to all inputs node possessing a CoreQueueSender
  void addSlot(CoreSlot *slot) final {
    HLOG_SELF(0, "Add slot: " << slot->name() << "(" << slot->id() << ")")
    (CoreQueueSender<GraphInputs>::addSlot(slot), ...);
  }

  /// @brief Get the graph's source slot
  /// @attention should not be called
  /// @return Empty set of CoreSlot, the Graph's source have no slots connected
  std::set<CoreSlot *> getSlots() override { return {}; }

  /// @brief Notify all input nodes possessing a CoreQueueSender of termination
  void notifyAllTerminated() final {
    HLOG_SELF(2, "Notify all terminated")
    (CoreQueueSender<GraphInputs>::notifyAllTerminated(), ...);
  }

  /// @brief Get a clone of the graph's source, that does not exist, throw an error in every case
  /// @exception std::runtime_error A graph's source should not be cloned
  /// @return nothing, fail with a std::runtime_error
  [[noreturn]] std::shared_ptr<CoreNode> clone() override {
    std::ostringstream oss;
    oss
        << "Internal error, should not be called, graph's source can't be clone, as an outer graph can't be cloned : "
        << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Get a node from the graph's source, that does not exist, throw an error in every case
  /// @exception std::runtime_error A graph's source does not have node
  /// @return nothing, fail with a std::runtime_error
  [[noreturn]] behavior::Node *node() override {
    std::ostringstream oss;
    oss << "Internal error, should not be called, source does not have a node: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Duplicate edge for a source, throw an error in every case
  /// @exception std::runtime_error A source does not have edges to connect
  /// @param duplicateNode duplicateNode to connect
  /// @param correspondenceMap Node correspondence map
  [[noreturn]] void duplicateEdge(
      [[maybe_unused]] CoreNode *duplicateNode,
      [[maybe_unused]] std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
    std::ostringstream oss;
    oss << "Internal error, should not be called, source does not have edges to connect in an execution pipeline: "
        << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }
};

}
#endif //HEDGEHOG_CORE_GRAPH_SOURCE_H
