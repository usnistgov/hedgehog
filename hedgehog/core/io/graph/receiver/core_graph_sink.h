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


#ifndef HEDGEHOG_CORE_GRAPH_SINK_H
#define HEDGEHOG_CORE_GRAPH_SINK_H

#include "../../queue/receiver/core_queue_multi_receivers.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Part of the outer graph that gathers data from the output nodes and makes them available as the graph output
/// @tparam GraphOutput Graph's data output
template<class GraphOutput>
class CoreGraphSink : public CoreQueueMultiReceivers<GraphOutput> {
 public:
  /// @brief Default sink constructor
  CoreGraphSink() : CoreNode("Sink", NodeType::Sink, 1),
                    CoreSlot("Sink", NodeType::Sink, 1),
                    CoreReceiver<GraphOutput>("Sink", NodeType::Sink, 1),
                    CoreQueueMultiReceivers<GraphOutput>("Sink", NodeType::Sink, 1) {
    HLOG_SELF(0, "Creating CoreGraphSink")
  }

  /// @brief Default sink destructor
  ~CoreGraphSink() override {HLOG_SELF(0, "Destructing CoreGraphSink")}

  /// @brief Special visit method for graph's sink
  /// @param printer Printer visitor to print the graph's sink
  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    if (printer->hasNotBeenVisited(this)) {
      printer->printNodeInformation(this);
    }
  }

  /// @brief Get a node from the graph's sink, that does not exist, throw an error in every case
  /// @exception std::runtime_error A graph's sink does not have node
  /// @return nothing, fail with a std::runtime_error
  [[noreturn]] behavior::Node *node() override {
    std::ostringstream oss;
    oss << "Internal error, should not be called, sink does not have a node: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Get a clone of the graph's sink, that does not exist, throw an error in every case
  /// @exception std::runtime_error A graph's sink should not be cloned
  /// @return nothing, fail with a std::runtime_error
  [[noreturn]] std::shared_ptr<CoreNode> clone() override {
    std::ostringstream oss;
    oss
        << "Internal error, should not be called, graph's sink can't be clone, as an outer graph can't be cloned : "
        << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Wait for notification from the output nodes and return the state of the sink
  /// @return True if the sink is terminated, else False
  bool waitForNotification() override {
    std::unique_lock<std::mutex> lock(*(this->slotMutex()));

    HLOG_SELF(2, "Wait for the notification")

    this->notifyConditionVariable()->wait(lock,
                                          [this]() {
                                            return !this->receiversEmpty() || this->numberInputNodes() == 0;
                                          });
    HLOG_SELF(2, "Notification received")

    return true;
  }

  /// @brief Duplicate edge for a sink, throw an error in every case
  /// @exception std::runtime_error A sink does not have edges to connect
  /// @param duplicateNode duplicateNode to connect
  /// @param correspondenceMap Node correspondence map
  [[noreturn]] void duplicateEdge([[maybe_unused]]CoreNode *duplicateNode,
                                  [[maybe_unused]]std::map<CoreNode *,
                                                           std::shared_ptr<CoreNode>> &correspondenceMap) override {
    std::ostringstream oss;
    oss << "Internal error, should not be called, sink does not have edges to connect in an execution pipeline: "
        << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }
};

}
#endif //HEDGEHOG_CORE_GRAPH_SINK_H
