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



#ifndef HEDGEHOG_PRINTER_H
#define HEDGEHOG_PRINTER_H

#include <set>
#include <memory>
#include <ostream>

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/// @brief Forward declaration NodeAbstraction
class NodeAbstraction;
/// @brief Forward declaration GraphNodeAbstraction
class GraphNodeAbstraction;
/// @brief Forward declaration AnyGroupableAbstraction
class AnyGroupableAbstraction;
/// @brief Forward declaration ExecutionPipelineNodeAbstraction
class ExecutionPipelineNodeAbstraction;
#endif //DOXYGEN_SHOULD_SKIP_THIS

}
}

/// @brief Printer abstraction to get a snapshot of the metrics of the Hedgehog graph
class Printer {
 private:
  std::unique_ptr<std::set<core::abstraction::NodeAbstraction const * >>
      uniqueNodes_ = nullptr; ///< Uniques Nodes registered (already printed)

 public:
  /// @brief Default constructor
  Printer() : uniqueNodes_(std::make_unique<std::set<core::abstraction::NodeAbstraction const * >>()) {}

  /// @brief Default destructor
  virtual ~Printer() = default;

  /// @brief Print graph header
  /// @param graph Graph to print
  virtual void printGraphHeader(core::abstraction::GraphNodeAbstraction const *graph) = 0;

  /// @brief Print graph footer
  /// @param graph Graph to print
  virtual void printGraphFooter(core::abstraction::GraphNodeAbstraction const *graph) = 0;

  /// @brief Print execution pipeline header
  /// @param ep Execution pipeline to print
  /// @param switchNode Execution pipeline switch
  virtual void printExecutionPipelineHeader(
      core::abstraction::ExecutionPipelineNodeAbstraction const *ep,
      core::abstraction::NodeAbstraction const *switchNode) = 0;

  /// @brief Print execution pipeline footer
  virtual void printExecutionPipelineFooter() = 0;

  /// @brief Print node information
  /// @param node Node to print
  virtual void printNodeInformation(core::abstraction::NodeAbstraction const *node) = 0;

  /// @brief Print edge information
  /// @param from From node
  /// @param to To node
  /// @param edgeType Type linked to the edge
  /// @param queueSize Queue current numverElementsReceived
  /// @param maxQueueSize Queue maximum numverElementsReceived
  virtual void printEdge(core::abstraction::NodeAbstraction const *from, core::abstraction::NodeAbstraction const *to,
                         std::string const &edgeType,
                         size_t const &queueSize, size_t const &maxQueueSize) = 0;

  /// @brief Print group of nodes
  /// @param representative Group's representative
  /// @param group Group of nodes
  virtual void printGroup(core::abstraction::NodeAbstraction *representative,
                          std::vector<core::abstraction::NodeAbstraction *> const &group) = 0;

  /// @brief Print outer graph source
  /// @param source Graph source
  virtual void printSource(core::abstraction::NodeAbstraction const *source) = 0;

  /// @brief Print outer graph sink
  /// @param sink Graph sink
  virtual void printSink(core::abstraction::NodeAbstraction const *sink) = 0;

  /// @brief Register a visited node
  /// @param nodeAbstraction Node to register
  /// @return True if registered, false if already registered
  bool registerNode(core::abstraction::NodeAbstraction const *nodeAbstraction) {
    return uniqueNodes_->insert(nodeAbstraction).second;
  }
};

}
#endif //HEDGEHOG_PRINTER_H
