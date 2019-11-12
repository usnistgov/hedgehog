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


#ifndef HEDGEHOG_ABSTRACT_PRINTER_H
#define HEDGEHOG_ABSTRACT_PRINTER_H
#include <set>
#include "../../tools/logger.h"
/// @brief Hedgehog main namespace
namespace hh {


#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog core namespace
namespace core {
/// @brief Forward declaration of core::CoreNode
class CoreNode;
}
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Printer interface
/// @details Visiting the graph's node following Visitor pattern
class AbstractPrinter {
 private:
  std::set<core::CoreNode const *> uniqueNodes_ = {}; ///< Set of visited nodes by the printer
 public:
  /// @brief Default constructor
  AbstractPrinter() = default;

  /// @brief Default destructor
  virtual ~AbstractPrinter() = default;

  /// @brief Print graph header
  /// @param node Node to print
  virtual void printGraphHeader(core::CoreNode const *node) = 0;

  /// @brief Print graph footer
  /// @param node Node to print
  virtual void printGraphFooter(core::CoreNode const *node) = 0;

  /// @brief Print node information
  /// @param node Node to print
  virtual void printNodeInformation(core::CoreNode *node) = 0;

  /// @brief Print edge information
  /// @param from From node
  /// @param to To node
  /// @param edgeType Type linked to the edge
  /// @param queueSize Queue current size
  /// @param maxQueueSize Queue maximum size
  /// @param isMemoryManaged True if the edge hold a memory managed data, else False
  virtual void printEdge(core::CoreNode const *from,
                         core::CoreNode const *to,
                         std::string_view const &edgeType,
                         size_t const &queueSize,
                         size_t const &maxQueueSize,
                         bool isMemoryManaged) = 0;

  /// @brief Print cluster header
  /// @param clusterNode Node to print
  virtual void printClusterHeader(core::CoreNode const *clusterNode) = 0;

  /// @brief Print cluster footer
  virtual void printClusterFooter() = 0;

  /// @brief Print cluster edge
  /// @param clusterNode Node to print
  virtual void printClusterEdge(core::CoreNode const *clusterNode) = 0;

  /// @brief Print execution pipeline header
  /// @param epNode Execution pipeline node
  /// @param switchNode Switch node
  virtual void printExecutionPipelineHeader(core::CoreNode *epNode, core::CoreNode *switchNode) = 0;

  /// @brief Print execution pipeline footer
  virtual void printExecutionPipelineFooter() = 0;

  /// @brief Print the edges from the switch representation to a node
  /// @param to Edge destination node
  /// @param idSwitch Switch id
  /// @param edgeType Type linked to the edge
  /// @param queueSize Queue current size
  /// @param maxQueueSize Queue maximum size
  /// @param isMemoryManaged True if the edge hold a memory managed data, else False
  virtual void printEdgeSwitchGraphs(core::CoreNode *to,
                                     std::string const &idSwitch,
                                     std::string_view const &edgeType,
                                     size_t const &queueSize,
                                     size_t const &maxQueueSize,
                                     bool isMemoryManaged) = 0;

  /// @brief Accessor to check if a node has been visited by the printer
  /// @param node Node to test
  /// @return True if has already been visited, else False
  bool hasNotBeenVisited(core::CoreNode const *node) {
    return uniqueNodes_.insert(node).second;
  }
};
}
#endif //HEDGEHOG_ABSTRACT_PRINTER_H
