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



#ifndef HEDGEHOG_NODE_ABSTRACTION_H
#define HEDGEHOG_NODE_ABSTRACTION_H

#include <cstddef>
#include <cassert>
#include <string>
#include <chrono>
#include <ostream>
#include <map>
#include <utility>
#include <vector>
#include <sstream>
//#include "graph_node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog behavior namespace
namespace behavior {
/// @brief Forward declaration of Node
class Node;
}
#endif //DOXYGEN_SHOULD_SKIP_THIS


/// @brief Hedgehog core namespace
namespace core {

/// @brief Hedgehog abstraction namespace
namespace abstraction {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/// @brief Forward declaration of Node
class GraphNodeAbstraction;

#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Base core node abstraction
class NodeAbstraction {
 private:
  std::string const name_; ///< Name of the core
  bool isRegistered_ = false; ///< Is registered into a graph

  GraphNodeAbstraction *belongingGraph_ = nullptr; ///< Graph holding this node
  std::chrono::nanoseconds dequeueExecDuration_ = std::chrono::nanoseconds::zero(); ///< Node execution duration


  std::chrono::time_point<std::chrono::system_clock>
      startExecutionTimeStamp_ = std::chrono::system_clock::now(); ///< Node begin execution timestamp

 public:
  /// @brief Core node constructor using the core's name
  /// @param name Core's name
  explicit NodeAbstraction(std::string name) : name_(std::move(name)) {}

  /// @brief DEfault destructor
  virtual ~NodeAbstraction() = default;

  /// @brief Accessor to the core's name
  /// @return Name of the core
  [[nodiscard]] std::string const &name() const { return name_; }

  /// @brief Core's id ('x' + address of abstraction) as string
  /// @return Id as string
  [[nodiscard]] virtual std::string id() const {
    std::ostringstream oss;
    oss << "x" << this;
    return oss.str();
  }

  /// @brief Accessor to registration flag
  /// @return True if the node is registered in a graph, else false
  [[nodiscard]] bool isRegistered() const { return isRegistered_; }


  /// @brief Belonging graph accessor
  /// @return Belonging graph
  [[nodiscard]] GraphNodeAbstraction *belongingGraph() const { return belongingGraph_; }

  /// @brief Get the device identifier (got from belonging graph)
  /// @return Device id
  [[nodiscard]] virtual int deviceId() const { return ((NodeAbstraction *) this->belongingGraph())->deviceId(); };

  /// @brief Get the graph identifier (got from belonging graph)
  /// @return Graph id
  [[nodiscard]] virtual size_t graphId() const { return ((NodeAbstraction *) this->belongingGraph())->graphId(); };

  /// @brief Execution duration
  /// @return Duration in nanosecond
  [[nodiscard]] std::chrono::nanoseconds const &dequeueExecDuration() const { return dequeueExecDuration_; }

  /// @brief Accessor to the starting execution timestamp
  /// @return Execution time starting timestamp
  [[nodiscard]] std::chrono::time_point<std::chrono::system_clock> const &startExecutionTimeStamp() const {
    return startExecutionTimeStamp_;
  }

  /// @brief Setter to the starting execution timestamp
  /// @param startExecutionTimeStamp Starting execution timestamp
  void startExecutionTimeStamp(std::chrono::time_point<std::chrono::system_clock> const &startExecutionTimeStamp) {
    startExecutionTimeStamp_ = startExecutionTimeStamp;
  }

  /// @brief Increment execution duration
  /// @param exec Duration to add in nanoseconds
  void incrementDequeueExecutionDuration(std::chrono::nanoseconds const &exec) { this->dequeueExecDuration_ += exec; }

  /// @brief Register node to the given graph
  /// @param belongingGraph Belonging graph
  virtual void registerNode(GraphNodeAbstraction *belongingGraph) {
    assert(isRegistered_ == false);
    belongingGraph_ = belongingGraph;
    isRegistered_ = true;
  }

  /// @brief Node ids [nodeId, nodeGroupId] accessor
  /// @return  Node ids [nodeId, nodeGroupId]
  [[nodiscard]] virtual std::vector<std::pair<std::string const, std::string const>> ids() const = 0;

  /// @brief Node accessor
  /// @return Node attached to this core
  [[nodiscard]] virtual behavior::Node * node() const = 0;
};
}
}
}
#endif //HEDGEHOG_NODE_ABSTRACTION_H
