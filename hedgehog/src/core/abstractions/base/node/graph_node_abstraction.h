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

#ifndef HEDGEHOG_GRAPH_NODE_ABSTRACTION_H
#define HEDGEHOG_GRAPH_NODE_ABSTRACTION_H

#include "node_abstraction.h"
#include "task_node_abstraction.h"
#include "execution_pipeline_node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Base graph node abstraction
class GraphNodeAbstraction : public NodeAbstraction, public PrintableAbstraction {
 public:
  /// @brief Graph status (INITialisation, EXECution, TERMination, INSide)
  enum Status { INIT, EXEC, TERM, INS };
 private:
  std::chrono::time_point<std::chrono::system_clock> const
      graphStartCreation_ = std::chrono::system_clock::now(); ///< graph creation timestamp
  std::chrono::nanoseconds
      graphConstructionDuration_ = std::chrono::nanoseconds::zero(); ///< Node creation duration

  int deviceId_ = 0; ///< Device Id used for computation on devices
  size_t graphId_ = 0; ///< Graph Id used to identify a graph in an execution pipeline

 protected:
  std::unique_ptr<std::map<NodeAbstraction *, std::vector<NodeAbstraction *>>> const
      insideNodesAndGroups_ = nullptr; ///< All nodes of the graph, mapped to their group nodes

  Status graphStatus_ = Status::INIT; ///< Group status

 public:
  /// @brief Base graph abstraction
  /// @param name Graph's name
  explicit GraphNodeAbstraction(std::string const &name) :
      NodeAbstraction(name),
      insideNodesAndGroups_(std::make_unique<std::map<NodeAbstraction *, std::vector<NodeAbstraction *>>>()) {}

  /// @brief Default destructor
  ~GraphNodeAbstraction() override = default;

  /// @brief Device id accessor
  /// @return The device id
  [[nodiscard]] int deviceId() const override { return deviceId_; }

  /// @brief Graph id accessor
  /// @return The device id
  [[nodiscard]] size_t graphId() const override { return graphId_; }

  /// @brief Graph status accessor
  /// @return Graph status
  [[nodiscard]] Status graphStatus() const { return graphStatus_; }

  /// @brief Graph start creation timestamp accessor
  /// @return Graph start creation timestamp
  [[nodiscard]] std::chrono::time_point<std::chrono::system_clock> const &graphStartCreation() const {
    return graphStartCreation_;
  }

  /// @brief Graph construction duration accessor
  /// @return Graph construction duration
  [[nodiscard]] std::chrono::nanoseconds const &graphConstructionDuration() const { return graphConstructionDuration_; }

  /// @brief Setter to the device id
  /// @param deviceId Device id to set
  void deviceId(int deviceId) { deviceId_ = deviceId; }

  /// @brief Setter to the graph id
  /// @param graphId Graph id to set
  void graphId(size_t graphId) { graphId_ = graphId; }

  /// @brief Setter to the graph construction duration
  /// @param graphConstructionDuration Graph construction duration
  void graphConstructionDuration(std::chrono::nanoseconds const &graphConstructionDuration) {
    graphConstructionDuration_ = graphConstructionDuration;
  }

  /// @brief Register a graph inside a graph
  /// @param belongingGraph Belonging graph used to register this graph
  void registerNode(GraphNodeAbstraction *belongingGraph) override {
    NodeAbstraction::registerNode(belongingGraph);
    setInside();
  }

  /// @brief Accessor to the min / max execution duration of the nodes in the graph
  /// @return Min / max execution duration in nanoseconds of the nodes in the graph
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxExecutionDuration() const {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxExecDuration = {
        std::chrono::nanoseconds::max(), std::chrono::nanoseconds::min()
    };

    NodeAbstraction *node;

    for (auto insideNode : *(this->insideNodesAndGroups_)) {
      node = insideNode.first;

      if (dynamic_cast<hh::core::abstraction::PrintableAbstraction *>(node)) {
        auto nodeAsGraph = dynamic_cast<GraphNodeAbstraction *>(node);
        auto nodeAsEP = dynamic_cast<ExecutionPipelineNodeAbstraction *>(node);

        if (nodeAsGraph) {
          auto graphMinMax = nodeAsGraph->minMaxExecutionDuration();
          minMaxExecDuration.first = std::min(minMaxExecDuration.first, graphMinMax.first);
          minMaxExecDuration.second = std::max(minMaxExecDuration.second, graphMinMax.second);
        }
        if (nodeAsEP) {
          auto epMinMax = nodeAsEP->minMaxExecutionDuration();
          minMaxExecDuration.first = std::min(minMaxExecDuration.first, epMinMax.first);
          minMaxExecDuration.second = std::max(minMaxExecDuration.second, epMinMax.second);
        } else {
          auto execDuration = node->dequeueExecDuration();
          minMaxExecDuration.first = std::min(minMaxExecDuration.first, execDuration);
          minMaxExecDuration.second = std::max(minMaxExecDuration.second, execDuration);
          std::vector<NodeAbstraction *> &group = insideNode.second;
          if (!group.empty()) {
            auto [minElement, maxElement] = std::minmax_element(
                group.begin(), group.end(),
                [](auto lhs, auto rhs) { return lhs->dequeueExecDuration() < rhs->dequeueExecDuration(); }
            );
            minMaxExecDuration.first = std::min(minMaxExecDuration.first, (*minElement)->dequeueExecDuration());
            minMaxExecDuration.second = std::max(minMaxExecDuration.second, (*maxElement)->dequeueExecDuration());
          }
        }
      }
    }

    return minMaxExecDuration;
  }

  /// @brief Accessor to the min / max wait duration of the nodes in the graph
  /// @return Min / max wait duration in nanoseconds of the nodes in the graph
  /// @throw std::runtime_error All the nodes in a group are not of the same types
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxWaitDuration() const {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxWaitDuration = {
        std::chrono::nanoseconds::max(), std::chrono::nanoseconds::min()
    };

    NodeAbstraction *node;

    for (auto insideNode : *(this->insideNodesAndGroups_)) {
      node = insideNode.first;

      if (dynamic_cast<hh::core::abstraction::PrintableAbstraction *>(node)) {
        auto nodeAsGraph = dynamic_cast<GraphNodeAbstraction *>(node);
        auto nodeAsEP = dynamic_cast<ExecutionPipelineNodeAbstraction *>(node);

        if (nodeAsGraph) {
          auto graphMinMax = nodeAsGraph->minMaxWaitDuration();
          minMaxWaitDuration.first = std::min(minMaxWaitDuration.first, graphMinMax.first);
          minMaxWaitDuration.second = std::max(minMaxWaitDuration.second, graphMinMax.second);
        }
        if (nodeAsEP) {
          auto epMinMax = nodeAsEP->minMaxWaitDuration();
          minMaxWaitDuration.first = std::min(minMaxWaitDuration.first, epMinMax.first);
          minMaxWaitDuration.second = std::max(minMaxWaitDuration.second, epMinMax.second);
        } else if (auto task = dynamic_cast<TaskNodeAbstraction *>(node)) {
          auto waitDuration = task->waitDuration();
          minMaxWaitDuration.first = std::min(minMaxWaitDuration.first, waitDuration);
          minMaxWaitDuration.second = std::max(minMaxWaitDuration.second, waitDuration);
          std::vector<NodeAbstraction *> &group = insideNode.second;
          if (!group.empty()) {
            auto [minElement, maxElement] = std::minmax_element(
                group.begin(), group.end(),
                [](auto lhs, auto rhs) {
                  auto lhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(lhs);
                  auto rhsAsTask = dynamic_cast<TaskNodeAbstraction const *>(rhs);
                  if (lhsAsTask && rhsAsTask) {
                    return lhsAsTask->waitDuration() < rhsAsTask->waitDuration();
                  } else {
                    throw std::runtime_error(
                        "All the nodes in a group should be of the same type, the representative is of"
                        " type TaskNodeAbstraction but not the nodes in the group.");
                  }
                }
            );
            auto minTask = dynamic_cast<TaskNodeAbstraction *>(*minElement);
            auto maxTask = dynamic_cast<TaskNodeAbstraction *>(*maxElement);
            if (minTask && maxTask) {
              minMaxWaitDuration.first = std::min(minMaxWaitDuration.first, minTask->waitDuration());
              minMaxWaitDuration.second = std::max(minMaxWaitDuration.second, maxTask->waitDuration());
            } else {
              throw std::runtime_error("All the nodes in a group should be of the same type, the representative is of"
                                       " type TaskNodeAbstraction but not the nodes in the group.");
            }
          }
        } else {
          return {std::chrono::nanoseconds::zero(), std::chrono::nanoseconds::zero()};
        }
      }
    }

    return minMaxWaitDuration;
  }

/// @brief Interface to join the threads inside of a graph, called by the Default scheduler
  virtual void joinThreads() = 0;

/// @brief Set a graph inside another one
  virtual void setInside() = 0;

/// @brief Create the groups and launch the threads of the inside nodes
/// @param waitForInitialization Wait for internal nodes to be initialized flags
  virtual void createInnerGroupsAndLaunchThreads(bool waitForInitialization) = 0;

/// @brief Register a core inside a graph
/// @param core Core to add to the graph
  virtual void registerNodeInsideGraph(NodeAbstraction *core) = 0;
};
}
}
}

#endif //HEDGEHOG_GRAPH_NODE_ABSTRACTION_H
