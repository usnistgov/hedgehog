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

#ifndef HEDGEHOG_CX_CYCLE_TEST_H_
#define HEDGEHOG_CX_CYCLE_TEST_H_

#ifdef HH_ENABLE_HH_CX
#include "abstract_test.h"

/// @brief Hedgehog compile-time namespace
namespace hh_cx {

/// @brief Detect cycles in the graph. Hedgehog accepts cycles in graphs, a custom canTerminate method needs to be
/// overloaded in a cycle's node to define custom rule for termination.
/// @details Two algorithms are used to detect cycles:
/// - Tarjan algorithm to detect strongly connected components
/// (https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
/// - Johnson algorithm to detect all elementary circuits from the connected components (https://doi.org/10.1137/0204007)
/// A cycle will be filtered if the canTerminate method has been defined in one of the node in the cycle.
/// @tparam GraphType Graph type
template<tool::HedgehogDynamicGraphForStaticAnalysis GraphType>
class CycleTest : public hh_cx::AbstractTest<GraphType> {
  /// @brief Simple representation of a subgraph used to detect cycles
  /// @attention Use with caution, fewer checks are made to build the graph, are they are already made in the CXGraph
  class SimpleSubGraph {
    /// @brief Type definition for a pointer to const AbstractNode
    using pointer_const_abstract_node = hh_cx::behavior::AbstractNode const *;

    std::vector<pointer_const_abstract_node>
        registeredNodes_{}; ///< Nodes registered in the subgraph

    std::vector<std::vector<bool>>
        adjacencyMatrix_{}; ///< Adjacency's matrix in subgraph

   public:
    /// @brief Default constructor
    constexpr SimpleSubGraph() = default;

    /// @brief Create a subgraph from a graph. Add only nodes from a minimum node ID.
    /// @param graph Graph to extract nodes from.
    /// @param minNodeId Minimum id of nodes extracted
    constexpr SimpleSubGraph(hh_cx::Graph<GraphType> const *graph, size_t minNodeId) {
      for (size_t nodeIdSrc = minNodeId; nodeIdSrc < graph->numberNodesRegistered(); ++nodeIdSrc) {
        for (size_t nodeIdDest = minNodeId; nodeIdDest < graph->numberNodesRegistered(); ++nodeIdDest) {
          if (graph->isLinked(nodeIdSrc, nodeIdDest)) {
            this->addEdge(graph->node(nodeIdSrc), graph->node(nodeIdDest));
          }
        }
      }
    }

    /// @brief Default destructor
    virtual ~SimpleSubGraph() = default;

    /// @brief Registered Nodes accessor
    /// @return Registered Nodes
    [[nodiscard]] constexpr std::vector<pointer_const_abstract_node> const &registeredNodes() const {
      return registeredNodes_;
    }

    /// @brief Accessor to the number of nodes registered in the sub graph
    /// @return Number of nodes registered in the sub graph
    [[nodiscard]] constexpr size_t numberNodes() const { return registeredNodes_.size(); }

    /// @brief Get all adjacent nodes from an origin node
    /// @param origin Origin node to get adjacent nodes from
    /// @return A vector of nodes that receives information from the origin node
    [[nodiscard]] constexpr std::vector<pointer_const_abstract_node>
    adjacentNodes(pointer_const_abstract_node origin) const {
      std::vector<pointer_const_abstract_node> adjacentNodes{};
      auto senderId = nodeId(origin);
      for (size_t receiverId = 0; receiverId < numberNodes(); ++receiverId) {
        if (isLinked(senderId, receiverId)) {
          adjacentNodes.push_back(node(receiverId));
        }
      }
      return adjacentNodes;
    }

    /// @brief Number of registered nodes accessor
    /// @return The number of nodes accessed
    [[nodiscard]] constexpr size_t numberNodesRegistered() const { return registeredNodes_.size(); }

    /// @brief Test if an edge exists between a sender node and a receiver node
    /// @param sender Sender node
    /// @param receiver Receiver node
    /// @return True if an edge exists between a sender node and a receiver node, else False
    [[nodiscard]] constexpr bool isLinked(pointer_const_abstract_node sender,
                                          pointer_const_abstract_node receiver) const {
      auto
          idSender = nodeId(sender),
          idReceiver = nodeId(receiver);

      if (idSender == numberNodes() || idReceiver == numberNodes()) { return false; }
      return adjacencyMatrix_[idSender][idReceiver];
    }

    /// @brief Test if an edge exists between a sender node id and a receiver node id
    /// @param idSender Sender node id
    /// @param idReceiver Receiver node id
    /// @return True if an edge exists between a sender node id and a receiver node id, else False
    /// @throw std::runtime_error Try to access nodes that have not been registered
    [[nodiscard]] constexpr bool isLinked(size_t idSender, size_t idReceiver) const {
      if (idSender >= numberNodes() || idReceiver >= numberNodes()) {
        throw (std::runtime_error("The nodes you are trying to test do not exist in the graph."));
      }
      return adjacencyMatrix_[idSender][idReceiver];
    }

    /// @brief Get a subgraph node id from a node pointer
    /// @param node Node pointer to get an id from
    /// @return The subgraph node id corresponding to the node
    [[nodiscard]]  constexpr size_t nodeId(pointer_const_abstract_node node) const {
      if (!isNodeRegistered(node)) {
        throw (std::runtime_error("The node you are trying to get does not exist in the graph."));
      }
      size_t nodeId = 0;
      for (auto registeredNode : registeredNodes_) {
        if (registeredNode == node) { return nodeId; }
        else { ++nodeId; }
      }
      return nodeId;
    }

    /// @brief Get a node pointer from an id
    /// @param id Get the node corresponding to the id
    /// @return A node pointer corresponding to the id
    [[nodiscard]] constexpr pointer_const_abstract_node node(size_t id) const {
      if (id >= registeredNodes_.size()) {
        throw (std::runtime_error("The node you are requesting does not exist."));
      } else {
        return registeredNodes_.at(id);
      }
    }

    /// @brief Add an edge between a sender node and a receiver node
    /// @param sender Sender node
    /// @param receiver Receiver node
    constexpr void addEdge(pointer_const_abstract_node sender, pointer_const_abstract_node receiver) {
      registerNode(sender);
      registerNode(receiver);
      adjacencyMatrix_.at(nodeId(sender)).at(nodeId(receiver)) = true;
    }
   private:
    /// @brief Test if a node has been registered
    /// @param node Node to test
    /// @return True if the node has been registered, else false
    constexpr bool isNodeRegistered(pointer_const_abstract_node node) const {
      return std::find(registeredNodes_.cbegin(), registeredNodes_.cend(), node) != registeredNodes_.cend();
    }

    /// @brief Register a node in the subgraph if it has not already been registered
    /// @param node Node to register
    constexpr void registerNode(pointer_const_abstract_node node) {
      if (!isNodeRegistered(node)) { registeredNodes_.push_back(node); }
      for (auto &line : adjacencyMatrix_) {
        line.push_back(false);
      }
      adjacencyMatrix_.emplace_back(adjacencyMatrix_.size() + 1, false);
    }
  };

  std::vector<size_t>
      tarjanNodeNumbers_{}, ///< Number associated to the node for the Tarjan algorithm
  tarjanLowLink_{}; ///< Low link property associated to the node for the Tarjan algorithm

  std::vector<bool>
      tarjanIsOnStack_{}, ///< Direct accessor to test node presence in a stack for the Tarjan algorithm
  johnsonBlockedSetArray_{}; ///< List of blocked nodes for the the Johnson algorithm

  std::vector<hh_cx::behavior::AbstractNode const *>
      tarjanStack_{}, ///< Stack used for the the Tarjan algorithm
  johnsonStack_{}; ///< Stack used for the the Johnson algorithm

  std::vector<std::vector<hh_cx::behavior::AbstractNode const *>>
      tarjanConnectedComponent_{}, ///< List of connected components
  johnsonCycles_{}, ///< List of cycles found
  johnsonBlockedMap_{}; ///< Map of nodes to unblock if a node is unblocked for the the Johnson algorithm

  size_t
      numberOfNodes_{}; ///< Number of nodes in the graph
 public:
  /// @brief Default constructor
  constexpr explicit CycleTest() : AbstractTest<GraphType>("Cycle test") {}

  /// @brief Default destructor
  constexpr ~CycleTest() override = default;

  /// @brief Test a graph to detect cycles
  /// @param graph Graph to test
  constexpr void test(hh_cx::Graph<GraphType> const *graph) override {
    numberOfNodes_ = graph->numberNodesRegistered();
    findAllCycles(graph);
    removeCyclesWhereNodesRedefineCanTerminate();
    if (johnsonCycles_.empty()) {
      this->graphValid(true);
    } else {
      this->graphValid(false);
      this->appendErrorMessage("Cycles found, the canTerminate() method needs to be defined for each of these cycles.");
      for (const auto &cycle : johnsonCycles_) {
        this->appendErrorMessage("\n\t");
        auto origin = cycle.at(0);
        for (const auto &node : cycle) {
          this->appendErrorMessage(node->name());
          this->appendErrorMessage(" -> ");
        }
        this->appendErrorMessage(origin->name());
      }
    }
  }

 private:
  /// @brief Find all cycles in the graph
  /// @param graph Graph to get the cycles from
  constexpr void findAllCycles(hh_cx::Graph<GraphType> const *graph) {
    size_t startIndex = 0;
    while (startIndex < graph->numberNodesRegistered()) {
      initTarjan();
      SimpleSubGraph subGraph(graph, startIndex);
      tarjanAllStrongConnect(subGraph);

      auto [leastIndex, minSubGraph] = leastIndexStrongConnectedComponents(subGraph);

      if (leastIndex == numberOfNodes_) {
        break;
      } else {
        initJohnson();
        auto node = subGraph.node(leastIndex);
        findCyclesInSCC(node, node, minSubGraph);
        ++startIndex;
      }
    }
  }

  /// @brief Initialize the data structures for the Tarjan Algorithm
  constexpr void initTarjan() {
    tarjanNodeNumbers_ = std::vector<size_t>(numberOfNodes_);
    tarjanLowLink_ = std::vector<size_t>(numberOfNodes_);
    tarjanIsOnStack_ = std::vector<bool>(numberOfNodes_, false);
    tarjanStack_.clear();
    tarjanConnectedComponent_.clear();
  }

  /// @brief Initialize the data structures for the Johnson Algorithm
  constexpr void initJohnson() {
    johnsonStack_.clear();
    johnsonBlockedSetArray_ = std::vector<bool>(numberOfNodes_, false);
    for (size_t nodeId = 0; nodeId < numberOfNodes_; ++nodeId) {
      johnsonBlockedMap_.emplace_back();
    }
  }

  /// @brief Generate all the strong connected components
  /// @param subGraph Subgraph to get the strong connected components from
  constexpr void tarjanAllStrongConnect(SimpleSubGraph const &subGraph) {
    size_t num = 1;
    for (size_t nodeId = 0; nodeId < subGraph.numberNodesRegistered(); ++nodeId) {
      if (tarjanNodeNumbers_[nodeId] == 0) { tarjanStrongConnect(num, nodeId, subGraph); }
    }
  }

  /// @brief Tarjan algorithm to get the string connected components from a subgraph
  /// @param num Number used in the Tarjan algorithm
  /// @param nodeId Id of the current node
  /// @param subGraph Subgraph to extract the connected components from
  constexpr void tarjanStrongConnect(size_t &num, size_t &nodeId, SimpleSubGraph const &subGraph) {
    auto node = subGraph.node(nodeId);
    tarjanNodeNumbers_[nodeId] = num;
    tarjanLowLink_[nodeId] = num;
    tarjanIsOnStack_[nodeId] = true;
    tarjanStack_.push_back(node);
    num += 1;

    for (auto &neighbor : subGraph.adjacentNodes(node)) {
      size_t neighborId = subGraph.nodeId(neighbor);
      if (tarjanNodeNumbers_[neighborId] == 0) {
        tarjanStrongConnect(num, neighborId, subGraph);
        tarjanLowLink_[nodeId] = std::min(tarjanLowLink_[nodeId], tarjanLowLink_[neighborId]);
      } else if (tarjanIsOnStack_[neighborId]) {
        tarjanLowLink_[nodeId] = std::min(tarjanLowLink_[nodeId], tarjanNodeNumbers_[neighborId]);
      }
    }

    if (tarjanLowLink_[nodeId] == tarjanNodeNumbers_[nodeId]) {
      std::vector<hh_cx::behavior::AbstractNode const *> component{};
      hh_cx::behavior::AbstractNode const *nodeStack = nullptr;
      do {
        nodeStack = tarjanStack_.back();
        tarjanStack_.pop_back();
        tarjanIsOnStack_[subGraph.nodeId(nodeStack)] = false;
        component.push_back(nodeStack);
      } while (node != nodeStack);
      tarjanConnectedComponent_.push_back(component);
    }
  }

  /// @brief Get the least index and the subgraph containing the least index
  /// @param subGraph Subgraph to get the node from
  /// @return The least index and the subgraph containing the least index
  constexpr std::pair<size_t, SimpleSubGraph> leastIndexStrongConnectedComponents(SimpleSubGraph const &subGraph) {
    size_t min = numberOfNodes_;
    std::vector<hh_cx::behavior::AbstractNode const *> minSCC(numberOfNodes_, nullptr);

    for (auto connectedComponent : tarjanConnectedComponent_) {
      if (connectedComponent.size() == 1) {
        auto node = connectedComponent.at(0);
        if (subGraph.isLinked(node, node)) {
          johnsonCycles_.push_back({node});
        }
      } else {
        for (auto node : connectedComponent) {
          if (subGraph.nodeId(node) < min) {
            min = subGraph.nodeId(node);
            minSCC = connectedComponent;
          }
        }
      }
    }

    if (min == numberOfNodes_) { return {numberOfNodes_, SimpleSubGraph{}}; }
    else {
      SimpleSubGraph minSubGraph{};

      std::vector<size_t> minIndexes{};
      for (auto &sender : subGraph.registeredNodes()) {
        for (auto &receiver : subGraph.registeredNodes()) {
          if (subGraph.isLinked(sender, receiver)) {
            if (std::find(minSCC.cbegin(), minSCC.cend(), sender) != minSCC.cend()
                && std::find(minSCC.cbegin(), minSCC.cend(), receiver) != minSCC.cend()) {
              minSubGraph.addEdge(sender, receiver);
              minIndexes.push_back(subGraph.nodeId(sender));
              minIndexes.push_back(subGraph.nodeId(receiver));
            }
          }
        }
      }
      return {*std::min_element(minIndexes.cbegin(), minIndexes.cend()), minSubGraph};
    }
  }

  /// @brief Find all the cycles in a subgraph
  /// @param startNode Cycle start node
  /// @param currentNode Cycle current node
  /// @param subGraph Subgraph used
  /// @return True if cycles are found, else False
  constexpr bool findCyclesInSCC(
      hh_cx::behavior::AbstractNode const *startNode,
      hh_cx::behavior::AbstractNode const *currentNode,
      SimpleSubGraph const &subGraph) {
    bool cycleFound = false;
    johnsonStack_.push_back(currentNode);
    johnsonBlockedSetArray_.at(subGraph.nodeId(currentNode)) = true;

    for (auto neighbor : subGraph.adjacentNodes(currentNode)) {
      if (neighbor == startNode) {
        std::vector<hh_cx::behavior::AbstractNode const *> cycle{};
        auto tempStack(johnsonStack_);
        while (!tempStack.empty()) {
          cycle.push_back(tempStack.back());
          tempStack.pop_back();
        }
        std::reverse(cycle.begin(), cycle.end());

        bool johnsonCycleFound = false;
        for (auto const &johnsonCycle : johnsonCycles_) {
          if (johnsonCycle.size() == cycle.size()) {
            bool cycleSame = true;
            for (size_t nodeId = 0; nodeId < johnsonCycle.size(); ++nodeId) {
              cycleSame &= (johnsonCycle.at(nodeId) == cycle.at(nodeId));
            }
            johnsonCycleFound |= cycleSame;
          }
        }

        if (!johnsonCycleFound) {
          johnsonCycles_.push_back(cycle);
        }
        cycleFound = true;
      } else if (!johnsonBlockedSetArray_.at(subGraph.nodeId(neighbor))) {
        cycleFound |= findCyclesInSCC(startNode, neighbor, subGraph);
      }
    }

    if (cycleFound) { unblock(currentNode, subGraph); }
    else {
      for (auto neighbors : subGraph.adjacentNodes(currentNode)) {
        auto nodes = johnsonBlockedMap_.at(subGraph.nodeId(neighbors));
        if (std::find(nodes.cbegin(), nodes.cend(), currentNode) == nodes.cend()) {
          nodes.push_back(currentNode);
          johnsonBlockedMap_.at(subGraph.nodeId(neighbors)) = nodes;
        }
      }
    }

    johnsonStack_.pop_back();
    return cycleFound;
  }

  /// @brief Remove the cycles in which a node overload the canTerminate method
  constexpr void removeCyclesWhereNodesRedefineCanTerminate() {
    std::vector<std::vector<hh_cx::behavior::AbstractNode const *>> temp{};
    for (const auto &cycle : johnsonCycles_) {
      if (!std::any_of(cycle.cbegin(), cycle.cend(),
                       [](auto const &node) { return node->isCanTerminateOverloaded(); })) {
        temp.push_back(cycle);
      }
    }
    johnsonCycles_ = temp;
  }

  /// @brief Unblock a node, and all cascading nodes found in johnsonBlockedMap
  /// @param node Node to unblock
  /// @param subGraph Subgraph considered
  constexpr void unblock(hh_cx::behavior::AbstractNode const *node, SimpleSubGraph const &subGraph) {
    johnsonBlockedSetArray_.at(subGraph.nodeId(node)) = false;
    for (auto nodeBlocked : johnsonBlockedMap_.at(subGraph.nodeId(node))) {
      if (johnsonBlockedSetArray_.at(subGraph.nodeId(nodeBlocked))) {
        unblock(nodeBlocked, subGraph);
      }
    }
    johnsonBlockedMap_.at(subGraph.nodeId(node)) = {};
  }

};

}

#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_CYCLE_TEST_H_
