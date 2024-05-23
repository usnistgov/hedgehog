// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve`, modify and create
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

#ifndef HEDGEHOG_CX_GRAPH_H_
#define HEDGEHOG_CX_GRAPH_H_

#ifdef HH_ENABLE_HH_CX

#include "node.h"

#include "../tools/concepts.h"
#include "../tools/types_nodes_map.h"

#include "../../src/tools/meta_functions.h"

/// @brief Hedgehog compile-time namespace
namespace hh_cx {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of AbstractTest
/// @tparam GraphType Type of dynamic graph
template<tool::HedgehogDynamicGraphForStaticAnalysis GraphType>
class AbstractTest;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Graph representation at compile time
/// @tparam GraphType Type of dynamic graph
template<tool::HedgehogDynamicGraphForStaticAnalysis GraphType>
class Graph : public Node<GraphType> {
 private:
  std::vector<hh_cx::behavior::AbstractNode const *>
      registeredNodes_{}; ///< Registered nodes

  std::vector<std::string>
      registeredNodesName_{}; ///< Registered nodes name

  tool::TypesNodesMap
      inputNodes_{}, ///< Map between graph's input types and list of input nodes
      outputNodes_{}; ///< Map between graph's output types and list of output nodes

  std::vector<std::vector<std::vector<std::string>>>
      adjacencyMatrix_{}, ///< Adjacency matrix (Sender -> Receiver -> List of types)
      ROEdges_{}, ///< Read only edges (Sender -> Receiver -> List of types)
      constEdges_{}; ///< Const edges (Sender -> Receiver -> List of types)

  std::vector<AbstractTest<GraphType> const *> tests_{}; ///< Graph's tests

  std::string report_{}; ///< Graph's tests' report

 public:
  /// @brief CXGraph Constructor
  /// @param name CXGraph name, used to build the hh::Graph
  constexpr explicit Graph(std::string const &name) : hh_cx::Node<GraphType>(name) {}

  /// @brief Default destructor
  constexpr virtual ~Graph() = default;

  /// @brief Test if two nodes are connected in the graph for any type
  /// @param sender Sender node
  /// @param receiver Receiver node
  /// @return True if the nodes are connected, else false
  [[nodiscard]] constexpr bool isLinked(
      hh_cx::behavior::AbstractNode const *sender,
      hh_cx::behavior::AbstractNode const *receiver) const {
    return isLinked(nodeId(sender), nodeId(receiver));
  }

  /// @brief Test if two nodes (represented by their ids) are connected in the graph for any type
  /// @param senderId Sender node id
  /// @param receiverId Receiver node id
  /// @return True if the nodes are connected, else false
  [[nodiscard]] constexpr bool isLinked(
      size_t const senderId,
      size_t const receiverId) const {
    if (senderId >= numberNodesRegistered() || receiverId >= numberNodesRegistered()) {
      throw (std::runtime_error("The node you are trying to get does not exist in the graph."));
    }
    return !adjacencyMatrix_.at(senderId).at(receiverId).empty();
  }

  /// @brief Test if two nodes are connected in the graph for a declared RO type
  /// @param sender Sender node
  /// @param receiver Receiver node
  /// @param typeName Type to test
  /// @return True if the nodes are connected, else false
  [[nodiscard]] constexpr bool isROLinked(
      hh_cx::behavior::AbstractNode const *sender,
      hh_cx::behavior::AbstractNode const *receiver,
      std::string const &typeName) const {
    return isROLinked(nodeId(sender), nodeId(receiver), typeName);
  }

  /// @brief Test if two nodes (represented by their ids) are connected in the graph for a declared RO type
  /// @param senderId Sender node id
  /// @param receiverId Receiver node id
  /// @param typeName Type to test
  /// @return True if the nodes are connected, else false
  [[nodiscard]] constexpr bool isROLinked(
      size_t const senderId,
      size_t const receiverId,
      std::string const &typeName) const {
    if (senderId >= numberNodesRegistered() || receiverId >= numberNodesRegistered()) {
      throw (std::runtime_error("The node you are trying to get does not exist in the graph."));
    }
    auto listROType = ROEdges_.at(senderId).at(receiverId);
    return std::find(listROType.cbegin(), listROType.cend(), typeName) != listROType.cend();
  }

  /// @brief Test if two nodes are connected in the graph for a const type
  /// @param sender Sender node
  /// @param receiver Receiver node
  /// @param typeName Const type to test
  /// @return True if the nodes are connected, else false
  [[nodiscard]] constexpr bool isConstLinked(
      hh_cx::behavior::AbstractNode const *sender,
      hh_cx::behavior::AbstractNode const *receiver,
      std::string const &typeName) const {
    return isConstLinked(nodeId(sender), nodeId(receiver), typeName);
  }

  /// @brief Test if two nodes (represented by their ids) are connected in the graph for a const type
  /// @param senderId Sender node id
  /// @param receiverId Receiver node id
  /// @param typeName Const type to test
  /// @return True if the nodes are connected, else false
  [[nodiscard]] constexpr bool isConstLinked(
      size_t const senderId,
      size_t const receiverId,
      std::string const &typeName) const {
    if (senderId >= numberNodesRegistered() || receiverId >= numberNodesRegistered()) {
      throw (std::runtime_error("The node you are trying to get does not exist in the graph."));
    }
    auto listConstType = constEdges_.at(senderId).at(receiverId);
    return std::find(listConstType.cbegin(), listConstType.cend(), typeName) != listConstType.cend();
  }

  /// @brief Accessor to the report
  /// @return Graph's report
  [[nodiscard]] constexpr std::string const &report() const { return report_; }

  /// @brief Accessor to the registered nodes
  /// @return The registered nodes
  [[nodiscard]] constexpr std::vector<hh_cx::behavior::AbstractNode const *> const &registeredNodes() const {
    return registeredNodes_;
  }

  /// @brief Accessor to the registered nodes name
  /// @return The registered nodes name
  [[nodiscard]] constexpr std::vector<std::string> const &registeredNodesName() const {
    return registeredNodesName_;
  }
  /// @brief Access the maximum number of edges and the longest type name size
  /// @return The maximum number of edges and the longest type name
  [[nodiscard]] constexpr std::pair<size_t, size_t> maxEdgeSizes() const {
    size_t maxNumberEdges = 0, maxEdgeTypeSize = 0;
    for (auto const &sender : adjacencyMatrix_) {
      for (auto const &receiver : sender) {
        if (maxNumberEdges < receiver.size()) { maxNumberEdges = receiver.size(); }
        for (auto const &outputType : receiver) {
          if (maxEdgeTypeSize < outputType.size() + 1) { maxEdgeTypeSize = outputType.size() + 1; }
        }
      }
    }
    return {maxNumberEdges, maxEdgeTypeSize};
  }

  /// @brief Access the maximum node name size
  /// @return Return the maximum node name size
  [[nodiscard]] constexpr size_t maxNodeNameSize() const {
    size_t maxNodeNameSize = 0;
    for (auto const &nodeName : registeredNodesName_) {
      if (nodeName.size() > maxNodeNameSize) { maxNodeNameSize = nodeName.size(); }
    }
    return maxNodeNameSize;
  }

  /// @brief Accessor to the adjacency matrix
  /// @return The adjacency matrix
  [[nodiscard]] constexpr std::vector<std::vector<std::vector<std::string>>> const &adjacencyMatrix() const {
    return adjacencyMatrix_;
  }

  /// @brief Accessor to the input nodes
  /// @return Return a map between the node input types, and the list of nodes
  [[nodiscard]] constexpr tool::TypesNodesMap const &inputNodes() const { return inputNodes_; }

  /// @brief Accessor to the output nodes
  /// @return Return a map between the node output types, and the list of nodes
  [[nodiscard]] constexpr tool::TypesNodesMap const &outputNodes() const { return outputNodes_; }

  /// @brief Get the list of connected nodes and the list of types for the connections
  /// @param node Sender node
  /// @return List of connected nodes and the list of types for the connections (std::vector<std::vector<std::string>>
  /// ReceiverId -> List of types)
  [[nodiscard]] constexpr std::vector<std::vector<std::string>> const &
  adjacentNodesTypes(hh_cx::behavior::AbstractNode const *node) const {
    return adjacencyMatrix_.at(nodeId(node));
  }

  /// @brief Get all adjacent nodes from an origin node for any types
  /// @param node Origin node to get adjacent nodes from
  /// @return A vector of nodes that receives information the origin node
  [[nodiscard]] constexpr std::vector<hh_cx::behavior::AbstractNode const *>
  adjacentNodes(hh_cx::behavior::AbstractNode const *node) const {

    std::vector<hh_cx::behavior::AbstractNode const *> adjacentNodes{};

    for (auto const &possibleReceiver : this->registeredNodes_) {
      if (isLinked(node, possibleReceiver)) {
        adjacentNodes.push_back(possibleReceiver);
      }
    }
    return adjacentNodes;
  }

  /// @brief Get all adjacent nodes from an origin node for a type
  /// @param node Origin node to get adjacent nodes from
  /// @param type Connection type
  /// @return A vector of nodes that receives type data from the origin node
  [[nodiscard]] constexpr std::vector<hh_cx::behavior::AbstractNode const *>
  adjacentNodes(hh_cx::behavior::AbstractNode const *node, std::string const &type) const {

    std::vector<hh_cx::behavior::AbstractNode const *> adjacentNodes{};

    for (auto const &possibleReceiver : this->registeredNodes_) {
      auto const &connectionTypes = this->adjacencyMatrix_.at(nodeId(node)).at(nodeId(possibleReceiver));
      if (std::find(connectionTypes.cbegin(), connectionTypes.cend(), type) != connectionTypes.cend()) {
        adjacentNodes.push_back(possibleReceiver);

      }
    }

    return adjacentNodes;
  }

  /// @brief Get the number of registered nodes
  /// @return Number of registered nodes
  [[nodiscard]] constexpr size_t numberNodesRegistered() const { return registeredNodes_.size(); }

  /// @brief Set a static node as input of the static graph for all common types. Register the node if need be.
  /// @tparam InputNode Type of the input node
  /// @param inputNode Instance of the input node
  template<tool::HedgehogStaticNode InputNode>
  constexpr void inputs(InputNode const &inputNode) {
    using commonTypes = hh::tool::Intersect_t<typename InputNode::inputs_t, typename GraphType::inputs_t>;
    static_assert(std::tuple_size_v<commonTypes> != 0,
                  "The node can not be an input node, at least one of its input types should be the same of "
                  "the graph input types");

    splitInputNodeRegistration<commonTypes>(inputNode, std::make_index_sequence<std::tuple_size_v<commonTypes>>());
  }

  /// @brief Set a static node as input of the static graph for the InputType. Register the node if need be.
  /// @tparam InputType Connection type
  /// @tparam InputNode Type of the input node
  /// @param inputNode Instance of the input node
  template<class InputType, tool::HedgehogStaticNode InputNode>
  constexpr void input(InputNode const &inputNode) {
    auto typeName = hh::tool::typeToStr<InputType>();
    static_assert(
        hh::tool::isContainedInTuple_v<InputType, typename InputNode::inputs_t> &&
            hh::tool::isContainedInTuple_v<InputType, typename GraphType::inputs_t>,
        "The input type is not shared by the node and the graph.");
    registerNode(&inputNode);
    inputNodes_.insert(typeName, &inputNode);
  }

  /// @brief Set a static node as output of the static graph for all common types. Register the node if need be.
  /// @tparam OutputNode Type of the outputNode node
  /// @param outputNode Instance of the outputNode node
  template<tool::HedgehogStaticNode OutputNode>
  constexpr void outputs(OutputNode const &outputNode) {
    using commonTypes = hh::tool::Intersect_t<typename OutputNode::outputs_t, typename GraphType::outputs_t>;
    static_assert(std::tuple_size_v<commonTypes> != 0,
                  "The node can not be an output node, at least one of its output types should be the same of "
                  "the graph iouput types");
    splitOutputNodeRegistration<commonTypes>(outputNode, std::make_index_sequence<std::tuple_size_v<commonTypes>>{});
  }

  /// @brief Set a static node as Output of the static graph for the OutputType. Register the node if need be.
  /// @tparam OutputType Connection type
  /// @tparam OutputNode Type of the output node
  /// @param outputNode Output node instance
  template<class OutputType, tool::HedgehogStaticNode OutputNode>
  constexpr void output(OutputNode const &outputNode) {
    auto typeName = hh::tool::typeToStr<OutputType>();
    static_assert(
        hh::tool::isContainedInTuple_v<OutputType, typename OutputNode::outputs_t> &&
            hh::tool::isContainedInTuple_v<OutputType, typename GraphType::outputs_t>,
        "The output type is not shared by the node and the graph.");
    registerNode(&outputNode);
    outputNodes_.insert(typeName, &outputNode);
  }

  /// @brief Set edges between two nodes for all common types. Register the nodes if need be.
  /// @tparam SenderNode Type of the sender
  /// @tparam ReceiverNode Type of the receiver
  /// @param senderNode Sender node instance
  /// @param receiverNode Receiver node instance
  template<tool::HedgehogStaticNode SenderNode, tool::HedgehogStaticNode ReceiverNode>
  constexpr void edges(SenderNode const &senderNode, ReceiverNode const &receiverNode) {
    using commonTypes = hh::tool::Intersect_t<typename SenderNode::outputs_t, typename ReceiverNode::inputs_t>;
    static_assert(std::tuple_size_v<commonTypes> != 0,
                  "The edge cannot be created, there is no common type between the nodes.");

    splitEdgeRegistration<commonTypes>(senderNode,
                                       receiverNode,
                                       std::make_index_sequence<std::tuple_size_v<commonTypes>>{});
  }

  /// @brief Set an edge between two nodes for a type. Register the nodes if need be.
  /// @tparam EdgeType Type for the connection
  /// @tparam SenderNode Type of the sender
  /// @tparam ReceiverNode Type of the receiver
  /// @param senderNode Sender node instance
  /// @param receiverNode Receiver node instance
  template<class EdgeType, tool::HedgehogStaticNode SenderNode, tool::HedgehogStaticNode ReceiverNode>
  constexpr void edge(SenderNode &senderNode, ReceiverNode &receiverNode) {
    using sender_outputs_t = typename SenderNode::outputs_t;
    using receiver_inputs_t = typename ReceiverNode::inputs_t;
    using ro_receiver_inputs_t = typename ReceiverNode::ro_type_t;
    std::string typeName = hh::tool::typeToStr<EdgeType>();
    static_assert(
        hh::tool::isContainedInTuple_v<EdgeType, sender_outputs_t>
            && hh::tool::isContainedInTuple_v<EdgeType, receiver_inputs_t>,
        "The edge cannot be created, the type is not part of the sender's outputs or receiver's inputs.");

    registerNode(&senderNode);
    registerNode(&receiverNode);

    adjacencyMatrix_.at(nodeId(&senderNode)).at(nodeId(&receiverNode)).push_back(typeName);

    if constexpr (hh::tool::isContainedInTuple_v<EdgeType, ro_receiver_inputs_t>) {
      ROEdges_.at(nodeId(&senderNode)).at(nodeId(&receiverNode)).push_back(typeName);
    }

    if constexpr (std::is_const_v<EdgeType>) {
      constEdges_.at(nodeId(&senderNode)).at(nodeId(&receiverNode)).push_back(typeName);
    }
  }

  /// @brief Get the nodeId for a node
  /// @tparam Node Type of the node
  /// @param node node instance
  /// @return The node id
  template<class Node>
  requires std::is_base_of_v<hh_cx::behavior::AbstractNode, Node>
  constexpr size_t nodeId(Node *node) const {
    if (std::find(registeredNodes_.cbegin(), registeredNodes_.cend(), node) == registeredNodes_.cend()) {
      throw (std::runtime_error("The node you are trying to get does not exist in the graph."));
    }
    size_t nodeId = 0;
    for (auto registeredNode : registeredNodes_) {
      if (registeredNode == node) { return nodeId; }
      else { ++nodeId; }
    }
    return nodeId;
  }

  /// @brief Get the node from its id
  /// @param id Id to search for
  /// @return Node corresponding to the id
  [[nodiscard]] constexpr hh_cx::behavior::AbstractNode const *node(size_t id) const {
    if (id >= registeredNodes_.size()) { throw (std::runtime_error("The node you are requesting does not exist.")); }
    else { return registeredNodes_.at(id); }
  }

  /// @brief Add a test to the graph
  /// @tparam UserTest Test type
  /// @param test Test instance
  template<class UserTest>
  requires std::is_base_of_v<AbstractTest<GraphType>, UserTest>
  constexpr void addTest(UserTest *test) {
    // Add the test if not already added
    if (std::find(tests_.cbegin(), tests_.cend(), test) == tests_.cend()) {
      tests_.push_back(test);
      test->test(this);
      report_.append(test->errorMessage());
      report_.append("\n");
    }
  }

  /// @brief Test if the graph is valid against all of the tests
  /// @return True if the graph is valid against all of the test, else false
  [[nodiscard]] constexpr bool isValid() const {
    bool ret = true;
    for (auto const &test : tests_) { ret &= test->isGraphValid(); }
    return ret;
  }

 private:
  /// @brief Split all the common type between an input node and the graph and do the connection for each of them.
  /// @tparam CommonInputs Common input types
  /// @tparam T Node type
  /// @tparam Is Index of the common types
  /// @param node Node to set as input
  template<class CommonInputs, class T, std::size_t... Is>
  constexpr void splitInputNodeRegistration(T const &node, std::index_sequence<Is...>) {
    (input<std::tuple_element_t<Is, CommonInputs>>(node), ...);
  }

  /// @brief Split all the common type between an output node and the graph and do the connection for each of them.
  /// @tparam CommonOutputs Common output types
  /// @tparam T Node type
  /// @tparam Is Index of the common types
  /// @param node Node to set as output
  template<class CommonOutputs, class T, std::size_t... Is>
  constexpr void splitOutputNodeRegistration(T const &node, std::index_sequence<Is...>) {
    (output<std::tuple_element_t<Is, CommonOutputs>>(node), ...);
  }

  /// @brief Split all the common type between two nodes and do the connection for each of them.
  /// @tparam CommonTypes Common edge types
  /// @tparam S Sender node type
  /// @tparam R Receiver node type
  /// @tparam Is Index of the common types
  /// @param sender Sender instance
  /// @param receiver Receiver instance
  template<class CommonTypes, class S, class R, std::size_t... Is>
  constexpr void splitEdgeRegistration(S const &sender, R const &receiver, std::index_sequence<Is...>) {
    (edge<std::tuple_element_t<Is, CommonTypes>>(sender, receiver), ...);
  }

  /// @brief Register a node, increase the adjacency matrix size
  /// @param node Node to register
  constexpr void registerNode(hh_cx::behavior::AbstractNode const *node) {
    if (std::find(registeredNodes_.cbegin(), registeredNodes_.cend(), node) == registeredNodes_.cend()) {
      validateName(node);
      registeredNodes_.push_back(node);
      registeredNodesName_.push_back(node->name());
      adjacencyMatrix_.emplace_back();
      ROEdges_.emplace_back();
      constEdges_.emplace_back();

      for (auto &sender : adjacencyMatrix_) { sender.resize(registeredNodes_.size()); }
      for (auto &sender : ROEdges_) { sender.resize(registeredNodes_.size()); }
      for (auto &sender : constEdges_) { sender.resize(registeredNodes_.size()); }
    }
  }

  /// @brief Validate the name given to the static nodes
  /// @param node Node to validate
  constexpr void validateName(hh_cx::behavior::AbstractNode const *node) {
    if (std::any_of(
        registeredNodes_.cbegin(), registeredNodes_.cend(),
        [&node](auto const &registeredNode) { return node->name() == registeredNode->name(); })) {
      throw std::runtime_error("Another node with the same name has already been registered.");
    }
  }
};
}

#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_CX_GRAPH_H_
