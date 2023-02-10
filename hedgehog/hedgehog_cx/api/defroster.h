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

#ifndef HEDGEHOG_CX_DEFROSTER_H_
#define HEDGEHOG_CX_DEFROSTER_H_

#ifdef HH_ENABLE_HH_CX

#include <array>
#include <ostream>
#include "../tools/type_nodes_map_array.h"
#include "../tools/concepts.h"
#include "graph.h"

/// @brief Hedgehog compile-time namespace
namespace hh_cx {

/// @brief The defroster has an analyser role during the compile time analysis and will convert the compile-time
/// representation to the Hedgehog dynamic Graph. At construction, the structural information from the CXGraph will
/// be extracted and the compile time tests will be invoked. The validity of the graph and the error messages will be updated
/// consequently. Finally, the defroster is used to convert its internal representation of the CXGraph to hedgehog dynamic
/// Hedgehog graph.
/// @attention The easiest way to create a defroster is to use createDefroster with a pointer to a constexpr callable creating a compile-time graph
/// @tparam DynGraphType Graph type
/// @tparam name_t Type of the flatten graph name std::array<char, graphNameSize>
/// @tparam report_t Type of the flatten report std::array<char, reportSize>
/// @tparam adjacencyMatrix_t Type of the flatten graph adjacency matrix std::array<std::array<std::array<std::array<char, maxEdgeTypeSize>, maxNumberEdges>, numberNodes>, numberNodes>
/// @tparam registeredNodesName_t Type of the flatten registered nodes names std::array<std::array<char, maxNodeNameSize>, numberNodes>
/// @tparam inputMap_t Type of the flatten graph input nodes name hh_cx::tool::TypeNodesMapArray<nbTypesInput, maxTypeSizeInput, maxNumberNodesInput, maxSizeNameInput>
/// @tparam outputMap_t Type of the flatten graph output nodes name hh_cx::tool::TypeNodesMapArray<nbTypesOutput, maxTypeSizeOutput, maxNumberNodesOutput, maxSizeNameOutput>
template<class DynGraphType, class name_t, class report_t, class adjacencyMatrix_t, class registeredNodesName_t, class inputMap_t, class outputMap_t>
class Defroster {
 private:
  name_t const name_; ///< Flattened graph's name
  bool const isValid_; ///< Graph's validity
  report_t const report_; ///< Flattened graph's report
  adjacencyMatrix_t const adjacencyMatrix_; ///< Flattened graph's adjacency matrix
  registeredNodesName_t const registeredNodesName_; ///< Flattened graph's nodes name
  inputMap_t const inputMap_; ///< Flattened graph's input map
  outputMap_t const outputMap_; ///< Flattened graph's output map
 public:
  /// @brief Defroster constructor from flattened graph's data structures
  /// @attention The easiest way to create a defroster is to use createDefroster with a pointer to a constexpr callable creating a compile-time graph
  /// @param name Flattened graph's name
  /// @param isValid Graph's validity
  /// @param report Flattened graph's report
  /// @param adjacencyMatrix Flattened graph's adjacency matrix
  /// @param registeredNodesName Flattened graph's nodes name
  /// @param inputMap Flattened graph's input map
  /// @param outputMap Flattened graph's output map
  constexpr Defroster(
      name_t const name,
      bool const isValid,
      report_t const &report,
      adjacencyMatrix_t const &adjacencyMatrix,
      registeredNodesName_t const &registeredNodesName,
      inputMap_t const &inputMap,
      outputMap_t const &outputMap)
      : name_(name),
        isValid_(isValid),
        report_(report),
        adjacencyMatrix_(adjacencyMatrix),
        registeredNodesName_(registeredNodesName),
        inputMap_(inputMap),
        outputMap_(outputMap) {}

  /// @brief Default destructor
  constexpr virtual ~Defroster() = default;

  /// @brief Accessor to the graph's name
  /// @return Graph's name (std::string)
  [[nodiscard]] std::string graphName() const { return std::string(name_.data()); }

  /// @brief Accessor to the graph validity
  /// @return True if the graph has passed all of its tests, else false
  [[nodiscard]] constexpr bool isValid() const { return isValid_; }

  /// @brief Accessor to the report of all tests
  /// @return The tests report
  [[nodiscard]] std::string report() const { return std::string(report_.data()); }

  /// @brief Convert hedgehog representation of the hh::cx::Graph to hedgehog dynamic hh::Graph thanks to the nodes instances in parameter.
  /// @details "The convert function need an even number of arguments, following the pattern: id1,
  /// instanceDynamicNode1, id2, instanceDynamicNode2..."
  /// @attention The ids should be the name given to the static nodes representing the dynamic nodes
  /// @tparam Args Types of the mapped node.
  /// @param args The parameters should follow the pattern: instanceCXNode1, instanceDynamicNode1, instanceCXNode2,
  /// instanceDynamicNode2 ect.. Their numbers should be even.
  /// @return A dynamic hh::Graph created from the hh::cx::Graph structure and the mapped dynamic instances.
  template<class ...Args>
  auto map(Args... args) const {
    if constexpr (sizeof...(args) > 0 && sizeof...(args) % 2 != 0) {
      throw std::runtime_error(
          "The map function only accepts an even number of parameters as follows: name1, dynNode1, name2, dynNode2...");
    }
    std::set<std::string> validatedNames{};
    auto [names, dynNodes] = validateAndSplit(validatedNames, args...);
    return generateGraph(names, dynNodes);
  }

 private:
  /// @brief Validate and split the arguments given to the map function
  /// @tparam NodeName First type of the given id
  /// @tparam Node First type of the given dynamic node
  /// @tparam Args Rest of the argument types
  /// @param setName Set of ids already parsed
  /// @param nodeName Current id
  /// @param node Current node
  /// @param args Rest of the arguments
  /// @return Pair containing 1) a vector of ids and 2) A tuple with the real dynamic nodes
  template<class NodeName, tool::HedgehogConnectableNode Node, class ...Args>
  requires std::is_convertible_v<NodeName, std::string>
  auto validateAndSplit(
      std::set<std::string> &setName,
      NodeName const &nodeName, std::shared_ptr<Node> node, Args... args) const {
    bool nameFound = false;
    std::string name = nodeName;
    for (size_t nameId = 0; nameId < registeredNodesName_.size() && !nameFound; ++nameId) {
      if (std::string(registeredNodesName_.at(nameId).data()) == name) { nameFound = true; }
    }
    if (!nameFound) {
      std::ostringstream oss;
      oss << "The node name identifier \"" << name
          << "\" has not been found in the list of names identifier of static nodes.";
      throw std::runtime_error(oss.str());
    } else {
      if (setName.insert(name).second == false) {
        std::ostringstream oss;
        oss << "The node name identifier \"" << name << "\" has already been used to map a dynamic node.";
        throw std::runtime_error(oss.str());
      }
      if constexpr (sizeof...(args) > 0) {
        auto [names, dynNodes] = validateAndSplit(setName, args...);
        std::vector<std::string> newNames{nodeName};
        newNames.insert(newNames.end(), names.cbegin(), names.cend());
        return std::make_pair(
            newNames,
            std::tuple_cat(std::tuple<std::shared_ptr<Node>>{node}, dynNodes)
        );
      } else {
        if (registeredNodesName_.size() != setName.size()) {
          std::ostringstream oss;
          oss << "There are static nodes that are not mapped.";
          throw std::runtime_error(oss.str());
        }
        return std::make_pair(
            std::vector<std::string>{nodeName},
            std::tuple<std::shared_ptr<Node>>{node}
        );
      }
    }
  }

  /// @brief Fallback case if the types of the inputs are not valid
  /// @tparam T Type of the first argument
  /// @tparam U Type of the second arguments
  /// @tparam Args Rest of the arguments
  /// @throws std::runtime_error in all cases
  /// @return Nothing, throw an error
  template<class T, class U, class ...Args>
  auto validateAndSplit(std::set<std::string> &, T const &, U const &, Args...) const {
    std::ostringstream oss;
    oss << "The map function has not the rights type, the sequence of types " << hh::tool::typeToStr<T>() << " "
        << hh::tool::typeToStr<U>()
        << " has been found";
    throw std::runtime_error(oss.str());
  }

  /// @brief Generate a dynamic graph from the list of ids and list of dynamic nodes from the map function
  /// @tparam DynamicNodes Types of the dynamic nodes
  /// @param dynamicNameIds List of dynamic nodes-ids
  /// @param dynamicNodes List of dynamic nodes
  /// @return A dynamic graph
  template<class DynamicNodes>
  auto generateGraph(std::vector<std::string> const &dynamicNameIds,
                     DynamicNodes const &dynamicNodes) const {
    auto graph = std::make_shared<DynGraphType>(this->graphName());
    std::vector<typename hh_cx::tool::UniqueVariantFromTuple_t<DynamicNodes>> dynamicNodesVariant;
    populateVariants(dynamicNodes, dynamicNodesVariant);

    // Deal with input nodes
    setInputNodes<DynamicNodes>(graph, dynamicNameIds, dynamicNodesVariant);

    // Deal with output nodes
    setOutputNodes<DynamicNodes>(graph, dynamicNameIds, dynamicNodesVariant);

    // Deal with edges
    setEdges<DynamicNodes>(graph, dynamicNameIds, dynamicNodesVariant);

    return graph;
  }

  /// @brief Set the nodes as input of the graph following the structure of the static graph
  /// @tparam DynamicNodes Types of the dynamic nodes
  /// @param graph Dynamic graph
  /// @param dynamicNameIds List of dynamic nodes-ids
  /// @param dynamicNodesVariant List of dynamic nodes (under the form a variants)
  template<class DynamicNodes>
  void setInputNodes(
      std::shared_ptr<DynGraphType> graph,
      std::vector<std::string> const &dynamicNameIds,
      std::vector<typename hh_cx::tool::UniqueVariantFromTuple_t<DynamicNodes>> dynamicNodesVariant) const {
    using GraphInputsTuple = typename DynGraphType::inputs_t;

    // For all ids for a type
    for (auto const &inputNodeNameToTypes : inputMap_.mapNodeNameToTypeNames()) {
      // Get the variant
      auto staticNameId = inputNodeNameToTypes.first;
      auto positionDynNode =
          std::distance(
              dynamicNameIds.begin(),
              std::find(dynamicNameIds.begin(), dynamicNameIds.end(), staticNameId)
          );
      auto dynamicNode = dynamicNodesVariant.at(positionDynNode);
      // Visit the variant
      std::visit([&](auto const &node) {
        // Get the common types between the node and the graph
        using IntersectionInputTypes =
            hh::tool::Intersect_t<GraphInputsTuple, typename std::remove_reference_t<decltype(*node)>::inputs_t>;
        // For all of these types
        for (auto const &typeName : inputNodeNameToTypes.second) {
          // If the connection type from the map is in the common types
          if (isTypeNameInTuple<IntersectionInputTypes>(
              typeName, std::make_index_sequence<std::tuple_size_v<IntersectionInputTypes>>{})
              ) {
            // Set the node as input
            setNodeAsGraphInput<IntersectionInputTypes>(
                typeName, node, graph, std::make_index_sequence<std::tuple_size_v<IntersectionInputTypes>>{}
            );
          } else {
            // else throw an error
            std::ostringstream oss;
            oss << "Problem during mapping: the node " << node->name() << " cannot be set as input of the graph " <<
                graphName() << " for the type " << typeName;
            throw std::runtime_error(oss.str());
          }
        }
      }, dynamicNode);
    }
  }

  /// @brief Set the nodes as output of the graph following the structure of the static graph
  /// @tparam DynamicNodes Types of the dynamic nodes
  /// @param graph Dynamic graph
  /// @param dynamicNameIds List of dynamic nodes-ids
  /// @param dynamicNodesVariant List of dynamic nodes (under the form a variants)
  template<class DynamicNodes>
  void setOutputNodes(
      std::shared_ptr<DynGraphType> graph,
      std::vector<std::string> const &dynamicNameIds,
      std::vector<typename hh_cx::tool::UniqueVariantFromTuple_t<DynamicNodes>> dynamicNodesVariant) const {
    using GraphOutputsTuple = typename DynGraphType::outputs_t;
    // For all ids for a type
    for (auto const &outputNodeNameToTypes : outputMap_.mapNodeNameToTypeNames()) {
      // Get the variant
      auto staticNameId = outputNodeNameToTypes.first;
      auto positionDynNode =
          std::distance(
              dynamicNameIds.begin(),
              std::find(dynamicNameIds.begin(), dynamicNameIds.end(), staticNameId)
          );
      auto dynamicNode = dynamicNodesVariant.at(positionDynNode);
      // Visit the variant
      std::visit([&](auto const &node) {
        // Get the common types between the node and the graph
        using IntersectionOutputTypes =
            hh::tool::Intersect_t<GraphOutputsTuple, typename std::remove_reference_t<decltype(*node)>::outputs_t>;
        // For all of these types
        for (auto const &typeName : outputNodeNameToTypes.second) {
          // If the connection type from the map is in the common types
          if (isTypeNameInTuple<IntersectionOutputTypes>(
              typeName, std::make_index_sequence<std::tuple_size_v<IntersectionOutputTypes>>{})
              ) {
            // Set the node as input
            setNodeAsGraphOutput<IntersectionOutputTypes>(
                typeName, node, graph, std::make_index_sequence<std::tuple_size_v<IntersectionOutputTypes>>{}
            );
          } else {
            // else throw an error
            std::ostringstream oss;
            oss << "Problem during mapping: the node " << node->name() << " cannot be set as input of the graph " <<
                graphName() << " for the type " << typeName;
            throw std::runtime_error(oss.str());
          }
        }
      }, dynamicNode);
    }
  }

  /// @brief Set the edges of the graph following the structure of the static graph
  /// @tparam DynamicNodes Types of the dynamic nodes
  /// @param graph Dynamic graph
  /// @param dynamicNameIds List of dynamic nodes-ids
  /// @param dynamicNodesVariant List of dynamic nodes (under the form a variants)
  template<class DynamicNodes>
  void setEdges(
      std::shared_ptr<DynGraphType> graph,
      std::vector<std::string> const &dynamicNameIds,
      std::vector<typename hh_cx::tool::UniqueVariantFromTuple_t<DynamicNodes>> dynamicNodesVariant) const {

    // For all nodes as sender
    for (size_t senderId = 0; senderId < adjacencyMatrix_.size(); ++senderId) {
      // Get the name-id
      auto staticSenderNodeNameId = std::string(registeredNodesName_.at(senderId).data());
      // Get the dynamic variant
      auto positionSenderDynNode =
          std::distance(
              dynamicNameIds.begin(),
              std::find(dynamicNameIds.begin(), dynamicNameIds.end(), staticSenderNodeNameId)
          );
      auto dynamicSenderNodeVariant = dynamicNodesVariant.at(positionSenderDynNode);\
      // Visit the variant
      std::visit([&](auto const &dynamicSenderNode) {
        // For all the nodes as receiver
        for (size_t receiverId = 0; receiverId < adjacencyMatrix_.at(senderId).size(); ++receiverId) {
          // Get the name-id
          auto staticReceiverNodeNameId = std::string(registeredNodesName_.at(receiverId).data());
          // Get the variant
          auto positionReceiverDynNode =
              std::distance(
                  dynamicNameIds.begin(),
                  std::find(dynamicNameIds.begin(), dynamicNameIds.end(), staticReceiverNodeNameId)
              );
          auto dynamicReceiverNodeVariant = dynamicNodesVariant.at(positionReceiverDynNode);
          // Visit the vriant
          std::visit([&](auto const &dynamicReceiverNode) {
            // Get the common types between the 2 nodes
            using commonTypesDynamicNodes =
                hh::tool::Intersect_t<
                    typename std::remove_reference_t<decltype(*dynamicSenderNode)>::outputs_t,
                    typename std::remove_reference_t<decltype(*dynamicReceiverNode)>::inputs_t>;

            // For all the types in the adjacency matrix
            for (size_t typeId = 0; typeId < adjacencyMatrix_.at(senderId).at(receiverId).size(); ++typeId) {
              auto typeName = std::string(adjacencyMatrix_.at(senderId).at(receiverId).at(typeId).data());
              if (!typeName.empty()) {
                // If the connexion type is in the common types
                if (isTypeNameInTuple<commonTypesDynamicNodes>(
                    typeName, std::make_index_sequence<std::tuple_size_v<commonTypesDynamicNodes>>{})
                    ) {
                  // Set the edge
                  setEdge<commonTypesDynamicNodes>(
                      typeName,
                      dynamicSenderNode,
                      dynamicReceiverNode,
                      graph,
                      std::make_index_sequence<std::tuple_size_v<commonTypesDynamicNodes>>{}
                  );
                } else {
                  // else, throw an error
                  std::ostringstream oss;
                  oss
                      << "Problem during mapping: the node " << dynamicSenderNode->name()
                      << " cannot be set linked to " << dynamicReceiverNode->name()
                      << " for the type " << typeName;
                  throw std::runtime_error(oss.str());
                }
              }
            }
          }, dynamicReceiverNodeVariant);
        }
      }, dynamicSenderNodeVariant);
    }
  }

  /// @brief Populate a vector of variants from a tuple
  /// @tparam Position Position in the tuple
  /// @tparam Tuple Tuple type to transform in a variant
  /// @tparam VectorVariant Type of vector of variants
  /// @param tup Tuple to transform
  /// @param vectorVariant Vector of vriants
  template<size_t Position = 0, class Tuple, class VectorVariant>
  void populateVariants(Tuple const &tup, VectorVariant &vectorVariant) const {
    if constexpr (Position < std::tuple_size_v<Tuple>) {
      vectorVariant.emplace_back(std::get<Position>(tup));
      populateVariants<Position + 1, Tuple, VectorVariant>(tup, vectorVariant);
    }
  }

  /// @brief Decomposition strategy of the tuple of common types to set the node as input of the graph for the typeName
  /// @tparam TupleOfTypes Tuple containing all the common types between the graph and the node
  /// @tparam NodeType Type of the node
  /// @tparam GraphType Type of the graph
  /// @tparam Indices Tuple indices
  /// @param typeName Type name to do the connection
  /// @param node Node to connect as input
  /// @param graph Dynamic graph
  template<class TupleOfTypes, class NodeType, class GraphType, size_t... Indices>
  void setNodeAsGraphInput(std::string const &typeName,
                           NodeType &node,
                           GraphType &graph,
                           std::index_sequence<Indices...>) const {
    (setNodeAsGraphInputType<std::tuple_element_t<Indices, TupleOfTypes>>(typeName, node, graph), ...);
  }

  /// @brief  Set the node as input for the type T if T as string is equal to typeName
  /// @tparam T Type to maybe set the node as input
  /// @tparam NodeType Type of the node
  /// @tparam GraphType Type of the graph
  /// @param typeName Name of the type to set
  /// @param node Node to set as input
  /// @param graph Dynamic graph
  template<class T, class NodeType, class GraphType>
  void setNodeAsGraphInputType(std::string const &typeName, NodeType &node, GraphType &graph) const {
    if (hh::tool::typeToStr<T>() == typeName) { graph->template input<T>(node); }
  }

  /// @brief Decomposition strategy of the tuple of common types to set the node as output of the graph for the typeName
  /// @tparam TupleOfTypes Tuple containing all the common types between the graph and the node
  /// @tparam NodeType Type of the node
  /// @tparam GraphType Type of the graph
  /// @tparam Indices Tuple indices
  /// @param typeName Type name to do the connection
  /// @param node Node to connect as output
  /// @param graph Dynamic graph
  template<class TupleOfTypes, class NodeType, class GraphType, size_t... Indices>
  void setNodeAsGraphOutput(std::string const &typeName,
                            NodeType &node,
                            GraphType &graph,
                            std::index_sequence<Indices...>) const {
    (setNodeAsGraphOutputType<std::tuple_element_t<Indices, TupleOfTypes>>(typeName, node, graph), ...);
  }

  /// @brief  Set the node as output for the type T if T as string is equal to typeName
  /// @tparam T Type to maybe set the node as output
  /// @tparam NodeType Type of the node
  /// @tparam GraphType Type of the graph
  /// @param typeName Name of the type to set
  /// @param node Node to set as input
  /// @param graph Dynamic graph
  template<class T, class NodeType, class GraphType>
  void setNodeAsGraphOutputType(std::string const &typeName, NodeType &node, GraphType &graph) const {
    if (hh::tool::typeToStr<T>() == typeName) { graph->template output<T>(node); }
  }

  /// @brief Decomposition strategy of the tuple of common types to set the edge between two nodes for the typeName
  /// @tparam TupleOfTypes Tuple containing all the common types between the graph and the node
  /// @tparam SenderNodeType Type of the sender node
  /// @tparam ReceiverNodeType Type of the receiver node
  /// @tparam GraphType Type of the graph
  /// @tparam Indices Tuple indices
  /// @param typeName Type name to do the connection
  /// @param sender Sender node
  /// @param receiver Receiver node
  /// @param graph Dynamic graph
  template<class TupleOfTypes, class SenderNodeType, class ReceiverNodeType, class GraphType, size_t... Indices>
  void setEdge(std::string const &typeName,
               SenderNodeType &sender,
               ReceiverNodeType &receiver,
               GraphType &graph,
               std::index_sequence<Indices...>) const {
    (setEdgeType<std::tuple_element_t<Indices, TupleOfTypes>>(typeName, sender, receiver, graph), ...);
  }

  /// @brief Set an edge between two nodes for the Type T if T's string is equal to typeName
  /// @tparam T Connection type
  /// @tparam SenderNodeType Type of the sender
  /// @tparam ReceiverNodeType Type of the receiver
  /// @tparam GraphType Graph's type
  /// @param typeName Type's name
  /// @param sender Sender node
  /// @param receiver Receiver node
  /// @param graph Dynamic graph
  template<class T, class SenderNodeType, class ReceiverNodeType, class GraphType>
  void setEdgeType(std::string const &typeName,
                   SenderNodeType &sender,
                   ReceiverNodeType &receiver,
                   GraphType &graph) const {
    if (hh::tool::typeToStr<T>() == typeName) { graph->template edge<T>(sender, receiver); }
  }

  /// @brief Test if a type name is equal to one of the types as string in a tuple
  /// @tparam TupleOfTypes Tuple of types
  /// @tparam Indices Indices to visit the tuple
  /// @param typeName Type name
  /// @return True is one ot the types as string in the tuple is equal to typeName, else false
  template<class TupleOfTypes, size_t... Indices>
  [[nodiscard]] bool isTypeNameInTuple(std::string const &typeName, std::index_sequence<Indices...>) const {
    return (isSameTypeName<std::tuple_element_t<Indices, TupleOfTypes>>(typeName) || ...);
  }

  /// @brief Test if a type T as a string is equal to typeName
  /// @tparam T Type to test
  /// @param typeName Type name to test against
  /// @return True if the type as string is equal to typeName, else false
  template<class T>
  [[nodiscard]] inline bool isSameTypeName(std::string const &typeName) const {
    return hh::tool::typeToStr<T>() == typeName;
  }
};

/// @brief Create a defroster from a static hedgehog graph (hh_cx::Graph)
/// @tparam FctGraph Function generating a hh_cx::Graph
/// @code
/// // Nodes definitions
/// class TaskIFD : public hh::AbstractTask<3, int, float, double, int, float, double> {
///  public:
///   void execute(std::shared_ptr<int> data) override { this->addResult(data); }
///   void execute(std::shared_ptr<float> data) override { this->addResult(data); }
///   void execute(std::shared_ptr<double> data) override { this->addResult(data); }
/// };
///
/// class TaskCOutput : public hh::AbstractTask<3, int, float, double, int const, float const, double const> {
///  public:
///   void execute(std::shared_ptr<int> data) override { this->addResult(std::make_shared<int const>(*data)); }
///   void execute(std::shared_ptr<float> data) override { this->addResult(std::make_shared<float const>(*data)); }
///   void execute(std::shared_ptr<double> data) override { this->addResult(std::make_shared<double const>(*data)); }
/// };
///
/// class TaskCIntput : public hh::AbstractTask<3, int const, float const, double const, int, float, double> {
///  public:
///   void execute(std::shared_ptr<int const> data) override { this->addResult(std::make_shared<int>(*data)); }
///   void execute(std::shared_ptr<float const> data) override { this->addResult(std::make_shared<float>(*data)); }
///   void execute(std::shared_ptr<double const> data) override { this->addResult(std::make_shared<double>(*data)); }
/// };
///
/// class GraphIFD : public hh::Graph<3, int, float, double, int, float, double> {
///  public:
///   explicit GraphIFD(std::string const &name = "graph") : hh::Graph<3, int, float, double, int, float, double>(name) {}
/// };
///
/// // constexpr function generating a hh_cx::Graph
/// constexpr auto constructGraphComplex() {
///   hh_cx::Node<TaskIFD, float, double>
///       nodeI_IFD("I_IFD"),
///       nodeO_IFD("O_IFD"),
///       nodeI_I("I_I"),
///       nodeI_F("I_F"),
///       nodeI_D("I_D"),
///       nodeO_I("O_I"),
///       nodeO_F("O_F"),
///       nodeO_D("O_D");
///
///   hh_cx::Node<TaskCOutput> nodeI_CIFD("I_CIFD");
///   hh_cx::Node<TaskCIntput> nodeO_CIFD("O_CIFD");
///
///   hh_cx::Graph<GraphIFD> graph{"graph"};
///
///   graph.input<int>(nodeI_I);
///   graph.input<float>(nodeI_F);
///   graph.input<double>(nodeI_D);
///   graph.inputs(nodeI_IFD);
///   graph.inputs(nodeI_CIFD);
///
///   graph.output<int>(nodeO_I);
///   graph.output<float>(nodeO_F);
///   graph.output<double>(nodeO_D);
///   graph.outputs(nodeO_IFD);
///   graph.outputs(nodeO_CIFD);
///
///   graph.edge<int>(nodeI_I, nodeO_I);
///   graph.edge<int>(nodeI_I, nodeO_IFD);
///
///   graph.edge<float>(nodeI_F, nodeO_F);
///   graph.edge<float>(nodeI_F, nodeO_IFD);
///
///   graph.edge<double>(nodeI_D, nodeO_D);
///   graph.edge<double>(nodeI_D, nodeO_IFD);
///
///   graph.edges(nodeI_IFD, nodeO_IFD);
///   graph.edges(nodeI_CIFD, nodeO_CIFD);
///
///   auto dataRaceTest = hh_cx::DataRaceTest < GraphIFD > {};
///   auto cycleTest = hh_cx::CycleTest < GraphIFD > {};
///   graph.addTest(&dataRaceTest);
///   graph.addTest(&cycleTest);
///   return graph;
/// }
///
/// int main() {
///   // Calling createDefroster with the pointer to the function constructGraphComplex
///   constexpr auto defrosterComplex = hh_cx::createDefroster<&constructGraphComplex>();
///   static_assert(!defrosterComplex.isValid(), "Valid");
///   if (!defrosterComplex.isValid()) { std::cout << defrosterComplex.report() << std::endl; }
///
///   auto complexGraph = defrosterComplex.map(
///       "I_IFD", std::make_shared<TaskIFD>(),
///       "O_IFD", std::make_shared<TaskIFD>(),
///       "I_I", std::make_shared<TaskIFD>(),
///       "I_F", std::make_shared<TaskIFD>(),
///       "I_D", std::make_shared<TaskIFD>(),
///       "O_I", std::make_shared<TaskIFD>(),
///       "O_F", std::make_shared<TaskIFD>(),
///       "O_D", std::make_shared<TaskIFD>(),
///       "I_CIFD", std::make_shared<TaskCOutput>(),
///       "O_CIFD", std::make_shared<TaskCIntput>()
///   );
///
///   complexGraph->createDotFile("complexGraph.dot",
///                               hh::ColorScheme::EXECUTION,
///                               hh::StructureOptions::ALL,
///                               hh::DebugOptions::ALL);
/// }
/// @endcode
/// @return The defroster
template<auto FctGraph>
constexpr auto createDefroster() {
  // Static graph
  auto graph = FctGraph();
  static_assert(
      std::is_base_of_v<hh_cx::Graph<typename decltype(graph)::dynamic_node_t>, decltype(graph)>,
      "The callable given to the createDefroster function should 1) be a constexpr function and 2) return a valid hh_cx::Graph");
  using DynGraphType = typename decltype(graph)::dynamic_node_t;


  // Static sizes
  constexpr size_t
      graphNameSize = FctGraph().name().size() + 1,
      reportSize = FctGraph().report().size() + 1,
      numberNodes = FctGraph().numberNodesRegistered(),
      maxNodeNameSize = FctGraph().maxNodeNameSize() + 1,
      maxNumberEdges = FctGraph().maxEdgeSizes().first,
      maxEdgeTypeSize = FctGraph().maxEdgeSizes().second,
      nbTypesInput = FctGraph().inputNodes().nbTypes(),
      maxTypeSizeInput = FctGraph().inputNodes().maxTypeSize() + 1,
      maxNumberNodesInput = FctGraph().inputNodes().maxNumberNodes(),
      maxSizeNameInput = FctGraph().inputNodes().maxSizeName() + 1,
      nbTypesOutput = FctGraph().outputNodes().nbTypes(),
      maxTypeSizeOutput = FctGraph().outputNodes().maxTypeSize() + 1,
      maxNumberNodesOutput = FctGraph().outputNodes().maxNumberNodes(),
      maxSizeNameOutput = FctGraph().outputNodes().maxSizeName() + 1;

  // Get Graph name
  auto const &graphName = graph.name();
  std::array<char, graphNameSize> nameArray;
  std::copy(graphName.cbegin(), graphName.cend(), nameArray.begin());
  nameArray.at(graphName.size()) = '\0';

  // Get graph validity
  bool isGraphValid = graph.isValid();

  // Copy reports
  auto const &reportString = graph.report();
  std::array<char, reportSize> reportArray{};
  std::copy(reportString.begin(), reportString.end(), reportArray.begin());
  reportArray.at(reportString.size()) = '\0';

  // Copy adjacency matrix
  auto const &adjacencyMatrix = graph.adjacencyMatrix();
  std::array<std::array<std::array<std::array<char, maxEdgeTypeSize>, maxNumberEdges>, numberNodes>, numberNodes>
      adjacencyMatrixArray{};

  for (size_t senderId = 0; senderId < adjacencyMatrix.size(); ++senderId) {
    for (size_t receiverId = 0; receiverId < adjacencyMatrix.at(senderId).size(); ++receiverId) {
      for (size_t typeId = 0; typeId < adjacencyMatrix.at(senderId).at(receiverId).size(); ++typeId) {
        std::copy(
            adjacencyMatrix.at(senderId).at(receiverId).at(typeId).cbegin(),
            adjacencyMatrix.at(senderId).at(receiverId).at(typeId).cend(),
            adjacencyMatrixArray.at(senderId).at(receiverId).at(typeId).begin()
        );
      }
    }
  }

  // Copy registered nodes
  std::array<std::array<char, maxNodeNameSize>, numberNodes> registeredNodesNameArray;
  auto const &registeredNodesName = graph.registeredNodesName();
  for (size_t nodeId = 0; nodeId < numberNodes; ++nodeId) {
    std::copy(
        registeredNodesName.at(nodeId).cbegin(), registeredNodesName.at(nodeId).cend(),
        registeredNodesNameArray.at(nodeId).begin()
    );
    registeredNodesNameArray.at(nodeId).at(registeredNodesName.at(nodeId).size()) = '\0';
  }

  // Copy Input nodes
  auto inputTypeNodesMap = graph.inputNodes();
  hh_cx::tool::TypeNodesMapArray<nbTypesInput, maxTypeSizeInput, maxNumberNodesInput, maxSizeNameInput>
      inputTypeNodesMapArray{};
  for (size_t typeId = 0; typeId < inputTypeNodesMap.nbTypes(); ++typeId) {
    std::copy(
        inputTypeNodesMap.types().at(typeId).cbegin(), inputTypeNodesMap.types().at(typeId).cend(),
        inputTypeNodesMapArray.types().at(typeId).begin()
    );
    for (size_t nodeId = 0; nodeId < inputTypeNodesMap.nodesName().at(typeId).size(); ++nodeId) {
      std::copy(
          inputTypeNodesMap.nodesName().at(typeId).at(nodeId).cbegin(),
          inputTypeNodesMap.nodesName().at(typeId).at(nodeId).cend(),
          inputTypeNodesMapArray.nodes().at(typeId).at(nodeId).begin()
      );
    }
  }

  // Copy Output nodes
  auto outputTypeNodesMap = graph.outputNodes();
  hh_cx::tool::TypeNodesMapArray<nbTypesOutput, maxTypeSizeOutput, maxNumberNodesOutput, maxSizeNameOutput>
      outputTypeNodesMapArray{};
  for (size_t typeId = 0; typeId < outputTypeNodesMap.nbTypes(); ++typeId) {
    std::copy(
        outputTypeNodesMap.types().at(typeId).cbegin(), outputTypeNodesMap.types().at(typeId).cend(),
        outputTypeNodesMapArray.types().at(typeId).begin()
    );
    for (size_t nodeId = 0; nodeId < outputTypeNodesMap.nodesName().at(typeId).size(); ++nodeId) {
      std::copy(
          outputTypeNodesMap.nodesName().at(typeId).at(nodeId).cbegin(),
          outputTypeNodesMap.nodesName().at(typeId).at(nodeId).cend(),
          outputTypeNodesMapArray.nodes().at(typeId).at(nodeId).begin()
      );
    }
  }

  return hh_cx::Defroster<
      DynGraphType,
      decltype(nameArray),
      decltype(reportArray),
      decltype(adjacencyMatrixArray),
      decltype(registeredNodesNameArray),
      decltype(inputTypeNodesMapArray),
      decltype(outputTypeNodesMapArray)
  >(
      nameArray,
      isGraphValid, reportArray,
      adjacencyMatrixArray, registeredNodesNameArray,
      inputTypeNodesMapArray, outputTypeNodesMapArray);
}
}
#endif //HH_ENABLE_HH_CX

#endif //HEDGEHOG_CX_DEFROSTER_H_
