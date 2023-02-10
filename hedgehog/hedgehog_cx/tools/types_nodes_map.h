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

#ifndef HEDGEHOG_CX_TYPES_NODES_MAP_H_
#define HEDGEHOG_CX_TYPES_NODES_MAP_H_

#ifdef HH_ENABLE_HH_CX

#include <vector>
#include <algorithm>
#include "../behavior/abstract_node.h"

/// @brief Hedgehog compile-time namespace
namespace hh_cx {

/// Hedgehog tool namespace
namespace tool {

/// @brief Constexpr map from type names to vector of nodes
class TypesNodesMap {
 private:
  std::vector<std::string> types_{}; ///< Types registered to the map
  std::vector<std::vector<behavior::AbstractNode const *>> nodes_{}; ///< Corresponding nodes
  std::vector<std::vector<std::string>> nodesName_{}; ///< Corresponding nodes name

 public:
  /// @brief Default constructor
  constexpr TypesNodesMap() = default;
  /// @brief Default destructor
  constexpr ~TypesNodesMap() = default;

  /// @brief Accessor to the number of types (keys) in the map
  /// @return The number of  of types
  [[nodiscard]] constexpr size_t nbTypes() const { return types_.size(); }

  /// @brief Accessor to the types
  /// @return Types of the map
  [[nodiscard]] constexpr std::vector<std::string> const &types() const { return types_; }

  /// @brief Accessor to all the corresponding nodes
  /// @return All the corresponding nodes
  [[nodiscard]] constexpr std::vector<std::vector<behavior::AbstractNode const *>> const &nodes() const {
    return nodes_;
  }

  /// @brief Accessor to all the corresponding nodes name
  /// @return All the corresponding nodes name
  [[nodiscard]] constexpr std::vector<std::vector<std::string>> const &nodesName() const { return nodesName_; }

  /// @brief Accessor to the corresponding nodes for a type
  /// @param type Type (Key) to look for
  /// @return The list of nodes linked to the type
  [[nodiscard]] constexpr std::vector<behavior::AbstractNode const *> nodes(std::string const &type) const {
    auto posIt = std::find(types_.cbegin(), types_.cend(), type);
    if (posIt != types_.cend()) { return nodes_.at(static_cast<unsigned long>(std::distance(types_.cbegin(), posIt))); }
    else { return {}; }
  }

  /// @brief Clear the map
  constexpr void clear() {
    types_.clear();
    for (auto &nodes : nodes_) { nodes.clear(); }
    for (auto &nodesName : nodesName_) { nodesName.clear(); }
  }

  /// @brief Insert a node for a type inside of the map
  /// @param type Type (Key) to add
  /// @param node Node to add
  constexpr void insert(std::string const &type, hh_cx::behavior::AbstractNode const *node) {
    auto posIt = std::find(types_.cbegin(), types_.cend(), type);
    if (posIt != types_.cend()) {
      nodes_.at(static_cast<unsigned long>(std::distance(types_.cbegin(), posIt))).push_back(node);
      nodesName_.at(static_cast<unsigned long>(std::distance(types_.cbegin(), posIt))).push_back(node->name());
    } else {
      types_.push_back(type);
      nodes_.push_back(std::vector<behavior::AbstractNode const *>{node});
      nodesName_.push_back(std::vector<std::string>{node->name()});
    }
  }

  /// @brief Test if the type (key) exists
  /// @param type Type to test
  /// @return True if the type exists, else false
  [[nodiscard]] constexpr bool contains(std::string const &type) const {
    return std::find(types_.cbegin(), types_.cend(), type) != types_.cend();
  }

  /// @brief Test if a node exists for a given type (key)
  /// @param type Type (Key) to test
  /// @param node Node to test
  /// @return True if the node exists for a given type, else false
  [[nodiscard]] constexpr bool contains(std::string const &type, hh_cx::behavior::AbstractNode const *node) const {
    if (contains(type)) {
      auto nodes = this->nodes(type);
      if (std::find(nodes.cbegin(), nodes.cend(), node) == nodes.cend()) { return false; }
      else { return true; }
    } else { return false; }
  }

  /// @brief Return the maximum type name size (used to create TypeNodesMapArray)
  /// @return Maximum type name size
  [[nodiscard]] constexpr size_t maxTypeSize() const {
    if(types_.empty()) { return 0; }
    return std::max_element(
        types_.cbegin(), types_.cend(),
        [](auto const &lhs, auto const &rhs) { return lhs.size() < rhs.size(); }
    )->size();
  }

  /// @brief Return the maximum number of nodes (used to create TypeNodesMapArray)
  /// @return Maximum number of nodes
  [[nodiscard]] constexpr size_t maxNumberNodes() const {
    if(nodes_.empty()) {
      return 0;
    }
    return std::max_element(
        nodes_.cbegin(), nodes_.cend(),
        [](auto const &lhs, auto const &rhs) { return lhs.size() < rhs.size(); }
    )->size();
  }

  /// @brief Return the maximum node name size (used to create TypeNodesMapArray)
  /// @return Maximum node name size
  [[nodiscard]] constexpr size_t maxSizeName() const {
    size_t ret = 0;
    for (auto const &type : nodesName_) {
      for (auto const &nodeName : type) {
        if (nodeName.size() > ret) { ret = nodeName.size(); }
      }
    }
    return ret;
  }
};
}
}

#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_CX_TYPES_NODES_MAP_H_
