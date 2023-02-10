
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

#ifndef HEDGEHOG_TYPE_NODES_MAP_ARRAY_H_
#define HEDGEHOG_TYPE_NODES_MAP_ARRAY_H_
#ifdef HH_ENABLE_HH_CX


#include <array>
#include <cstdlib>
#include <ostream>
#include <map>
#include <vector>

/// @brief Hedgehog compile-time namespace
namespace hh_cx {
/// Hedgehog tool namespace
namespace tool {

/// @brief Flattened TypesNodesMap used to transfer a TypeNodesMap from compile-time to runtime
/// @tparam NbTypes Number of types (keys in the map)
/// @tparam MaxSizeTypeName Max types names size
/// @tparam MaxNumberNodes Max number of nodes
/// @tparam MaxSizeNodeName Max node names size
template<size_t NbTypes, size_t MaxSizeTypeName, size_t MaxNumberNodes, size_t MaxSizeNodeName>
class TypeNodesMapArray {
  std::array<std::array<char, MaxSizeTypeName>, NbTypes> types_{}; ///< Types stored (keys)
  std::array<std::array<std::array<char, MaxSizeNodeName>, MaxNumberNodes>, NbTypes> nodes_{}; ///< Nodes name stored
 public:
  /// @brief Default constructor
  constexpr TypeNodesMapArray() = default;
  /// @brief Default destructor
  constexpr virtual ~TypeNodesMapArray() = default;

  /// @brief Accessor to the types (key)
  /// @return Types (key)
  constexpr std::array<std::array<char, MaxSizeTypeName>, NbTypes> &types() { return types_; }

  /// @brief Accessor to the nodes names (values)
  /// @return  Nodes names (values)
  constexpr std::array<std::array<std::array<char, MaxSizeNodeName>, MaxNumberNodes>, NbTypes> &nodes() {
    return nodes_;
  }

  /// @brief Reverse the map (type -> nodes to node -> types) used to simplify the dynamic graph generation
  /// @return A map from node -> types
  [[nodiscard]] std::map<std::string, std::vector<std::string>> mapNodeNameToTypeNames() const{
    std::map<std::string, std::vector<std::string>> map{};

    for (size_t typeId = 0; typeId < types_.size(); ++typeId) {
      auto typeName = std::string(types_.at(typeId).data());
      for (auto const &nameArray : nodes_.at(typeId)) {
        auto const &nodeName = std::string(nameArray.data());
        if (!nodeName.empty()) {
          if (map.count(nodeName)) { map.at(nodeName).push_back(typeName); }
          else { map.insert({nodeName, {typeName}}); }
        }
      }
    }
    return map;
  }

};
}
}
#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_TYPE_NODES_MAP_ARRAY_H_
