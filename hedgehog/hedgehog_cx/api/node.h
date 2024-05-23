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

#ifndef HEDGEHOG_CX_NODE_H_
#define HEDGEHOG_CX_NODE_H_

#ifdef HH_ENABLE_HH_CX

#include <string>

#include "../tools/concepts.h"
#include "../tools/meta_functions.h"

#include "../behavior/abstract_node.h"

/// @brief Hedgehog compile-time namespace
namespace hh_cx {

/// @brief Base node representation for the static analysis
/// @details A static node represents a dynamic node of type: HedgehogDynamicNodeType. The following types
/// (InputTypesRO...) are the input types declared as read only for the static analysis.
/// @attention Each of the static nodes need to have a unique string-id provided at compilation, that will be used to do the
/// mapping to real instance of dynamic nodes and create the runtime graph
/// @tparam HedgehogDynamicNodeType Type of the dynamic node represented by this static node
/// @tparam InputTypesRO List of input types treated as read only for static analysis
template<tool::HedgehogConnectableNode HedgehogDynamicNodeType, class ...InputTypesRO> requires (hh::tool::ContainsInTupleConcept<
    InputTypesRO,
    typename HedgehogDynamicNodeType::inputs_t> &&...)
class Node : public hh_cx::behavior::AbstractNode {
 private:
  std::vector<std::string>
      inputTypesAsName_{}, ///< Input's type names
  outputTypesAsName_{}, ///< Output's type names
  roTypesAsName_{}, ///< Read only input's type names
  constInputTypesAsName_{}; ///< Const output's type names

 public:
  using ro_type_t = std::tuple<InputTypesRO...>; ///< Types designed as read only
  using dynamic_node_t = HedgehogDynamicNodeType; ///< Dynamic node type represented byt he CXNode
  using inputs_t = typename HedgehogDynamicNodeType::inputs_t; ///< Node's inputs type
  using outputs_t = typename HedgehogDynamicNodeType::outputs_t; ///< Node's output type

  /// @brief Node constructor
  /// @param name Unique name of the Node
  constexpr explicit Node(std::string const &name) : AbstractNode(name) {
    using OutputIndices = std::make_index_sequence<std::tuple_size_v<outputs_t>>;
    using ROIndices = std::make_index_sequence<std::tuple_size_v<ro_type_t>>;
    using InputIndices = std::make_index_sequence<std::tuple_size_v<ro_type_t>>;
    registerInputType(InputIndices{});
    registerROInputType(ROIndices{});
    registerOutputType(OutputIndices{});
  };

  /// @brief Node destructor
  constexpr ~Node() override = default;

  /// @brief Test if the canTerminate method has been overloaded in the dynamic node type
  /// @return True, if canTerminate method has been overloaded in the dynamic node type, else False
  [[nodiscard]] constexpr bool isCanTerminateOverloaded() const final {
    if constexpr (std::is_base_of_v<hh::behavior::CanTerminate, HedgehogDynamicNodeType>) {
      using task_t =
          hh_cx::tool::AbstractTask_t<
              std::tuple_size_v<inputs_t>,
              hh_cx::tool::CatTuples_t<inputs_t, outputs_t>
          >;

      using stateManager_t =
          hh_cx::tool::StateManager_t<
              std::tuple_size_v<inputs_t>,
              hh_cx::tool::CatTuples_t<inputs_t, outputs_t>
          >;

      if constexpr (std::is_base_of_v<task_t, HedgehogDynamicNodeType>) {
        return !std::is_same_v<decltype(&task_t::canTerminate), decltype(&dynamic_node_t::canTerminate)>;
      } else if constexpr (std::is_base_of_v<stateManager_t, HedgehogDynamicNodeType>) {
        return !std::is_same_v<decltype(&stateManager_t::canTerminate), decltype(&dynamic_node_t::canTerminate)>;
      } else {
        throw std::runtime_error("The overload detection can not detect the tested type");
      }
    }
    return false;
  }

  /// @brief Test if a type (as a string) is part of the registered read-only types
  /// @param typeName Type name to test
  /// @return True if the type name is part of the read-only types
  [[nodiscard]] constexpr bool isTypeAnROType(std::string const &typeName) const final {
    return std::any_of(
        roTypesAsName_.cbegin(), roTypesAsName_.cend(),
        [&typeName](auto const &type) { return type == typeName; });
  }

  /// @brief Test if a type (as a string) is part of the registered input const type
  /// @param typeName Type name to test
  /// @return True if the type name is a registered input const type
  [[nodiscard]] constexpr bool isTypeAConstType(std::string const &typeName) const final {
    return std::any_of(
        constInputTypesAsName_.cbegin(), constInputTypesAsName_.cend(),
        [&typeName](auto const &type) { return type == typeName; });
  }

 private:
  /// @brief Decomposition strategy for the tuple of input types and store the types into vectors of type names (one for the input name and one for the const input name)
  /// @tparam Indices Indices of the tuple
  template<size_t ...Indices>
  constexpr void registerInputType(std::index_sequence<Indices...>) {
    (inputTypesAsName_.push_back(hh::tool::typeToStr<std::tuple_element_t<Indices, inputs_t>>()), ...);
    (testAndRegisterConstInputType<std::tuple_element_t<Indices, inputs_t>>(), ...);
  }

  /// @brief Test if the input type is const, register it if it is the case
  /// @tparam T Input type
  template<class T>
  constexpr void testAndRegisterConstInputType() {
    if (std::is_const_v<T>) { constInputTypesAsName_.push_back(hh::tool::typeToStr<T>()); }
  }

  /// @brief Decomposition strategy for the tuple of output types and store the types into a vector of type names
  /// @tparam Indices Indices of the tuple
  template<size_t ...Indices>
  constexpr void registerOutputType(std::index_sequence<Indices...>) {
    (outputTypesAsName_.push_back(hh::tool::typeToStr<std::tuple_element_t<Indices, outputs_t>>()), ...);
  }

  /// @brief Decomposition strategy for the tuple of read-only output types and store the types into a vector of type names
  /// @tparam Indices Indices of the tuple
  template<size_t ...Indices>
  constexpr void registerROInputType(std::index_sequence<Indices...>) {
    (roTypesAsName_.push_back(hh::tool::typeToStr<std::tuple_element_t<Indices, ro_type_t>>()), ...);
  }
};

}

#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_CX_NODE_H_
