
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

#ifndef HEDGEHOG_CX_CONCEPTS_H_
#define HEDGEHOG_CX_CONCEPTS_H_

#ifdef HH_ENABLE_HH_CX

#include "meta_functions.h"

/// @brief Hedgehog compile-time namespace
namespace hh_cx {
/// @brief Hedgehog behavior namespace
namespace behavior {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of AbstractNode
class AbstractNode;
#endif //DOXYGEN_SHOULD_SKIP_THIS

}
/// Hedgehog tool namespace
namespace tool {

/// @brief Concept defining a Hedgehog dynamic connectable node, especially it can receives and sends data
/// @tparam HedgehogNode Type of dynamic node
template<class HedgehogNode>
concept HedgehogConnectableNode = hh::tool::SenderNode<HedgehogNode> && hh::tool::ReceiverNode<HedgehogNode>;

/// @brief Concept defining an Hedgehog static node, it inherits from hh_cx::behavior::AbstractNode and has the right
/// type accessor
/// @tparam StaticHedgehogNode Type of static node
template<typename StaticHedgehogNode>
concept HedgehogStaticNode = requires{
  std::is_base_of_v<hh_cx::behavior::AbstractNode, StaticHedgehogNode>;
  typename StaticHedgehogNode::ro_type_t; ///< The static type should provide an accessor to its read-only types
  typename StaticHedgehogNode::dynamic_node_t; ///< The static type should provide an accessor to its dynamic type
  typename StaticHedgehogNode::inputs_t; ///< The static type should provide an accessor to its input type
  typename StaticHedgehogNode::outputs_t; ///< The static type should provide an accessor to its output type
};

/// @brief Concept to define a dynamic graph for the static analysis
/// @details The dynamic Graph's type should have a constructor with only a name
/// (DynamicHedgehogNode(std::string_view const &))
/// @tparam DynamicHedgehogNode Type to test
template<typename DynamicHedgehogNode>
concept HedgehogDynamicGraphForStaticAnalysis =
HedgehogConnectableNode<DynamicHedgehogNode>
    && std::is_base_of_v<
        hh_cx::tool::Graph_t<
            std::tuple_size_v<typename DynamicHedgehogNode::inputs_t>,
            hh_cx::tool::CatTuples_t<
                typename DynamicHedgehogNode::inputs_t,
                typename DynamicHedgehogNode::outputs_t>
            >,
        DynamicHedgehogNode>
    && std::is_constructible_v<DynamicHedgehogNode, std::string const &>;
}

}

#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_CX_CONCEPTS_H_
