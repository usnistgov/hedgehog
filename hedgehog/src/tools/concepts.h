//  NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
//  software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
//  derivative works of the software or any portion of the software, and you may copy and distribute such modifications
//  or works. Modified works should carry a notice stating that you changed the software and should note the date and
//  nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
//  source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
//  EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
//  WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
//  CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
//  THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
//  are solely responsible for determining the appropriateness of using and distributing the software and you assume
//  all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
//  with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
//  operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
//  damage to property. The software developed by NIST employees is not subject to copyright protection within the
//  United States.



#ifndef HEDGEHOG_CONCEPTS_H
#define HEDGEHOG_CONCEPTS_H
#pragma once

#include <type_traits>

#include "meta_functions.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog core namespace
namespace core {

/// @brief Hedgehog abstraction namespace
namespace abstraction {
/// @brief Forward declaration of SlotAbstraction
class SlotAbstraction;
/// @brief Forward declaration of NotifierAbstraction
class NotifierAbstraction;
/// @brief Forward declaration of NodeAbstraction
class NodeAbstraction;
/// @brief Forward declaration of ReceiverAbstraction
/// @tparam Input Input type
template<class Input>
class ReceiverAbstraction;
/// @brief Forward declaration of SenderAbstraction
/// @tparam Output Output type
template<class Output>
class SenderAbstraction;
}

/// @brief Hedgehog implementor namespace
namespace implementor {
/// @brief Forward declaration of ImplementorSlot
class ImplementorSlot;
/// @brief Forward declaration of ImplementorNotifier
class ImplementorNotifier;
/// @brief Forward declaration of ImplementorReceiver
/// @tparam Input Input type
template<class Input>
class ImplementorReceiver;
/// @brief Forward declaration of ImplementorExecute
/// @tparam Input Input type
template<class Input>
class ImplementorExecute;
/// @brief Forward declaration of ImplementorSender
/// @tparam Output Output type
template<class Output>
class ImplementorSender;
}
}

/// @brief Forward declaration of ManagedMemory
class ManagedMemory;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// Hedgehog tool namespace
namespace tool {

/// @brief Concept verifying that a type is in a variadic
/// @tparam T Type to test in variadic Ts
/// @tparam Ts Variadic of types
template<class T, class ...Ts>
concept ContainsConcept = isContainedIn_v<T, Ts...>; // The type T is not in Ts...

/// @brief Concept verifying that a type is in a tuple
/// @tparam T Type to test in tuple Tuple
/// @tparam Tuple Tuple of types
template<class T, class Tuple>
concept ContainsInTupleConcept = isContainedInTuple_v<T, Tuple>; // The type T is not in Ts...

/// @brief Test if an input type is in the list of input types (tuple)
/// @tparam I Input type to test
/// @tparam Is Tuple of input types
template<class I, class Is>
concept MatchInputTypeConcept =
ContainsInTupleConcept<I, Is>; // The input type (I) should be part of the list of input types (Is)

/// @brief Test if an output type is in the list of output types (variadic)
/// @tparam O Output type to test
/// @tparam Os Tuple of output types
template<class O, class ...Os>
concept MatchOutputTypeConcept =
ContainsConcept<O, Os...>; // The output type (O) should be part of the list of output types (Os)

/// @brief Test if a type inherit from core::abstraction::SenderAbstraction for all Os and core::abstraction::NotifierAbstraction
/// @tparam T Type to test
/// @tparam Os Output types list (variadic)
template<class T, class ...Os>
concept MultiSendersAndNotifierAbstractionConcept =
(std::is_base_of_v<core::abstraction::SenderAbstraction<Os>, T> && ...) &&
    std::is_base_of_v<core::abstraction::NotifierAbstraction, T>;

/// @brief Test if a type T is manageable in a Memory manager (derives from ManagedMemory and is default constructible)
/// @tparam T Type to test
template<class T>
concept ManageableMemory =
std::is_base_of_v<ManagedMemory, T> // Derives from ManagedMemory
    && std::is_default_constructible_v<T>; // default constructible

//////////////////////////////// CORE //////////////////////////////////////////
/// @brief Test if a core can be input of a graph (derives from core::abstraction::NodeAbstraction and shares at least
/// one input type with the graph)
/// @tparam CoreNode Type of core
/// @tparam TupleCoreInputs Tuple of core input types
/// @tparam TupleGraphInputs Tuple of graph input types
template<class CoreNode, class TupleCoreInputs, class TupleGraphInputs>
concept CompatibleInputCore =
std::is_base_of_v<core::abstraction::NodeAbstraction, CoreNode> // The core given derives from NodeAbstraction
    && std::tuple_size_v<Intersect_t<TupleCoreInputs, TupleGraphInputs>> != 0; // There is at least one common type


/// @brief Test if a core can be input of a graph  for a type (derives from core::abstraction::NodeAbstraction and type
/// InputType is in the tuples of the core and graph input types)
/// @tparam CoreNode Type of core
/// @tparam InputType Input type
/// @tparam TupleCoreInputs Tuple of core input types
/// @tparam TupleGraphInputs Tuple of graph input types
template<class CoreNode, class InputType, class TupleCoreInputs, class TupleGraphInputs>
concept CompatibleInputCoreForAType =
std::is_base_of_v<core::abstraction::NodeAbstraction, CoreNode> // The core given derives from NodeAbstraction
    && ContainsInTupleConcept<InputType, TupleGraphInputs> // The input type is part of the graph input types
    && ContainsInTupleConcept<InputType, TupleCoreInputs>; // The input type is part of the core input types

/// @brief Test if a core can be output of a graph (derives from core::abstraction::NodeAbstraction and shares at least
/// one output type with the graph)
/// @tparam CoreNode Type of core
/// @tparam TupleCoreOutputs Tuple of core output types
/// @tparam TupleGraphOutputs Tuple of graph output types
template<class CoreNode, class TupleCoreOutputs, class TupleGraphOutputs>
concept CompatibleOutputCore =
std::is_base_of_v<core::abstraction::NodeAbstraction, CoreNode> // The core given derives from NodeAbstraction
    && std::tuple_size_v<Intersect_t<TupleCoreOutputs,
                                     TupleGraphOutputs>> != 0;  // There is at least one common type


/// @brief Test if a core can be output of a graph  for a type (derives from core::abstraction::NodeAbstraction and type
/// OutputType is in the tuples of the core and graph output types)
/// @tparam CoreNode Type of core
/// @tparam OutputType Output type
/// @tparam TupleCoreOutputs Tuple of core output types
/// @tparam TupleGraphOutputs Tuple of graph output types
template<class CoreNode, class OutputType, class TupleCoreOutputs, class TupleGraphOutputs>
concept CompatibleOutputCoreForAType =
std::is_base_of_v<core::abstraction::NodeAbstraction, CoreNode> // The core given derives from NodeAbstraction
    && ContainsInTupleConcept<OutputType, TupleGraphOutputs> // The output type is part of the graph output types
    && ContainsInTupleConcept<OutputType, TupleCoreOutputs>; // The output type is part of the core output types

//////////////////////////////// NODE //////////////////////////////////////////
/// @brief Test if a node is a sender node
/// @tparam NodeType Type of node to test
template<class NodeType>
concept SenderNode =
std::is_base_of_v<behavior::Node, NodeType> // The node derives from Node
    && requires(NodeType *node){
      typename NodeType::outputs_t; // The node has its output types accessible
      std::is_base_of_v<behavior::MultiSenders<typename NodeType::outputs_t>, NodeType
      >; // The node derives from MultiSenders
    };

/// @brief Test if a node is a receiver node
/// @tparam NodeType Type of node to test
template<class NodeType>
concept ReceiverNode =
std::is_base_of_v<behavior::Node, NodeType> // The node derives from Node
    && requires(NodeType *node){
      typename NodeType::inputs_t; // The node has its input types accessible
      std::is_base_of_v<behavior::MultiReceivers<typename NodeType::inputs_t>, NodeType
      >; // The node derives from MultiReceivers
    };

/// @brief Test if a node is a sender for a type
/// @tparam NodeType Type of node to test
/// @tparam OutputType Type to test
template<class NodeType, class OutputType>
concept SenderNodeForAType =
SenderNode<NodeType> // The node is a sender node
    && ContainsInTupleConcept<OutputType,
                              typename NodeType::outputs_t>; // The OutputType is part of the sender output types

/// @brief Test if a node is a receiver for a type
/// @tparam NodeType Type of node to test
/// @tparam InputType Type to test
template<class NodeType, class InputType>
concept ReceiverNodeForAType =
ReceiverNode<NodeType> // The node is a receiver node
    && ContainsInTupleConcept<InputType,
                              typename NodeType::inputs_t>; // The InputType is part of the sender input types

/// @brief Test if a node can be input of a graph
/// @tparam NodeType Type of a node
/// @tparam TupleGraphInputs Graph input types (tuple)
template<class NodeType, class TupleGraphInputs>
concept CompatibleInputNode =
ReceiverNode<NodeType> // The node is a receiver node
    && std::tuple_size_v<
        Intersect_t<typename NodeType::inputs_t, TupleGraphInputs>
    > != 0; // There is at least one common type between the task and graph input types

/// @brief Test if a node can be input of a graph for a type
/// @tparam NodeType Type of node
/// @tparam InputType Input type to test against
/// @tparam TupleGraphInputs Tuple of graph's input types
template<class NodeType, class InputType, class TupleGraphInputs>
concept CompatibleInputNodeForAType =
ReceiverNodeForAType<NodeType, InputType> // The node is a receiver node
    && ContainsInTupleConcept<InputType, TupleGraphInputs>; // The input type is part of the graph input types

/// @brief Test if a node can be output of a graph
/// @tparam NodeType Type of a node
/// @tparam TupleGraphOutputs Graph output types (tuple)
template<class NodeType, class TupleGraphOutputs>
concept CompatibleOutputNode =
SenderNode<NodeType> // The node is a sender node
    && std::tuple_size_v<
        Intersect_t<typename NodeType::outputs_t, TupleGraphOutputs>
    > != 0; // There is at least one common type between the task and graph output types

/// @brief Test if a node can be output of a graph for a type
/// @tparam NodeType Type of node
/// @tparam OutputType Input type to test against
/// @tparam TupleGraphOutputs Tuple of graph's input types (tuple)
template<class NodeType, class OutputType, class TupleGraphOutputs>
concept CompatibleOutputNodeForAType =
SenderNodeForAType<NodeType, OutputType> // The node is a sender node
    && ContainsInTupleConcept<OutputType, TupleGraphOutputs>; // The output type is part of the graph output types

/// @brief Test if a node is copyable (copy method is callable and has a the right return type)
/// @tparam NodeType Type of the node
template<class NodeType>
concept CopyableNode = requires(NodeType *n){
  { n->copy() }; // Check if the copy method is available
  std::is_same_v<decltype(n->copy()), std::shared_ptr<NodeType>>; // Check if copy() returns the right type
};

/// @brief Test if a type is a valid concrete implementation of the core::implementor::ImplementorReceiver for the
/// InputTypes
/// @tparam MultiReceiver Type to test
/// @tparam InputTypes Input types list (variadic)
template<class MultiReceiver, class ...InputTypes>
concept ConcreteMultiReceiverImplementation =
(std::is_base_of_v<core::implementor::ImplementorReceiver<InputTypes>, MultiReceiver>, ...);

/// @brief Test if a type is a valid concrete implementation of the core::implementor::ImplementorSender for the
/// OutputTypes
/// @tparam MultiSender Type to test
/// @tparam OutputTypes Output types list (variadic)
template<class MultiSender, class ...OutputTypes>
concept ConcreteMultiSenderImplementation =
(std::is_base_of_v<core::implementor::ImplementorSender<OutputTypes>, MultiSender>, ...);

/// @brief Test if a type is a valid concrete implementation of the core::implementor::ImplementorExecute for the
/// InputTypes
/// @tparam MultiExecute Type to test
/// @tparam InputTypes Input types list (variadic)
template<class MultiExecute, class ...InputTypes>
concept ConcreteMultiExecuteImplementation =
(std::is_base_of_v<core::implementor::ImplementorExecute<InputTypes>, MultiExecute>, ...);

}
}
#endif //HEDGEHOG_CONCEPTS_H
