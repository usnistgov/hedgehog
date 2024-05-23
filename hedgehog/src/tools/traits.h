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

#ifndef HEDGEHOG_TRAITS_H
#define HEDGEHOG_TRAITS_H

#pragma once

#include <tuple>

#include "../core/implementors/concrete_implementor/default_multi_senders.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog abstraction namespace
namespace abstraction {
/// @brief Forward declaration of TaskInputsManagementAbstraction
/// @tparam Inputs Input types
template<class ...Inputs>
class TaskInputsManagementAbstraction;
/// @brief Forward declaration of GraphInputsManagementAbstraction
/// @tparam Inputs Input types
template<class ...Inputs>
class GraphInputsManagementAbstraction;
/// @brief Forward declaration of GraphOutputsManagementAbstraction
/// @tparam Inputs Input types
template<class ...Outputs>
class GraphOutputsManagementAbstraction;
/// @brief Forward declaration of TaskOutputsManagementAbstraction
/// @tparam Inputs Input types
template<class ...Outputs>
class TaskOutputsManagementAbstraction;
/// @brief Forward declaration of ExecutionPipelineInputsManagementAbstraction
/// @tparam Inputs Input types
template<class ...Inputs>
class ExecutionPipelineInputsManagementAbstraction;
/// @brief Forward declaration of ExecutionPipelineOutputsManagementAbstraction
/// @tparam Inputs Input types
template<class ...Outputs>
class ExecutionPipelineOutputsManagementAbstraction;
}

/// @brief Hedgehog implementor namespace
namespace implementor {
/// @brief Forward declaration of DefaultMultiExecutes
/// @tparam Inputs Input types
template<class ...Inputs>
class DefaultMultiExecutes;

/// @brief Forward declaration of MultiQueueReceivers
/// @tparam Inputs Input types
template<class ...Inputs>
class MultiQueueReceivers;
}
}

/// @brief Hedgehog behavior namespace
namespace behavior {
/// @brief Forward declaration of MultiExecute
/// @tparam Inputs Input types
template<class ...Inputs>
class MultiExecute;
/// @brief Forward declaration of MultiReceivers
/// @tparam Inputs Input types
template<class ...Inputs>
class MultiReceivers;
/// @brief Forward declaration of MultiSwitchRules
/// @tparam Inputs Input types
template<class ...Inputs>
class MultiSwitchRules;
/// @brief Forward declaration of MultiSenders
/// @tparam Outputs Output types
template<class ...Outputs>
class MultiSenders;
/// @brief Forward declaration of TaskMultiSenders
/// @tparam Outputs Output types
template<class ...Outputs>
class TaskMultiSenders;
/// @brief Forward declaration of StateMultiSenders
/// @tparam Outputs Output types
template<class ...Outputs>
class StateMultiSenders;
}

#endif //DOXYGEN_SHOULD_SKIP_THIS

/// Hedgehog tool namespace
namespace tool {

////////////////////////////// Abstractions
/// @brief Base definition of the type deducer for TaskInputsManagementAbstraction
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct TaskInputsManagementAbstractionTypeDeducer;

/// @brief Definition of the type deducer for TaskInputsManagementAbstraction
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct TaskInputsManagementAbstractionTypeDeducer<std::tuple<Inputs...>> {
  using type = core::abstraction::TaskInputsManagementAbstraction<Inputs...>;
};

/// @brief Helper to the deducer for TaskInputsManagementAbstraction
template<class TupleInputs>
using TaskInputsManagementAbstractionTypeDeducer_t =
    typename TaskInputsManagementAbstractionTypeDeducer<TupleInputs>::type;

/// @brief Base definition of the type deducer for TaskOutputsManagementAbstraction
/// @tparam Outputs Output types as tuple
template<class Outputs>
struct TaskOutputsManagementAbstractionTypeDeducer;

/// @brief Definition of the type deducer for TaskOutputsManagementAbstraction
/// @tparam Outputs Variadic of types
template<class ...Outputs>
struct TaskOutputsManagementAbstractionTypeDeducer<std::tuple<Outputs...>> {
  using type = core::abstraction::TaskOutputsManagementAbstraction<Outputs...>;
};

/// @brief Helper to the deducer for TaskOutputsManagementAbstraction
template<class TupleOutputs>
using TaskOutputsManagementAbstractionTypeDeducer_t =
    typename TaskOutputsManagementAbstractionTypeDeducer<TupleOutputs>::type;

/// @brief Base definition of the type deducer for GraphInputsManagementAbstraction
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct GraphInputsManagementAbstractionTypeDeducer;

/// @brief Definition of the type deducer for GraphInputsManagementAbstraction
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct GraphInputsManagementAbstractionTypeDeducer<std::tuple<Inputs...>> {
  using type = core::abstraction::GraphInputsManagementAbstraction<Inputs...>;
};

/// @brief Helper to the deducer for GraphInputsManagementAbstraction
template<class TupleInputs>
using GraphInputsManagementAbstractionTypeDeducer_t =
    typename GraphInputsManagementAbstractionTypeDeducer<TupleInputs>::type;

/// @brief Base definition of the type deducer for GraphOutputsManagementAbstraction
/// @tparam Outputs Output types as tuple
template<class Outputs>
struct GraphOutputsManagementAbstractionTypeDeducer;

/// @brief Definition of the type deducer for GraphOutputsManagementAbstraction
/// @tparam Outputs Variadic of types
template<class ...Outputs>
struct GraphOutputsManagementAbstractionTypeDeducer<std::tuple<Outputs...>> {
  using type = core::abstraction::GraphOutputsManagementAbstraction<Outputs...>;
};

/// @brief Helper to the deducer for GraphOutputsManagementAbstraction
template<class TupleOutputs>
using GraphOutputsManagementAbstractionTypeDeducer_t =
    typename GraphOutputsManagementAbstractionTypeDeducer<TupleOutputs>::type;

/// @brief Base definition of the type deducer for ExecutionPipelineInputsManagementAbstraction
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct ExecutionPipelineInputsManagementAbstractionTypeDeducer;

/// @brief Definition of the type deducer for ExecutionPipelineInputsManagementAbstraction
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct ExecutionPipelineInputsManagementAbstractionTypeDeducer<std::tuple<Inputs...>> {
  using type = core::abstraction::ExecutionPipelineInputsManagementAbstraction<Inputs...>;
};

/// @brief Helper to the deducer for ExecutionPipelineInputsManagementAbstraction
template<class TupleInputs>
using ExecutionPipelineInputsManagementAbstractionTypeDeducer_t =
    typename ExecutionPipelineInputsManagementAbstractionTypeDeducer<TupleInputs>::type;

/// @brief Base definition of the type deducer for ExecutionPipelineOutputsManagementAbstraction
/// @tparam Outputs Output types as tuple
template<class Outputs>
struct ExecutionPipelineOutputsManagementAbstractionTypeDeducer;

/// @brief Definition of the type deducer for ExecutionPipelineOutputsManagementAbstraction
/// @tparam Outputs Variadic of types
template<class ...Outputs>
struct ExecutionPipelineOutputsManagementAbstractionTypeDeducer<std::tuple<Outputs...>> {
  using type = core::abstraction::ExecutionPipelineOutputsManagementAbstraction<Outputs...>;
};

/// @brief Helper to the deducer for ExecutionPipelineOutputsManagementAbstraction
template<class TupleOutputs>
using ExecutionPipelineOutputsManagementAbstractionTypeDeducer_t =
    typename ExecutionPipelineOutputsManagementAbstractionTypeDeducer<TupleOutputs>::type;

////////////////////////////// Implementors

/// @brief Base definition of the type deducer for DefaultMultiExecutes
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct DefaultMultiExecutesTypeDeducer;

/// @brief Definition of the type deducer for DefaultMultiExecutes
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct DefaultMultiExecutesTypeDeducer<std::tuple<Inputs...>> {
  using type = core::implementor::DefaultMultiExecutes<Inputs...>;
};

/// @brief Helper to the deducer for DefaultMultiExecutes
/// @TupleInputs Tuple of input types
template<class TupleInputs>
using DefaultMultiExecutesTypeDeducer_t = typename DefaultMultiExecutesTypeDeducer<TupleInputs>::type;

/// @brief Helper to the deducer for DefaultMultiExecutes
/// @tparam Separator Separator of node template arg
/// @tparam AllTypes All types of node template arg
template<size_t Separator, class ...AllTypes>
using DME = DefaultMultiExecutesTypeDeducer_t<hh::tool::Inputs<Separator, AllTypes...>>;

/// @brief Base definition of the type deducer for DefaultMultiSenders
/// @tparam Outputs Variadic of output types
template<class Outputs>
struct MultiDefaultSendersTypeDeducer;

/// @brief Definition of the type deduced for DefaultMultiSenders
/// @tparam Outputs Variadic of output types
template<class ...Outputs>
struct MultiDefaultSendersTypeDeducer<std::tuple<Outputs...>> {
  using type = hh::core::implementor::DefaultMultiSenders<Outputs...>;
};

/// @brief Helper to the deducer for MultiDefaultSendersTypeDeducer
/// @tparam TupleOutputs Tuple of output types
template<class TupleOutputs>
using MultiDefaultSendersTypeDeducer_t = typename MultiDefaultSendersTypeDeducer<TupleOutputs>::type;

/// @brief Helper to the deducer for MultiDefaultSendersTypeDeducer
/// @tparam Separator Separator of node template arg
/// @tparam AllTypes All types of node template arg
template<size_t Separator, class ...AllTypes>
using MDS = MultiDefaultSendersTypeDeducer_t<hh::tool::Outputs<Separator, AllTypes...>>;

/// @brief Base definition of the type deducer for MultiQueueReceivers
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct MultiQueueReceiversTypeDeducer;

/// @brief Definition of the type deducer for MultiQueueReceivers
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct MultiQueueReceiversTypeDeducer<std::tuple<Inputs...>> {
  using type = core::implementor::MultiQueueReceivers<Inputs...>;
};

/// @brief Helper to the deducer for MultiQueueReceivers
/// @tparam TupleInputs Tuple of input types
template<class TupleInputs>
using MultiQueueReceiversTypeDeducer_t = typename MultiQueueReceiversTypeDeducer<TupleInputs>::type;

/// @brief Helper to the deducer for MultiQueueReceivers from the nodes template parameters
/// @tparam Separator Separator of node template arg
/// @tparam AllTypes All types of node template arg
template<size_t Separator, class ...AllTypes>
using MQR = MultiQueueReceiversTypeDeducer_t<hh::tool::Inputs<Separator, AllTypes...>>;

////////////////////////////// Behaviors
/// @brief Base definition of the type deducer for MultiExecute
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct BehaviorMultiExecuteTypeDeducer;

/// @brief Definition of the type deducer for MultiExecute
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct BehaviorMultiExecuteTypeDeducer<std::tuple<Inputs...>> { using type = behavior::MultiExecute<Inputs...>; };

/// @brief Helper to the deducer for MultiExecute
template<class TupleInputs>
using BehaviorMultiExecuteTypeDeducer_t = typename BehaviorMultiExecuteTypeDeducer<TupleInputs>::type;

/// @brief Base definition of the type deducer for MultiReceivers
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct BehaviorMultiReceiversTypeDeducer;

/// @brief Definition of the type deducer for MultiReceivers
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct BehaviorMultiReceiversTypeDeducer<std::tuple<Inputs...>> { using type = behavior::MultiReceivers<Inputs...>; };

/// @brief Helper to the deducer for MultiReceivers
template<class TupleInputs>
using BehaviorMultiReceiversTypeDeducer_t = typename BehaviorMultiReceiversTypeDeducer<TupleInputs>::type;

/// @brief Base definition of the type deducer for MultiSwitchRules
/// @tparam Inputs Input types as tuple
template<class Inputs>
struct BehaviorMultiSwitchRulesTypeDeducer;

/// @brief Definition of the type deducer for MultiSwitchRules
/// @tparam Inputs Variadic of types
template<class ...Inputs>
struct BehaviorMultiSwitchRulesTypeDeducer<std::tuple<Inputs...>> { using type = behavior::MultiSwitchRules<Inputs...>; };

/// @brief Helper to the deducer for MultiSwitchRules
template<class TupleInputs>
using BehaviorMultiSwitchRulesTypeDeducer_t = typename BehaviorMultiSwitchRulesTypeDeducer<TupleInputs>::type;

/// @brief Base definition of the type deducer for MultiSenders
/// @tparam Outputs Output types as tuple
template<class Outputs>
struct BehaviorMultiSenderTypeDeducer;

/// @brief Definition of the type deducer for MultiSenders
/// @tparam Outputs Variadic of types
template<class ...Outputs>
struct BehaviorMultiSenderTypeDeducer<std::tuple<Outputs...>> { using type = behavior::MultiSenders<Outputs...>; };

/// @brief Helper to the deducer for MultiSenders
template<class TupleOutputs>
using BehaviorMultiSendersTypeDeducer_t = typename BehaviorMultiSenderTypeDeducer<TupleOutputs>::type;

/// @brief Base definition of the type deducer for TaskMultiSenders
/// @tparam Outputs Output types as tuple
template<class Outputs>
struct BehaviorTaskMultiSenderTypeDeducer;

/// @brief Definition of the type deducer for TaskMultiSenders
/// @tparam Outputs Variadic of types
template<class ...Outputs>
struct BehaviorTaskMultiSenderTypeDeducer<std::tuple<Outputs...>> { using type = behavior::TaskMultiSenders<Outputs...>; };

/// @brief Helper to the deducer for TaskMultiSenders
template<class TupleOutputs>
using BehaviorTaskMultiSendersTypeDeducer_t = typename BehaviorTaskMultiSenderTypeDeducer<TupleOutputs>::type;

/// @brief Base definition of the type deducer for StateMultiSenders
/// @tparam Outputs Output types as tuple
template<class Outputs>
struct BehaviorStateMultiSenderTypeDeducer;

/// @brief Definition of the type deducer for StateMultiSenders
/// @tparam Outputs Variadic of types
template<class ...Outputs>
struct BehaviorStateMultiSenderTypeDeducer<std::tuple<Outputs...>> { using type = behavior::StateMultiSenders<Outputs...>; };

/// @brief Helper to the deducer for StateMultiSenders
template<class TupleOutputs>
using BehaviorStateMultiSendersTypeDeducer_t = typename BehaviorStateMultiSenderTypeDeducer<TupleOutputs>::type;
}
}

#endif //HEDGEHOG_TRAITS_H
