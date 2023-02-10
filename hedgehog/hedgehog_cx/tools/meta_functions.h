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

#ifndef HEDGEHOG_CX_METAFUNCTIONS_H_
#define HEDGEHOG_CX_METAFUNCTIONS_H_

#ifdef HH_ENABLE_HH_CX

#include <tuple>

#include "../../src/api/task/abstract_task.h"
#include "../../src/api/state_manager/state_manager.h"
#include "../../src/api/graph/graph.h"


/// @brief Hedgehog compile-time namespace
namespace hh_cx {

/// Hedgehog tool namespace
namespace tool {

/// Hedgehog internals namespace
namespace internals {

/// @brief Base definition of HelperCoreMultiReceiversType
/// @tparam Inputs Tuple of input types
template<size_t Separator, class AllTypes>
struct HelperAbstractTaskType;

/// @brief Used helper to get the type of hedgehog CoreMultiReceivers for inputs from hedgehog tuple of Input types
/// @tparam Inputs CoreMultiReceivers inputs
template<size_t Separator, class ...AllTypes>
struct HelperAbstractTaskType<Separator, std::tuple<AllTypes...>> {
  using type = hh::AbstractTask<Separator, AllTypes...>; ///< Type of the CoreMultiReceivers
};

/// @brief Base definition of HelperCoreMultiReceiversType
/// @tparam Inputs Tuple of input types
template<size_t Separator, class AllTypes>
struct HelperStateManagerType;

/// @brief Used helper to get the type of hedgehog CoreMultiReceivers for inputs from hedgehog tuple of Input types
/// @tparam Inputs CoreMultiReceivers inputs
template<size_t Separator, class ...AllTypes>
struct HelperStateManagerType<Separator, std::tuple<AllTypes...>> {
  using type = hh::StateManager<Separator, AllTypes...>; ///< Type of the CoreMultiReceivers
};

/// @brief Base definition of GraphTypeDeducer
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class AllTypes>
struct GraphTypeDeducer;

/// @brief Create a hh::Graph from the separator and the list of types
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
struct GraphTypeDeducer<Separator, std::tuple<AllTypes...>> {
  /// @brief Type accessor to the Graph
  using type = hh::Graph<Separator, AllTypes...>;
};

/// @brief Base definition of CatTuples
template<class, class>
struct CatTuples;

/// @brief Create a tuple of types that results of the concatenation of two variadic types
/// @tparam First First variadic to concatenate
/// @tparam Second Second variadic to concatenate
template<class... First, class... Second>
struct CatTuples<std::tuple<First...>, std::tuple<Second...>> {
  /// @brief Type accessor to the concatenated tuples
  using type = std::tuple<First..., Second...>;
};

/// @brief Base definition of getUniqueVariantFromTuple
/// @tparam Tuple Tuple of types
template<class Tuple>
class UniqueVariantFromTuple;

/// @brief Transform a tuple to a variant of unique types
/// @tparam AllTypes All the types contained in the tuple
template<class ...AllTypes>
class UniqueVariantFromTuple<std::tuple<AllTypes...>> {
  /// @brief Class used to create a tuple with unique types
  class MakeUnique {
   private:
    /// @brief Create a tuple with unique type
    /// @tparam Current Current type analysed
    /// @tparam Others Other type remaining in the tuple
    /// @return A tuple with unique types as pointers
    template<class Current, class ... Others>
    static auto makeUniqueTuple() {
      if constexpr (sizeof...(Others) == 0) {
        return std::tuple<Current *>{};
      } else {
        if constexpr (std::disjunction_v<std::is_same<Current, Others>...>) {
          return makeUniqueTuple<Others...>();
        } else {
          return std::tuple_cat(std::tuple<Current *>{}, makeUniqueTuple<Others...>());
        }
      }
    }

    /// @brief Create a variant from a tuple of pointer of types
    /// @tparam TupleType Type of the tuple
    /// @return A variant containing all types
    template<class ... TupleType>
    static auto makeVariant(std::tuple<TupleType...>) {
      return (std::variant<std::remove_pointer_t<TupleType>...>());
    }

   public:
    /// @brief Type of tuple with uniques pointer types
    using type = decltype(makeUniqueTuple<AllTypes...>());
    /// @brief Type of variant with uniques pointer
    using variant_type = decltype(makeVariant(std::declval<type>()));
  };

 public:
  /// @brief Direct accessor to the variant type with unique types
  using type = typename MakeUnique::variant_type;
};
} // internals

/// @brief Helper to the type hold by UniqueVariantFromTuple
template<class Tuple>
using UniqueVariantFromTuple_t = typename internals::UniqueVariantFromTuple<Tuple>::type;

/// @brief Helper to create a StateManager type from the Separator and all the types
template<size_t Separator, class AllTypes>
using StateManager_t = typename tool::internals::HelperStateManagerType<Separator, AllTypes>::type;

/// @brief Helper to create an AbstractTask type from the Separator and all the types
template<size_t Separator, class AllTypes>
using AbstractTask_t = typename tool::internals::HelperAbstractTaskType<Separator, AllTypes>::type;

/// @brief Helper to create a Graph type from the Separator and all the types
template<size_t Separator, class AllTypes>
using Graph_t = typename tool::internals::GraphTypeDeducer<Separator, AllTypes>::type;

/// @brief Helper to create a tuple which is the concatenation of two tuples
template<class First, class Second>
using CatTuples_t = typename tool::internals::CatTuples<First, Second>::type;
} // tool

} // hh_cx
#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_CX_METAFUNCTIONS_H_
