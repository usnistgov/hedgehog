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

#ifndef HEDGEHOG_META_FUNCTIONS_H
#define HEDGEHOG_META_FUNCTIONS_H

#pragma once
#include <iostream>
#include <tuple>


/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Forward declaration of ReceiverAbstraction
/// @tparam Input Input type
template<class Input>
class ReceiverAbstraction;

/// @brief Forward declaration of SenderAbstraction
/// @tparam Output output type
template<class Output>
class SenderAbstraction;
}
}

#endif //DOXYGEN_SHOULD_SKIP_THIS

/// Hedgehog tool namespace
namespace tool {

/// Hedgehog tool internals namespace
namespace internals {

/// @brief Base definition of PushFront accepting a tuple as template parameter and the element to add
/// @tparam Ts List of types as tuple
/// @tparam T Type to add in front of tuple
template<typename Ts, typename T>
struct PushFront;

/// @brief Definition of PushFront accepting a variadic as template parameter and the element to add
/// @tparam Vs List of types as variadic
/// @tparam T Type to add in front of variadic
template<typename... Vs, typename T>
struct PushFront<std::tuple<Vs...>, T> {
  /// @brief Accessor of the tuple type consisting of the type T followed by the Vs...
  using Type = std::tuple<T, Vs...>;
};

/// @brief Helper creating a tuple from a tuple Ts in wish we have added on front the type T
template<typename Ts, typename T>
using PushFront_t = typename PushFront<Ts, T>::Type;

/// @brief Base definition of SplitInput splitting a tuple of type to get the input types
/// @tparam Types Tuple of types
/// @tparam Indices Index of types to keep (tuple)
template<typename Types, typename Indices>
struct SplitInput;

/// @brief Definition of SplitInput splitting a tuple of type to get the input types
/// @tparam Types Tuple of types
/// @tparam Indices Indices of the types to keep (variadic)
template<typename Types, size_t... Indices>
struct SplitInput<Types, std::index_sequence<Indices...>> {
  /// @brief Type accessor
  using Type = std::tuple<std::tuple_element_t<Indices, Types>...>;
};

/// @brief Helper getting the input types in a tuple
template<typename Types, typename Indices>
using SplitInput_t = typename SplitInput<Types, Indices>::Type;

/// @brief Base definition of SplitOutput_t splitting a tuple of type to get the output types
/// @tparam Types Tuple of types
/// @tparam delta Starting position of the output types
/// @tparam Indices Index position of output types beginning at 0 (tuple)
template<typename Types, size_t delta, typename Indices>
struct SplitOutput;

/// @brief Base definition of SplitOutput_t splitting a tuple of type to get the output types
/// @tparam Types Tuple of types
/// @tparam delta Starting position of the output types
/// @tparam Indices Index position of output types beginning at 0 (variadic)
template<typename Types, size_t delta, size_t... Indices>
struct SplitOutput<Types, delta, std::index_sequence<Indices...>> {
  /// @brief Type accessor
  using Type = std::tuple<std::tuple_element_t<delta + Indices, Types>...>;
};

/// @brief Helper to output types of a tuple
template<typename Types, size_t delta, typename Indices>
using SplitOutput_t = typename SplitOutput<Types, delta, Indices>::Type;

/// @brief Metafunction splitting a variadic to input and output types at a specific delimiter
/// @tparam delimiter Position to split the variadic of types
/// @tparam Types Variadic of type
template<size_t delimiter, typename ... Types>
struct Splitter {
  static_assert(delimiter != 0, "The delimiter should not be 0.");
  static_assert(delimiter < sizeof...(Types), "The delimiter should be inferior to the number of types.");

  /// @brief Type accessor of the input types
  using Inputs = internals::SplitInput_t<std::tuple<Types...>, std::make_integer_sequence<size_t, delimiter>>;

  /// @brief Type accessor of the output types
  using Outputs = internals::SplitOutput_t<std::tuple<Types...>,
                                           delimiter,
                                           std::make_integer_sequence<size_t, sizeof...(Types) - delimiter>>;
};

/// @brief Base definition of HasType testing if a type T is in tuple Tuple
/// @tparam T Type to test
/// @tparam Tuple Tuple of types
template<typename T, typename Tuple>
struct HasType;

/// @brief Default definition of HasType if the tuple is empty
/// @tparam T Type to test
template<typename T>
struct HasType<T, std::tuple<>> : std::false_type {};

/// @brief Definition of HasType if T is different than the front type of the variadic
/// @tparam T Type to test
/// @tparam Front Front type of variadic
/// @tparam Ts Variadic of types
template<typename T, typename Front, typename... Ts>
struct HasType<T, std::tuple<Front, Ts...>> : HasType<T, std::tuple<Ts...>> {};

/// @brief Definition of HasType if T is the same type as the front type of the variadic
/// @tparam T Type to test
/// @tparam Ts Variadic of types
template<typename T, typename... Ts>
struct HasType<T, std::tuple<T, Ts...>> : std::true_type {};

/// @brief Base definition of the implementation of the Intersect metafunction
/// @tparam T1 Tuple of types
/// @tparam T2 Tuple of types
/// @tparam Index Index of element in T1
/// @tparam Size Size of T1
template<typename T1, typename T2, size_t Index, size_t Size>
struct IntersectImpl;

/// @brief Definition of the implementation of the Intersect metafunction when arriving at the end of T1
/// @tparam T1 Tuple of types
/// @tparam T2 Tuple of types
/// @tparam Size Size of T1
template<typename T1, typename T2, size_t Size>
struct IntersectImpl<T1, T2, Size, Size> {
  /// @brief Returning an empty tuple when arriving at the end of T1
  using type = std::tuple<>;
};

/// @brief Definition of the implementation of the Intersect metafunction
/// @tparam T1 Tuple of types
/// @tparam T2 Tuple of types
/// @tparam Index Index of element in T1
/// @tparam Size Size of T1
template<typename T1, typename T2, size_t Index, size_t Size>
struct IntersectImpl {
  /// @brief Accessor to the type of the tuple representing the intersection between two tuples
  using type = std::conditional_t<
      internals::HasType<std::tuple_element_t<Index, T1>, T2>::value,
      internals::PushFront_t<typename IntersectImpl<T1, T2, Index + 1, Size>::type, std::tuple_element_t<Index, T1>>,
      typename IntersectImpl<T1, T2, Index + 1, Size>::type>;
};

/// @brief Intersect metafunction creating a tuple of types that are in T1 and T2
/// @tparam T1 Tuple of types
/// @tparam T2 Tuple of types
template<typename T1, typename T2>
struct Intersect {
  /// @brief Accessor to the type of the tuple of types that are in T1 and T2
  using type = typename IntersectImpl<T1, T2, 0, std::tuple_size_v<T1>>::type;
};
}

/// @brief Helper getting the input types from a list of template types (variadic)
template<size_t delta, typename ... Types>
using Inputs = typename internals::Splitter<delta, Types...>::Inputs;

/// @brief Helper getting the output types from a list of template types (variadic)
template<size_t delta, typename ... Types>
using Outputs = typename internals::Splitter<delta, Types...>::Outputs;

/// @brief Helper getting the intersection of types between two type tuples
template<class Tuple1, class Tuple2>
using Intersect_t = typename internals::Intersect<Tuple1, Tuple2>::type;

/// @brief Helper testing if a type T is in variadic Ts
/// @tparam T Type to test
/// @tparam Ts Variadic of types
template<class T, class ...Ts>
constexpr bool isContainedIn_v = std::disjunction_v<std::is_same<T, Ts>...>;

/// @brief Helper testing if a type is in a tuple of types
/// @tparam T Type to test
/// @tparam Tuple Tuple of types
template<class T, class Tuple>
constexpr bool isContainedInTuple_v = std::tuple_size_v<Intersect_t<std::tuple<T>, Tuple>> == 1;

/// @brief Create a string_view containing the type name
/// @tparam T Type to create the string_view from
/// @return string_view containing the name of the type
template<typename T>
constexpr auto typeToStrView() {
  std::string_view name, prefix, suffix;
#ifdef __clang__
  name = __PRETTY_FUNCTION__;
  prefix = "auto hh::tool::typeToStrView() [T = ";
  suffix = "]";
#elif defined(__GNUC__)
  name = __PRETTY_FUNCTION__;
  prefix = "constexpr auto hh::tool::typeToStrView() [with T = ";
  suffix = "]";
#elif defined(_MSC_VER)
  name = __FUNCSIG__;
    prefix = "auto __cdecl typeToStrView<";
    suffix = ">(void)";
#endif
  name.remove_prefix(prefix.size());
  name.remove_suffix(suffix.size());

  return name;
}

/// @brief Create a string containing the type name
/// @tparam T Type to create the string from
/// @return String containing the name of the type
template<typename T>
constexpr auto typeToStr() { return std::string(typeToStrView<T>()); }

}
}

#endif //HEDGEHOG_META_FUNCTIONS_H
