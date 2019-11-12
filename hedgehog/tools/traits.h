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


#ifndef HEDGEHOG_TRAITS_H
#define HEDGEHOG_TRAITS_H

#include <type_traits>
#include <utility>
#include <string_view>

#if defined( __GLIBCXX__ ) || defined( __GLIBCPP__ )
#include <cxxabi.h>
#endif


/// @brief Hedgehog main namespace
namespace hh {
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of MemoryData
/// @tparam T Type presented by the MemoryData
template<class T>
class MemoryData;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// Traits used in Hedgehog
namespace traits {

/// @brief Check if all parameters in a parameters pack are unique, default case
/// @tparam ... Default list of templates
template<typename...>
inline constexpr auto isUnique = std::true_type{};

/// @brief Check if all parameters in a parameters pack are unique, looping through templates
/// @tparam T Current type to test
/// @tparam Rest Following type
template<typename T, typename... Rest>
inline constexpr auto isUnique<T, Rest...> =
    std::bool_constant<(!std::is_same_v<T, Rest> && ...) && isUnique<Rest...>>{};

/// @brief Test correctness of type given to a memory manager
/// @tparam PossibleManagedMemory Type to test to be used in AbstractMemoryManager
template<class PossibleManagedMemory>
struct IsManagedMemory {
  constexpr static bool const value =
      std::is_base_of_v<MemoryData<PossibleManagedMemory>, PossibleManagedMemory>
          && std::is_default_constructible_v<PossibleManagedMemory>;    ///< True if PossibleManagedMemory can be
  ///< handled by an AbstractMemoryManager, else
  ///< False
};

/// @brief Direct IsManagedMemory value accessor
/// @tparam PossibleManagedMemory Type to test to be used in AbstractMemoryManager
template<class PossibleManagedMemory>
inline constexpr bool is_managed_memory_v = IsManagedMemory<PossibleManagedMemory>::value;

/// @brief Check if a template T is in Template pack Ts
/// @tparam T Type to test
/// @tparam Ts Parameter pack
/// @return True if T is in Ts, else False
template<class T, class... Ts>
struct Contains {
  constexpr static bool value = std::disjunction_v<std::is_same<T, Ts>...>; ///< True if T in Ts, else False
};

/// @brief Check if a template T is in tuple containing template pack Ts
/// @tparam T Type to test
/// @tparam Ts Parameter pack
template<class T, class... Ts>
struct Contains<T, std::tuple<Ts...> > : Contains<T, Ts...> {};

/// @brief Direct Contains value accessor
/// @tparam T Type to test
/// @tparam Ts Parameter pack
template<class T, class... Ts>
inline constexpr bool Contains_v = Contains<T, std::tuple<Ts...>>::value;

/// @brief _is_included_ Default case
/// @tparam T1 Tuple of types that should be in type T2
/// @tparam T2 Tuple of types that should contain all type of T1
/// @tparam Is List of index of tuple T1
template<class T1, class T2, class Is>
struct _is_included_;

/// @brief Check if a tuple of types T1 is included in a tuple of type T2, all type in T1 are in T2
/// @tparam T1 Tuple of types that should be in type T2
/// @tparam T2 Tuple of types that should contain all type of T1
/// @tparam Is List of index of tuple T1
template<class T1, class T2, std::size_t... Is>
struct _is_included_<T1, T2, std::integer_sequence<std::size_t, Is...> > {
  static bool const
      value = std::disjunction_v<Contains<typename std::tuple_element<Is, T1>::type, T2>...>; ///< True if all type in
  ///< T1 are in T2, else False
};

/// @brief Check if a tuple of types T1 is included in a tuple of type T2, all type in T1 are in T2
/// @tparam T1 Tuple of types that should be in type T2
/// @tparam T2 Tuple of types that should contain all type of T1
template<class T1, class T2>
struct is_included : _is_included_<T1, T2, std::make_integer_sequence<std::size_t, std::tuple_size<T1>::value> > {};

/// @brief Value accessor to test if the types of tuple T1 are in types of tuple T2
/// @tparam T1 Tuple of types that should be in type T2
/// @tparam T2 Tuple of types that should contain all type of T1
template<class T1, class T2>
inline constexpr bool is_included_v = is_included<T1, T2>::value;

/// @brief Create a std::string_view containing a full type name of T
/// @tparam T Type to get the name from
/// @return std::string_view containing a full type name of T
template<typename T>
constexpr auto type_name() {
  std::string_view name, prefix, suffix;
#ifdef __clang__
  name = __PRETTY_FUNCTION__;
  prefix = "auto hh::traits::type_name() [T = ";
  suffix = "]";
#elif defined(__GNUC__)
  name = __PRETTY_FUNCTION__;
  prefix = "constexpr auto hh::traits::type_name() [with T = ";
  suffix = "]";
#elif defined(_MSC_VER)
  name = __FUNCSIG__;
    prefix = "auto __cdecl hh::traits::type_name<";
    suffix = ">(void)";
#endif
  name.remove_suffix(suffix.size());
  name.remove_prefix(prefix.size());
  return name;
}
}
}
#endif //HEDGEHOG_TRAITS_H
