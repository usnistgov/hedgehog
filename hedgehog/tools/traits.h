//
// Created by anb22 on 2/22/19.
//

#ifndef HEDGEHOG_TRAITS_H
#define HEDGEHOG_TRAITS_H

#include <type_traits>

template<class T>
class MemoryData;

namespace HedgehogTraits {
// Check if all parameters in a parameters pack are unique
template<typename...>
inline constexpr auto isUnique = std::true_type{};

template<typename T, typename... Rest>
inline constexpr auto isUnique<T, Rest...> = std::bool_constant<
    (!std::is_same_v<T, Rest> && ...) && isUnique<Rest...>
>{};

// Test correctness f type given to a memory manager
template<class POSSIBLEMAMANGEDMEMORY>
struct IsManagedMemory {
  constexpr static bool const value = std::is_base_of_v<MemoryData<POSSIBLEMAMANGEDMEMORY>, POSSIBLEMAMANGEDMEMORY>
      && std::is_default_constructible_v<POSSIBLEMAMANGEDMEMORY>;
};

template<class POSSIBLEMAMANGEDMEMORY>
inline constexpr bool is_managed_memory_v = IsManagedMemory<POSSIBLEMAMANGEDMEMORY>::value;

// Check if a template T is in Template pack Ts
template<typename T, typename... Ts>
constexpr bool contains() { return std::disjunction_v<std::is_same<T, Ts>...>; }

template<class T, class... Ts>
struct Contains {
  constexpr static bool value = std::disjunction_v<std::is_same<T, Ts>...>;
};

template<class T, class... Ts>
struct Contains<T, std::tuple<Ts...> > : Contains<T, Ts...> {};

template<class T1, class T2, class ID>
struct _is_included_;

template<class T1, class T2, std::size_t... Is>
struct _is_included_<T1, T2, std::integer_sequence<std::size_t, Is...> > {
  static bool const value = std::conjunction_v<Contains<typename std::tuple_element<Is, T1>::type, T2>...>;
};

template<class T1, class T2>
struct is_included : _is_included_<T1, T2, std::make_integer_sequence<std::size_t, std::tuple_size<T1>::value> > {};

template<class T1, class T2>
inline constexpr bool is_included_v = is_included<T1, T2>::value;

template<class T1, class... Ts>
inline constexpr bool contains_v = Contains<T1, std::tuple<Ts...>>::value;

template<class T>
constexpr std::string_view type_name() {
  using namespace std;
  string_view p = __PRETTY_FUNCTION__;
#ifdef __clang__
  return string_view(p.data() + 50, p.size() - 51);
//  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
#  if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#  else
  return string_view(p.data() + 65, p.find(';', 65) - 65);
//	return string_view(p.data() + 49, p.find(';', 49) - 49);
#  endif
#endif
}
}

#endif //HEDGEHOG_TRAITS_H
