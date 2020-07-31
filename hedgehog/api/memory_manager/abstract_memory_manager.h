//
// Created by Bardakoff, Alexandre (IntlAssoc) on 7/30/20.
//

#ifndef HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H
#define HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H


#include "../../tools/traits.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Deprecated Version of Base Memory Manager
/// @tparam ManagedMemory Type of Memory managed
template<class ManagedMemory, class = void>
class [[deprecated("Use hh::MemoryManager instead")]] AbstractMemoryManager;

/// @brief Deprecated Version of Base Memory Manager
/// @tparam ManagedMemory Type of Memory managed
template<class ManagedMemory>
class [[deprecated("Use hh::MemoryManager instead")]] AbstractMemoryManager
    <ManagedMemory, typename std::enable_if_t<!traits::is_managed_memory_v < ManagedMemory>>
>;
}
#endif //HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H
