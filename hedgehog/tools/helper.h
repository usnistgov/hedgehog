//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-03-20.
//

#ifndef HEDGEHOG_HELPER_H
#define HEDGEHOG_HELPER_H

#include <tuple>
#include "../core/io/base/receiver/core_multi_receivers.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Hedgehog behavior namespace
namespace behavior {
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of MultiReceivers
/// @tparam Inputs Inputs type of MultiReceivers
template<class ...Inputs>
class MultiReceivers;

/// @brief Forward declaration of Sender
/// @tparam Output Output Type of sender
template<class Output>
class Sender;
#endif // DOXYGEN_SHOULD_SKIP_THIS
}

/// Helper used in hedgehog
namespace helper {

/// @brief Base definition of HelperMultiReceiversType
/// @tparam Inputs Tuple of input types
template<class Inputs>
struct HelperMultiReceiversType;

/// @brief Used helper to get the type of a MultiReceivers for inputs from a tuple of Input types
/// @tparam Inputs MultiReceivers inputs
template<class ...Inputs>
struct HelperMultiReceiversType<std::tuple<Inputs...>> {
  using type = behavior::MultiReceivers<Inputs...>; ///< Type of the MultiReceivers
};

/// @brief Base definition of HelperCoreMultiReceiversType
/// @tparam Inputs Tuple of input types
template<class Inputs>
struct HelperCoreMultiReceiversType;

/// @brief Used helper to get the type of a CoreMultiReceivers for inputs from a tuple of Input types
/// @tparam Inputs CoreMultiReceivers inputs
template<class ...Inputs>
struct HelperCoreMultiReceiversType<std::tuple<Inputs...>> {
  using type = core::CoreMultiReceivers<Inputs...>; ///< Type of the CoreMultiReceivers
};

}
}
#endif //HEDGEHOG_HELPER_H
