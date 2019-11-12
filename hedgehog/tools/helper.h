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
