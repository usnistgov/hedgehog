//
// Created by anb22 on 5/2/19.
//


#ifndef HEDGEHOG_MULTI_RECEIVERS_H
#define HEDGEHOG_MULTI_RECEIVERS_H

#include "../node.h"

/// @brief Hedgehog behavior namespace
namespace hh::behavior {
/// @brief MultiReceivers Behavior definition, node has a list of input types
/// @tparam Inputs Input data types
template<class ...Inputs>
class MultiReceivers : public virtual Node {
 public:
  /// @brief Tuple with the list of input types
  using inputs_t = std::tuple<Inputs...>;
};
}
#endif //HEDGEHOG_MULTI_RECEIVERS_H
