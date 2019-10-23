//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_EXECUTE_H
#define HEDGEHOG_EXECUTE_H

#include <memory>

/// @brief Hedgehog behavior namespace
namespace hh::behavior {
/// @brief Execute Behavior definition, node that has an execution for an Input data type
/// @tparam Input Input data type
template<class Input>
class Execute {
 public:
  /// @brief Virtual declaration of execute function for a data of type Input
  virtual void execute(std::shared_ptr<Input>) = 0;
};
}
#endif //HEDGEHOG_EXECUTE_H
