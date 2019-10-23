//
// Created by anb22 on 6/10/19.
//

#ifndef HEDGEHOG_CORE_EXECUTE_H
#define HEDGEHOG_CORE_EXECUTE_H
#include <memory>

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Interface to call execute on the different nodes
/// @tparam NodeInput Type of input data
template<class NodeInput>
class CoreExecute {
 public:
  /// @brief Wrapper to call the user-defined Execute::execute
  /// @param data Data to send to the Execute::execute for a specific node
  virtual void callExecute(std::shared_ptr<NodeInput> data) = 0;
};

}
#endif //HEDGEHOG_CORE_EXECUTE_H
