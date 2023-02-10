

#ifndef HEDGEHOG_DEFAULT_MULTI_EXECUTES_H_
#define HEDGEHOG_DEFAULT_MULTI_EXECUTES_H_

#include "default_execute.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {


/// @brief Default concrete implementation for different typed execution method for tasks
/// @tparam Inputs Input data types
template<class ...Inputs>
class DefaultMultiExecutes : public DefaultExecute<Inputs>... {
 public:

  /// @brief Constructor using an user-defined node capable of doing computation (inheriting from behavior::Execute for
  /// all Inputs types)
  /// @tparam MultiExecute Type of the user-defined node
  /// @param multiExecutesNode user-defined node capable of doing computation (inheriting from behavior::Execute for
  /// all Inputs types)
  template <class MultiExecute> requires (std::is_base_of_v<behavior::Execute<Inputs>, MultiExecute> && ...)
  explicit DefaultMultiExecutes(MultiExecute *const multiExecutesNode) : DefaultExecute<Inputs>(multiExecutesNode)... {}

  /// @brief Default destructor
  ~DefaultMultiExecutes() override = default;
};

}
}
}
#endif //HEDGEHOG_DEFAULT_MULTI_EXECUTES_H_
