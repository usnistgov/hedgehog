
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

#ifndef HEDGEHOG_DEFAULT_EXECUTE_H
#define HEDGEHOG_DEFAULT_EXECUTE_H

#include "../implementor/implementor_execute.h"
#include "../../../behavior/execute.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Default execute implementor, only calls user-defined execute method in behavior::Execute interface
/// @tparam Input Input data type
template<class Input>
class DefaultExecute : public hh::core::implementor::ImplementorExecute<Input> {
  behavior::Execute<Input> *const executeNode_ = nullptr; ///< User abstraction for doing computation on data
 public:

  /// @brief Constructor needing a behavior::Execute implementation (user-defined implementation of behavior::Execute)
  /// @param executeNode Node inheriting from behavior::Execute
  explicit DefaultExecute(behavior::Execute<Input> *const executeNode) : executeNode_(executeNode) {
    if(executeNode_ == nullptr){
      throw std::runtime_error("The default execute implementor should have a valid execute node (!= nullptr);");
    }
  }

  /// @brief Default destructor
  virtual ~DefaultExecute() = default;

  /// @brief Interface to user defined execute method (behavior::Execute)
  /// @param data Data to transmit
  void execute(std::shared_ptr<Input> data) override { executeNode_->execute(std::move(data)); }
};

}
}
}

#endif //HEDGEHOG_DEFAULT_EXECUTE_H
