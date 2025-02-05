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

#ifndef HEDGEHOG_LAMBDA_EXECUTE_H
#define HEDGEHOG_LAMBDA_EXECUTE_H
#include <memory>
#include <functional>
#include "../../../../tools/task_interface.h"
#include "../../../../tools/traits.h"


/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Implementation of the Execute traits for the lambda task
/// @tparam LambdaTaskType Type of the lambda task (CRTP)
/// @tparam Input Type of the input.
template <typename LambdaTaskType, typename Input>
class LambdaExecute
    : public tool::BehaviorMultiExecuteTypeDeducer_t<std::tuple<Input>>
{
 public:
  /// @brief Type of the lambda function
  using LambdaType = std::function<void(std::shared_ptr<Input>, tool::TaskInterface<LambdaTaskType>)>;

 private:
  LambdaType lambda_; ///< Lambda function that should process the input type
  tool::TaskInterface<LambdaTaskType> task_; ///< Interface to the inheriting task

 public:
  /// @brief Constructor with the lambda and the task
  /// @param lambda Lambda function that will process the input type
  /// @param task Pointer to the task (CRTP)
  LambdaExecute(LambdaType lambda, LambdaTaskType *task) : lambda_(lambda), task_(task) { }

  /// @brief Implementation of the execute function from the Execute trait
  /// @param data Input data
  void execute(std::shared_ptr<Input> data) override {
      lambda_(data, task_);
  }

  /// @brief Reinitialize the members when the user call the `setLambda` function in the lambda task
  /// @parma lambda New lambda function
  /// @parma task New task
  void reinitialize(LambdaType lambda, LambdaTaskType *task) {
      lambda_ = lambda;
      task_.task(task);
  }
};

}
}
}

#endif //HEDGEHOG_TASK_LAMBDA_EXECUTE_H
