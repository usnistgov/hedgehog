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

#ifndef HEDGEHOG_INTERNAL_LAMBDA_TASK_H
#define HEDGEHOG_INTERNAL_LAMBDA_TASK_H

#include <memory>
#include <string>
#include <ostream>
#include <utility>
#include <functional>

#include "../../../nodes/core_lambda_task.h"
#include "../../../../behavior/task_node.h"
#include "../../../../behavior/cleanable.h"
#include "../../../../behavior/copyable.h"
#include "../../../../behavior/can_terminate.h"
#include "../../../../tools/traits.h"
#include "lambda_multi_execute.h"
#include "../../../nodes/core_lambda_task.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog implementor namespace
namespace implementor {

/// @brief Internal implementation of the lambda task
/// @tparam SubType Type of the inheriting specialized lambda task (CRTP, void by default)
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template <typename SubType, size_t Separator, typename... AllTypes>
class InternalLambdaTask
    : public behavior::TaskNode,
      public behavior::CanTerminate,
      public behavior::Cleanable,
      public behavior::Copyable<InternalLambdaTask<SubType, Separator, AllTypes...>>,
      public tool::BehaviorMultiReceiversTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public LambdaMultiExecute<SubType, tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>> {
 public:
  /// @brief Type alias to the lambda container type
  using Lambdas = tool::LambdaContainerDeducer_t<SubType, tool::Inputs<Separator, AllTypes...>>;

 private:
  std::shared_ptr<LambdaCoreTask<SubType, Separator, AllTypes...>> const
      coreTask_ = nullptr; ///< Task core
  Lambdas lambdas_ = {}; ///< Tuple of lambdas (one lambda function per input type)
  SubType *self_ = nullptr; ///< Pointer to the user-defined task (CRTP)

 public:
  /// @brief TaskInterface is a friend of this class so it can expose the task interface
  friend tool::TaskInterface<SubType>;

 public:
  /// @brief Construct a internal lambda task with a pointer to the subclass, the lambda functions, a name, its number
  ///        of threads (default 1) and if the task should start automatically or not (default should not start automatically)
  /// @param self Pointer to the user-defined LambdaTask
  /// @param lambdas Lambda functions container
  /// @param name Task name
  /// @param numberThreads Task number of threads
  /// @param automaticStart Flag to start the execution of the task without data (sending automatically nullptr to each
  ///        of the input types)
  /// @throw std::runtime_error the number of threads == 0 or the core is not of the right type (do not derive from CoreTask)
  explicit InternalLambdaTask(SubType *self, Lambdas lambdas, std::string const &name, size_t const numberThreads,
          bool const automaticStart)
      : behavior::TaskNode(std::make_shared<LambdaCoreTask<SubType, Separator, AllTypes...>>(this,
                                                                                             name,
                                                                                             numberThreads,
                                                                                             automaticStart)),
        behavior::Copyable<InternalLambdaTask<SubType, Separator, AllTypes...>>(numberThreads),
        LambdaMultiExecute<SubType, tool::Inputs<Separator, AllTypes...>>(lambdas, self),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            (std::dynamic_pointer_cast<LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core()))),
        coreTask_(std::dynamic_pointer_cast<LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core())),
        lambdas_(lambdas),
        self_(self)
  {
    if (numberThreads == 0) { throw std::runtime_error("A task needs at least one thread."); }
    if (coreTask_ == nullptr) { throw std::runtime_error("The core used by the task should be a CoreTask."); }
  }

  /// @brief A custom core can be used to customize how the task behaves internally. For example, by default any input
  ///        is stored in a std::queue, it can be changed to a std::priority_queue instead through the core.
  /// @param self Pointer to the lambda task
  /// @param coreTask Custom core used to change the behavior of the task
  explicit InternalLambdaTask(SubType *self, std::shared_ptr<LambdaCoreTask<SubType, Separator, AllTypes...>> coreTask)
      : behavior::TaskNode(std::move(coreTask)),
        behavior::Copyable<InternalLambdaTask<SubType, Separator, AllTypes...>>(
            std::dynamic_pointer_cast<LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core())->numberThreads()),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            std::dynamic_pointer_cast<LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core())),
        LambdaMultiExecute<SubType, tool::Inputs<Separator, AllTypes...>>({}, self),
        coreTask_(std::dynamic_pointer_cast<LambdaCoreTask<SubType, Separator, AllTypes...>>(this->core())),
        lambdas_({}),
        self_(self)
  {
  }

  /// @brief Default destructor
  ~InternalLambdaTask() override = default;

 public:
  /// @brief Set a lambda function for an input type
  /// @tparam Input Input type
  /// @param lambda Lambda function
  template<hh::tool::ContainsInTupleConcept<tool::Inputs<Separator, AllTypes...>> Input>
  void setLambda(std::function<void(std::shared_ptr<Input>, tool::TaskInterface<SubType>)> lambda) {
      std::get<std::function<void(std::shared_ptr<Input>, tool::TaskInterface<SubType>)>>(lambdas_) = lambda;
      LambdaMultiExecute<SubType, tool::Inputs<Separator, AllTypes...>>::reinitialize(lambdas_, self_);
  }

  /// @brief Lambda functions container accessor
  /// @return Lambda functions container
  [[nodiscard]] auto lambdas() const { return lambdas_; }

  /// @brief Belonging graph id accessor
  /// @return  Belonging graph id
  [[nodiscard]] size_t graphId() const { return coreTask_->graphId(); }

  /// @brief Default termination rule, it terminates if there is no predecessor connection and there is no input data
  /// @return True if the task can terminate, else false
  [[nodiscard]] bool canTerminate() const override {
    return !coreTask_->hasNotifierConnected() && coreTask_->receiversEmpty();
  }

  /// @brief Automatic start flag accessor
  /// @return True if the task is set to automatically start, else false
  [[nodiscard]] bool automaticStart() const { return this->coreTask()->automaticStart(); }

 protected:
  /// @brief Accessor to the core task
  /// @return Core task
  std::shared_ptr<LambdaCoreTask<SubType, Separator, AllTypes...>> const &coreTask() const { return coreTask_; }

  /// @brief Accessor to device id linked to the task (default 0 for CPU task)
  /// @return Device id linked to the task
  [[nodiscard]] int deviceId() const { return coreTask_->deviceId(); }
};

}
}
}

#endif //HEDGEHOG_INTERNAL_LAMBDA_TASK_H
