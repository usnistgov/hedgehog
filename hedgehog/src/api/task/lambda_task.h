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

#ifndef HEDGEHOG_LAMBDA_TASK_H
#define HEDGEHOG_LAMBDA_TASK_H

#include "../../core/implementors/concrete_implementor/task/internal_lambda_task.h"


/// @brief Hedgehog main namespace
namespace hh {

/// @brief Type alias to the InternalLambdaTask
template <class SubType, size_t Separator, class ...AllTypes>
using ILT = core::implementor::InternalLambdaTask<SubType, Separator, AllTypes...>;

/// @brief Specialized lambda task interface (for user-defined lambda task)
/// @tparam SubType Type of the inheriting specialized lambda task (CRTP, void by default)
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template <class SubType, size_t Separator, class ...AllTypes>
class SpecializedLambdaTask : 
    public ILT<SubType, Separator, AllTypes...> {
 public:
  /// @brief Type alias to the lambda container type
  using Lambdas = tool::LambdaContainerDeducer_t<SubType, tool::Inputs<Separator, AllTypes...>>;

 public:
  /// @brief Construct a lambda task with a pointer to the subclass, a tuple of lambda functions, a name, its number of 
  ///        threads (default 1) and if the task should start automatically or not (default should not start automatically)
  /// @param self Pointer to the user-defined lambda task (CRTP)
  /// @param lambda Tuple of lambda functions (one function per input type of the task)
  /// @param name Task name
  /// @param numberThreads Task number of threads
  /// @param automaticStart Flag to start the execution of the task without data (sending automatically nullptr to each
  /// of the input types)
  /// @throw std::runtime_error the number of threads == 0 or the core is not of the right type (do not derive from CoreTask)
  explicit SpecializedLambdaTask(SubType *self, Lambdas lambdas, std::string const &name = "Task", size_t const numberThreads = 1,
          bool const automaticStart = false)
      : ILT<SubType, Separator, AllTypes...>(self, lambdas, name, numberThreads, automaticStart) {}

  /// @brief Construct a lambda task with the pionter to the subclass, a name, its number of threads (default 1) and if
  ///        the task should start automatically or not (default should not start automatically)
  /// @param self Pointer to the user-defined lambda task (CRTP)
  /// @param name Task name
  /// @param numberThreads Task number of threads
  /// @param automaticStart Flag to start the execution of the task without data (sending automatically nullptr to each
  /// of the input types)
  /// @throw std::runtime_error the number of threads == 0 or the core is not of the right type (do not derive from CoreTask)
  explicit SpecializedLambdaTask(SubType *self, std::string const &name = "Task", size_t const numberThreads = 1,
          bool const automaticStart = false)
      : SpecializedLambdaTask<SubType, Separator, AllTypes...>(self, {}, name, numberThreads, automaticStart) {}

  /// @brief A custom core can be used to customize how the task behaves internally. For example, by default any input
  ///        is stored in a std::queue, it can be changed to a std::priority_queue instead through the core.
  /// @param self Pointer to the lambda task
  /// @param coreTask Custom core used to change the behavior of the task
  explicit SpecializedLambdaTask(SubType *self, std::shared_ptr<core::implementor::LambdaCoreTask<SubType, Separator, AllTypes...>> coreTask) 
      : ILT<SubType, Separator, AllTypes...>(self, coreTask) {}
};

/// @brief Default lambda task (there is no subtype so it shouldn't be specified)
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template <size_t Separator, class ...AllTypes>
class LambdaTask
    : public ILT<LambdaTask<Separator, AllTypes...>, Separator, AllTypes...> {
 public:
  /// @brief Type alias to the lambda container type
  using Lambdas = tool::LambdaContainerDeducer_t<LambdaTask<Separator, AllTypes...>, tool::Inputs<Separator, AllTypes...>>;

 public:
  /// @brief Construct a lambda task with a tuple of lambda functions, a name, its number of threads (default 1) and if
  ///        the task should start automatically or not (default should not start automatically)
  /// @param lambda Tuple of lambda functions (one function per input type of the task)
  /// @param name Task name
  /// @param numberThreads Task number of threads
  /// @param automaticStart Flag to start the execution of the task without data (sending automatically nullptr to each
  /// of the input types)
  /// @throw std::runtime_error the number of threads == 0 or the core is not of the right type (do not derive from CoreTask)
  explicit LambdaTask(Lambdas lambdas, std::string const &name = "Task", size_t const numberThreads = 1,
          bool const automaticStart = false)
      : ILT<LambdaTask<Separator, AllTypes...>, Separator, AllTypes...>(this, lambdas, name, numberThreads, automaticStart) {}

  /// @brief Construct a lambda task with a tuple of lambda functions, a name, its number of threads (default 1) and if
  ///        the task should start automatically or not (default should not start automatically)
  /// @param lambda Tuple of lambda functions (one function per input type of the task)
  /// @param name Task name
  /// @param numberThreads Task number of threads
  /// @param automaticStart Flag to start the execution of the task without data (sending automatically nullptr to each
  /// of the input types)
  /// @throw std::runtime_error the number of threads == 0 or the core is not of the right type (do not derive from CoreTask)
  explicit LambdaTask(std::string const &name = "Task", size_t const numberThreads = 1, bool const automaticStart = false)
      : LambdaTask<Separator, AllTypes...>({}, name, numberThreads, automaticStart) {}

  /// @brief A custom core can be used to customize how the task behaves internally. For example, by default any input
  ///        is stored in a std::queue, it can be changed to a std::priority_queue instead through the core.
  /// @param self Pointer to the lambda task
  /// @param coreTask Custom core used to change the behavior of the task
  explicit LambdaTask(std::shared_ptr<core::implementor::LambdaCoreTask<void, Separator, AllTypes...>> coreTask) 
      : ILT<LambdaTask<Separator, AllTypes...>, Separator, AllTypes...>(this, coreTask) {}

  /// @brief Implementation of the copy trait
  /// @return Copy of the lambda task
  std::shared_ptr<ILT<LambdaTask, Separator, AllTypes...>>
  copy() override {
    return std::make_shared<LambdaTask>(this->lambdas(), this->name(), this->numberThreads(), this->automaticStart());
  }
};

}

#endif //HEDGEHOG_LAMBDA_TASK_H
