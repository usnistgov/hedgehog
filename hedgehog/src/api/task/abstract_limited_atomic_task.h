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

#ifndef HEDGEHOG_ABSTRACT_LIMITED_ATOMIC_TASK_H
#define HEDGEHOG_ABSTRACT_LIMITED_ATOMIC_TASK_H

#include "abstract_task.h"
#include "../../core/implementors/concrete_implementor/slot/atomic_slot.h"
#include "../../core/implementors/concrete_implementor/receiver/limited_atomic_queue_receiver.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Task using a communication layer (slot) and a limited queue (LimitedAtomicQueueReceiver) to receive data
/// using atomics
/// @details For more details on the base, go to AbstractTask documentation.
/// @tparam MaxCapacity Queue maximum capacity
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<long long MaxCapacity, size_t Separator, class ...AllTypes>
class AbstractLimitedAtomicTask : public AbstractTask<Separator, AllTypes...> {
 public:
  /// @brief Create an AbstractLimitedAtomicTask
  /// @brief Construct a task with a name, its number of threads (default 1) and if the task should start automatically
  /// or not (default should not start automatically)
  /// @param name Task name
  /// @param numberThreads Task number of threads
  /// @param automaticStart Flag to start the execution of the task without data (sending automatically nullptr to each
  /// of the input types)
  /// @throw std::runtime_error the number of threads == 0 or the core is not of the right type (do not derive from CoreTask)
  explicit AbstractLimitedAtomicTask(
      std::string const &name, size_t const numberThreads = 1, bool const automaticStart = false)
      : hh::AbstractTask<Separator, AllTypes...>(
      std::make_shared<hh::core::CoreTask<Separator, AllTypes...>>(
          this,
          name, numberThreads, automaticStart,
          std::make_shared<hh::core::implementor::AtomicSlot>(),
          std::make_shared<hh::core::implementor::MLAQR<MaxCapacity, Separator, AllTypes...>>(),
          std::make_shared<hh::tool::DME<Separator, AllTypes...>>(this),
          std::make_shared<hh::core::implementor::DefaultNotifier>(),
          std::make_shared<hh::tool::MDS<Separator, AllTypes...>>()
      )) {}

  /// @brief Default constructor creating a mono threaded task called "LimitedAtomicTask"
  explicit AbstractLimitedAtomicTask() : AbstractLimitedAtomicTask("LimitedAtomicTask") {};
};

}

#endif //HEDGEHOG_ABSTRACT_LIMITED_ATOMIC_TASK_H
