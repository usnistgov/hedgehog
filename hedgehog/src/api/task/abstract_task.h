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



#ifndef HEDGEHOG_ABSTRACT_TASK_H
#define HEDGEHOG_ABSTRACT_TASK_H

#include <memory>
#include <string>
#include <ostream>
#include <utility>

#include "../../behavior/copyable.h"
#include "../../behavior/cleanable.h"
#include "../../behavior/task_node.h"
#include "../../behavior/multi_execute.h"
#include "../../behavior/input_output/multi_receivers.h"
#include "../../behavior/input_output/task_multi_senders.h"

#include "../../core/nodes/core_task.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Base node for computation
/// @details Hedgehog Graph's node made for processing data from the their overloaded execute method from
/// Execute::execute.
///
/// An AbstractTask can be bound to multiple threads, forming a group of tasks. The base AbstractTask will be copied n-1
/// times and each of them will be bound to a thread. The AbstractTask::copy method must be overloaded to use this
/// functionality. Also, if the AbstractTask is part of a Graph that will be duplicated with an AbstractExecutionPipeline,
/// AbstractTask::copy method needs to be overloaded.
///
/// A MemoryManager could be linked to a task:
/// @code
/// // Implementation of a basic Task
/// class StaticSizeTToManagedMemory : public hh::AbstractTask<1, size_t, StaticManagedMemory> {
/// public:
///  explicit StaticSizeTToManagedMemory(size_t numberThread = 1)
///      : hh::AbstractTask<1, size_t, StaticManagedMemory>("StaticManagedMemory", numberThread, false) {}
///  ~StaticSizeTToManagedMemory() override = default;
///
///  void execute([[maybe_unused]] std::shared_ptr<size_t> data) override {
///    this->addResult(std::dynamic_pointer_cast<StaticManagedMemory>(this->getManagedMemory()));
///  }
///
///  std::shared_ptr<hh::AbstractTask<1, size_t, StaticManagedMemory>> copy() override {
///    return std::make_shared<StaticSizeTToManagedMemory>(this->numberThreads());
///  }
/// };
///
/// // Implementation of a managed memory
/// class StaticManagedMemory : public hh::ManagedMemory {
///  int *array_ = nullptr;
/// public:
///  explicit StaticManagedMemory(size_t const sizeAlloc) { array_ = new int[sizeAlloc]; }
///  ~StaticManagedMemory() override { delete[] array_; }
/// };
///
/// // Instantiation and connection with a memory manager
/// auto staticTask = std::make_shared<StaticSizeTToManagedMemory>(2);
/// auto staticMM = std::make_shared<hh::StaticMemoryManager<StaticManagedMemory, size_t>>(2, 2);
/// staticTask->connectMemoryManager(staticMM);
/// @endcode
///
/// The default order of execution is:
///     -# The group is created, for each task:
///     -# Threads are spawned for each instance in the group,
///     -# The AbstractTask::initialize is called,
///     -# The memory manager, if it exists, is bound to the task (the device Id is shared, and is initialized),
///     -# Execute is called, while AbstractTask::canTerminate is True:
///         - If the task is set to start automatically Execute::execute is called with nullptr,
///         - If not, the task will wait for a data to come and Execute::execute is called with the received data,
///     -# AbstractTask::shutdown is called, and signals to the linked nodes to wake up.
///
/// Only Execute::execute method needs to be overloaded for each AbstractTask input type.
///
/// \par Virtual functions
///     - Execute::execute (one for each of TaskInputs) <br>
///     - AbstractTask::initialize <br>
///     - AbstractTask::shutdown <br>
///     - AbstractTask::copy (mandatory if usage of numberThreads greater than 1 or AbstractExecutionPipeline) <br>
///     - Node::canTerminate (mandatory if cycle in the graph) <br>
///     - Node::extraPrintingInformation
///
/// @attention In case of a cycle in the graph AbstractTask::canTerminate needs to be overloaded or the graph will
/// deadlock. By default, CanTerminate::canTerminate will be true if there is no "input node" connected AND no data
/// available in the task input queue (CF tutorial 3).
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class AbstractTask
    : public behavior::TaskNode,
      public behavior::CanTerminate,
      public behavior::Cleanable,
      public behavior::Copyable<AbstractTask<Separator, AllTypes...>>,
      public tool::BehaviorMultiReceiversTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorMultiExecuteTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
      public tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>> {
 private:
  std::shared_ptr<hh::core::CoreTask<Separator, AllTypes...>> const coreTask_ = nullptr; ///< Task core
 public:
  /// @brief Default constructor (Task with one thread named "Task")
  AbstractTask() : AbstractTask("Task", 1, false) {};

  /// AbstractTask main constructor
  /// @brief Construct a task with a name, its number of threads (default 1) and if the task should start automatically
  /// or not (default should not start automatically)
  /// @param name Task name
  /// @param numberThreads Task number of threads
  /// @param automaticStart Flag to start the execution of the task without data (sending automatically nullptr to each
  /// of the input types)
  /// @throw std::runtime_error the number of threads == 0 or the core is not of the right type (do not derive from CoreTask)
  explicit AbstractTask(std::string const &name, size_t const numberThreads = 1, bool const automaticStart = false)
      : behavior::TaskNode(std::make_shared<core::CoreTask<Separator, AllTypes...>>(this,
                                                                                    name,
                                                                                    numberThreads,
                                                                                    automaticStart)),
        behavior::Copyable<AbstractTask<Separator, AllTypes...>>(numberThreads),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            (std::dynamic_pointer_cast<hh::core::CoreTask<Separator, AllTypes...>>(this->core()))),
        coreTask_(std::dynamic_pointer_cast<core::CoreTask<Separator, AllTypes...>>(this->core())) {
    if (numberThreads == 0) { throw std::runtime_error("A task needs at least one thread."); }
    if (coreTask_ == nullptr) { throw std::runtime_error("The core used by the task should be a CoreTask."); }
  }

  /// Construct a task from a user-defined core.
  /// @brief A custom core can be used to customize how the task behaves internally. For example, by default any input
  /// is stored in a std::queue, it can be changed to a std::priority_queue instead through the core.
  /// @param coreTask Custom core used to change the behavior of the task
  explicit AbstractTask(std::shared_ptr<hh::core::CoreTask<Separator, AllTypes...>> coreTask)
      : behavior::TaskNode(std::move(coreTask)),
        behavior::Copyable<AbstractTask<Separator, AllTypes...>>(
            std::dynamic_pointer_cast<core::CoreTask<Separator, AllTypes...>>(this->core())->numberThreads()),
        tool::BehaviorTaskMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>(
            std::dynamic_pointer_cast<hh::core::CoreTask<Separator, AllTypes...>>(this->core())),
        coreTask_(std::dynamic_pointer_cast<core::CoreTask<Separator, AllTypes...>>(this->core())) {
  }

  /// @brief Default task destructor
  ~AbstractTask() override = default;

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
  std::shared_ptr<hh::core::CoreTask<Separator, AllTypes...>> const &coreTask() const { return coreTask_; }

  /// @brief Accessor to device id linked to the task (default 0 for CPU task)
  /// @return Device id linked to the task
  [[nodiscard]] int deviceId() const { return coreTask_->deviceId(); }
};
}
#endif //HEDGEHOG_ABSTRACT_TASK_H
