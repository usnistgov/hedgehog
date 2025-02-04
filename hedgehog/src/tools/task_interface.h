//  NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
//  software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
//  derivative works of the software or any portion of the software, and you may copy and distribute such modifications
//  or works. Modified works should carry a notice stating that you changed the software and should note the date and
//  nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
//  source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
//  EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
//  WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
//  CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
//  THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
//  are solely responsible for determining the appropriateness of using and distributing the software and you assume
//  all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
//  with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
//  operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
//  damage to property. The software developed by NIST employees is not subject to copyright protection within the
//  United States.

#ifndef HEDGEHOG_TASK_INTERFACE_H
#define HEDGEHOG_TASK_INTERFACE_H
#include "../api/memory_manager/managed_memory.h"
#include <memory>


/// @brief Hedgehog main namespace
namespace hh {
/// Hedgehog tool namespace
namespace tool {

/// @brief Class used to expose the Task interface in lambda functions of the lambda task
/// @tparam TaskType Type of the lambda task.
template <class TaskType>
class TaskInterface {
 private:
  TaskType *task_ = nullptr; ///< lambda task (LambdaTask::this)

 public:
  /// @brief Constructor with input task
  /// @param task Pointer to the task for which the interface is exposed
  TaskInterface(TaskType *task) : task_(task) {}

  /// @brief Default destructor
  ~TaskInterface() = default;

 public:
  /// @brief Exposes LambdaTask::addResult
  template <class T>
  void addResult(std::shared_ptr<T> data) {
      task_->addResult(data);
  }

  /// @brief Exposes LambdaTask::getManagedMemory
  [[nodiscard]] std::shared_ptr<ManagedMemory> getManagedMemory() { return task_->getManagedMemory(); }

  /// @brief Exposes LambdaTask::coreTask
  [[nodiscard]] auto const &coreTask() const { return task_->coreTask_; }

  /// @brief Exposes LambdaTask::deviceId
  [[nodiscard]] int deviceId() const { return task_->coreTask_->deviceId(); }

  /// @brief Operator that gives access to the methods of the specialized task
  [[nodiscard]] TaskType *operator->() const { return task_; }

  /// @brief Task setter
  /// @param Pointer to the new task
  void task(TaskType *task) { this->task_ = task; }
};

}
}

#endif //HEDGEHOG_TASK_INTERFACE_H
