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

#ifndef HEDGEHOG_TASK_NODE_H
#define HEDGEHOG_TASK_NODE_H

#include "node.h"

#include "../api/memory_manager/manager/abstract_memory_manager.h"
#include "../core/abstractions/base/node/task_node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog behavior namespace
namespace behavior {

/// Behavior abstraction for TaskNode
/// @brief Proposed abstraction for attached memory manager and user overloadable method (initialize, shutdown,
/// extraPrintingInformation)
class TaskNode : public Node {
 private:
  std::shared_ptr<AbstractMemoryManager>
      mm_ = nullptr; ///< Memory manager attached to the task
  std::shared_ptr<hh::core::abstraction::TaskNodeAbstraction> const
      taskNodeAbstraction_ = nullptr; ///< Core abstraction for the task
 public:
  /// @brief Constructor using a core
  /// @param core Task core
  explicit TaskNode(std::shared_ptr<hh::core::abstraction::TaskNodeAbstraction> core)
      : Node(std::move(core)),
        taskNodeAbstraction_(std::dynamic_pointer_cast<core::abstraction::TaskNodeAbstraction>(this->core())) {}

  /// @brief Default destructor
  ~TaskNode() override = default;

  /// @brief Memory manager accessor
  /// @return Attached memory manager
  [[nodiscard]] std::shared_ptr<AbstractMemoryManager> const &memoryManager() const { return mm_; }

  /// @brief Connect a memory manager to a task
  /// @param mm Memory manager to attach to t he node
  void connectMemoryManager(std::shared_ptr<AbstractMemoryManager> mm) { mm_ = std::move(mm); }

  /// @brief Get a managed memory for the memory manager attached to the task, can block if none are available at the
  /// time of the call
  /// @return A managed memory
  /// @throw std::runtime_error if no memory manager is attached to the task
  std::shared_ptr<ManagedMemory> getManagedMemory() {
    if (mm_ == nullptr) {
      std::ostringstream oss;
      oss << "For the node:\"" << this->name()
          << "\"in order to get managed memory, you need first to connect a memory manager to the task via "
             "\"connectMemoryManager()\""
          << std::endl;
      throw std::runtime_error(oss.str());
    }
    auto start = std::chrono::system_clock::now();
    this->taskNodeAbstraction_->nvtxProfiler()->startRangeWaitingForMemory();
    auto data = mm_->getManagedMemory();
    this->taskNodeAbstraction_->nvtxProfiler()->endRangeWaitingForMem();
    auto finish = std::chrono::system_clock::now();
    this->taskNodeAbstraction_->incrementMemoryWaitDuration(finish - start);
    return data;
  }

  /// @brief initialize step for the task
  virtual void initialize() {}

  /// @brief shutdown step for the task
  virtual void shutdown() {}

  /// @brief Print extra information for the task
  /// @return string with extra information
  [[nodiscard]] virtual std::string extraPrintingInformation() const { return ""; }
};
}
}

#endif //HEDGEHOG_TASK_NODE_H
