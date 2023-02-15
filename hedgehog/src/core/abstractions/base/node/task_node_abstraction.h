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



#ifndef HEDGEHOG_TASK_NODE_ABSTRACTION_H
#define HEDGEHOG_TASK_NODE_ABSTRACTION_H

#include <ostream>
#include "node_abstraction.h"
#include "../../../../api/memory_manager/manager/abstract_memory_manager.h"
#include "../../../../tools/nvtx_profiler.h"
#include "../printable_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Task core abstraction used to define some method for task-like behaving cores like CoreExecutionPipeline,
/// CoreStateManager, CoreTask
class TaskNodeAbstraction : public NodeAbstraction, public PrintableAbstraction {
 private:
  size_t numberReceivedElements_ = 0; ///< Number of elements received
  bool
      isActive_ = false, ///< Active flag
  isInitialized_ = false; ///< Is initialized flag

  std::chrono::nanoseconds
      perElementExecutionDuration_ = std::chrono::nanoseconds::zero(), ///< Node per element duration
  waitDuration_ = std::chrono::nanoseconds::zero(), ///< Node wait duration
  memoryWaitDuration_ = std::chrono::nanoseconds::zero(); ///< Node memory wait duration

  std::shared_ptr<NvtxProfiler>
      nvtxProfiler_ = nullptr; ///< Store hedgehog nvtx profiler for the task

  behavior::Node *
      node_ = nullptr; ///< Node attache to this task-like core

  std::mutex
      initMutex_ = {}; ///< Mutex used to set initialisation flag

 public:
  /// @brief Create the abstraction with the node's name
  /// @param name Name of the node
  /// @param node Node attached to this core
  explicit TaskNodeAbstraction(std::string const &name, behavior::Node *node) : NodeAbstraction(name), node_(node) {
    nvtxProfiler_ = std::make_shared<NvtxProfiler>(name);
  }

  /// @brief Default destructor
  ~TaskNodeAbstraction() override = default;

  /// @brief Accessor to task status
  /// @return True if the thread is up and running, else false
  [[nodiscard]] bool isActive() const { return isActive_; }

  /// @brief Accessor to initialized flag
  /// @return True if the task is initialized, else false
  [[nodiscard]] bool isInitialized() {
    initMutex_.lock();
    auto ret = isInitialized_;
    initMutex_.unlock();
    return ret;
  }

  /// @brief Accessor to the number of received elements
  /// @return Number of received elements
  [[nodiscard]] size_t numberReceivedElements() const { return numberReceivedElements_; }

  /// @brief Accessor to the duration the node was in a wait state
  /// @return Wait state duration in nanoseconds
  [[nodiscard]] std::chrono::nanoseconds const &waitDuration() const { return waitDuration_; }

  /// @brief Accessor to the duration the node was in a memory wait state
  /// @return Memory wait state duration in nanoseconds
  [[nodiscard]] std::chrono::nanoseconds const &memoryWaitDuration() const { return memoryWaitDuration_; }

  /// @brief Accessor to the NVTX profiler attached to the node
  /// @return Wait state duration in nanoseconds
  [[nodiscard]] std::shared_ptr<NvtxProfiler> const &nvtxProfiler() const { return nvtxProfiler_; }

  /// @brief Accessor to the duration the average duration of processing an input data
  /// @return Average duration of processing an input data in nanoseconds
  [[nodiscard]] std::chrono::nanoseconds perElementExecutionDuration() const {
    return
        this->numberReceivedElements() == 0 ?
        std::chrono::nanoseconds::zero() :
        (std::chrono::nanoseconds) (perElementExecutionDuration_ / this->numberReceivedElements());
  }

  /// @brief Setter to the task status
  /// @param isActive Status to set
  void isActive(bool isActive) { isActive_ = isActive; }

  /// @brief Increment the number of elements received
  void incrementNumberReceivedElements() { ++this->numberReceivedElements_; }

  /// @brief Increment the wait duration
  /// @param wait Duration in nanoseconds
  void incrementWaitDuration(std::chrono::nanoseconds const &wait) { this->waitDuration_ += wait; }

  /// @brief Increase the memory wait duration
  /// @param wait Duration in nanoseconds
  void incrementMemoryWaitDuration(std::chrono::nanoseconds const &wait) { this->memoryWaitDuration_ += wait; }

  /// @brief Increase the execution time per elements
  /// @param exec Duration in nanoseconds
  void incrementPerElementExecutionDuration(std::chrono::nanoseconds const &exec) {
    this->perElementExecutionDuration_ += exec;
  }

  /// @brief Pre run method, called only once
  virtual void preRun() = 0;

  /// @brief Run method, called when thread is attached
  virtual void run() = 0;

  /// @brief Post run method, called only once
  virtual void postRun() = 0;

  /// @brief Abstraction to add user-defined message for the printers
  /// @return String containing user-defined method
  [[nodiscard]] virtual std::string extraPrintingInformation() const = 0;

  /// @brief Flag accessor to the presence of memory manager attached
  /// @return True if the memory manager is attached, else false
  [[nodiscard]] virtual bool hasMemoryManagerAttached() const = 0;

  /// @brief Accessor to the attached memory manager
  /// @return The attached memory manager
  [[nodiscard]] virtual std::shared_ptr<AbstractMemoryManager> memoryManager() const = 0;

  /// @brief Node accessor
  /// @return Node attached to this core
  [[nodiscard]] behavior::Node *node() const override { return node_; }

 protected:
  /// @brief Set the task as initialized
  void setInitialized() {
    initMutex_.lock();
    isInitialized_ = true;
    initMutex_.unlock();
  }
};
}
}
}
#endif //HEDGEHOG_TASK_NODE_ABSTRACTION_H
