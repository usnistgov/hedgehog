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

#ifndef HEDGEHOG_EXECUTION_PIPELINE_NODE_ABSTRACTION_H_
#define HEDGEHOG_EXECUTION_PIPELINE_NODE_ABSTRACTION_H_

#include "task_node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Abstraction specialized for the execution pipeline
class ExecutionPipelineNodeAbstraction : public TaskNodeAbstraction{
 public:
  /// @brief Constructor using the node name
  /// @param name Node's name
  /// @param node Node attached to this core
  explicit ExecutionPipelineNodeAbstraction(std::string const &name, behavior::Node *node)
      : TaskNodeAbstraction(name, node) {
    this->printOptions().background({0xC0, 0xC0, 0xC0, 0xff});
  }

  /// @brief Default destructor
  ~ExecutionPipelineNodeAbstraction() override = default;

  /// @brief Node ids [nodeId, nodeGroupId] accessor
  /// @return  Node ids [nodeId, nodeGroupId] (this and this)
  [[nodiscard]] std::vector<std::pair<std::string const, std::string const>> ids() const override {
    return {{this->id(), this->id()}};
  }

  /// @brief Test if a node has a memory attached, an execution pipeline can't have a memory manager
  /// @return false
  [[nodiscard]] bool hasMemoryManagerAttached() const override { return false; }

  /// @brief Accessor to memory manager, an execution pipeline has no memory manager
  /// @return nullptr
  [[nodiscard]] std::shared_ptr<AbstractMemoryManager> memoryManager() const override { return nullptr; }

  /// @brief Launch the graphs inside of the execution pipeline, called when the outer graph is executed
  /// @param waitForInitialization Wait for internal nodes to be initialized flags
  virtual void launchGraphThreads(bool waitForInitialization) = 0;

  /// @brief Getter to the min max execution duration from the nodes inside the graphs in the execution pipeline
  /// @return Min max execution duration from the nodes inside the graphs in the execution pipeline
  [[nodiscard]] virtual std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxExecutionDuration() const  = 0;

  /// @brief Getter to the min max wait duration from the nodes inside the graphs in the execution pipeline
  /// @return Min max wait duration from the nodes inside the graphs in the execution pipeline
  [[nodiscard]] virtual std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxWaitDuration() const  = 0;
};
}
}
}
#endif //HEDGEHOG_EXECUTION_PIPELINE_NODE_ABSTRACTION_H_
