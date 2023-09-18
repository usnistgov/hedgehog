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



#ifndef HEDGEHOG_TASK_INPUTS_MANAGEMENT_ABSTRACTION_H
#define HEDGEHOG_TASK_INPUTS_MANAGEMENT_ABSTRACTION_H

#include <mutex>
#include <condition_variable>
#include <ostream>

#include "../base/node/node_abstraction.h"
#include "../base/execute_abstraction.h"
#include "../base/input_output/slot_abstraction.h"

#include "../../../tools/concepts.h"
#include "../../implementors/concrete_implementor/default_slot.h"
#include "../../implementors/concrete_implementor/queue_receiver.h"
#include "../../implementors/concrete_implementor/default_execute.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Input management abstraction for the task
/// @tparam Inputs Types of input data
template<class ...Inputs>
class TaskInputsManagementAbstraction :
    public SlotAbstraction,
    public ReceiverAbstraction<Inputs> ...,
    public ExecuteAbstraction<Inputs> ... {
 private:
  TaskNodeAbstraction *const coreTask_ = nullptr; ///< Accessor to the core task

 protected:
  std::map<std::string, std::chrono::nanoseconds>
      executionDurationPerInput_, ///< Node execution per input
  dequeueExecutionDurationPerInput_; ///< Node dequeue + execution per input

  std::map<std::string, std::size_t> nbElementsPerInput_; ///< Number of elements received per input

 public:
  using inputs_t = std::tuple<Inputs...>; ///< Accessor to the input types

  /// @brief Constructor using a node task and its core
  /// @tparam NodeType Type of the node
  /// @param nodeTask Node instance
  /// @param coreTask Core node instance
  template<class NodeType>
  explicit TaskInputsManagementAbstraction(NodeType *const nodeTask, TaskNodeAbstraction *const coreTask)
      : SlotAbstraction(std::make_shared<implementor::DefaultSlot>()),
        ReceiverAbstraction<Inputs>(
            std::make_shared<implementor::QueueReceiver<Inputs>>(), SlotAbstraction::mutex()
        )...,
      ExecuteAbstraction<Inputs>(std::make_shared<implementor::DefaultExecute<Inputs>>(
  static_cast<behavior::Execute<Inputs>*>(nodeTask)))...,
  coreTask_(coreTask) {
      (initializeMapsExecutionDurationPerInput<Inputs>(), ...);
  }

  /// @brief Constructor using a node and the concrete implementor
  /// @tparam ConcreteMultiReceivers Concrete implementation of multi receivers abstraction
  /// @tparam ConcreteMultiExecutes Concrete implementation of execute abstraction
  /// @param coreTask Core task instance
  /// @param concreteSlot Concrete slot implementation
  /// @param concreteMultiReceivers Concrete multi receivers implementation
  /// @param concreteMultiExecutes Concrete multi executes implementation
  template<
      hh::tool::ConcreteMultiReceiverImplementation<Inputs...> ConcreteMultiReceivers,
      hh::tool::ConcreteMultiExecuteImplementation<Inputs...> ConcreteMultiExecutes
  >
  explicit TaskInputsManagementAbstraction(
      TaskNodeAbstraction *const coreTask,
      std::shared_ptr<implementor::ImplementorSlot> concreteSlot,
      std::shared_ptr<ConcreteMultiReceivers> concreteMultiReceivers,
      std::shared_ptr<ConcreteMultiExecutes> concreteMultiExecutes) :
      SlotAbstraction(concreteSlot),
      ReceiverAbstraction<Inputs>(concreteMultiReceivers, SlotAbstraction::mutex())...,
      ExecuteAbstraction<Inputs>(concreteMultiExecutes)
  ...,
  coreTask_(coreTask) {
      (initializeMapsExecutionDurationPerInput<Inputs>(), ...);
  }

  /// @brief Default destructor
  ~TaskInputsManagementAbstraction() override = default;

  /// @brief Test if the receivers are empty for the task
  /// @return True if the receivers are empty, else false
  [[nodiscard]] bool receiversEmpty() const { return (ReceiverAbstraction<Inputs>::empty() && ...); }

 protected:

  /// @brief Accessor to the total number of elements received for all input types
  /// @return The total number of elements received for all input types
  [[nodiscard]] size_t totalNumberElementsReceived() const {
    return (ReceiverAbstraction<Inputs>::numberElementsReceived() + ...);
  }

  /// @brief Access all the task receivers to process an element
  void operateReceivers() { (this->operateReceiver<Inputs>(), ...); }

  /// @brief Call for all types the user-defined execute method with nullptr as data
  void callAllExecuteWithNullptr() { (callExecuteForATypeWithNullptr<Inputs>(), ...); }

  /// @brief Wake up implementation (notify one node waiting on the condition variable)
  void wakeUp() final { this->slotConditionVariable()->notify_one(); }

  /// @brief Copy the task core inner structure to this
  /// @param copyableCore Task core to copy from
  void copyInnerStructure(TaskInputsManagementAbstraction<Inputs...> *copyableCore) {
    (ReceiverAbstraction<Inputs>::copyInnerStructure(copyableCore), ...);
    SlotAbstraction::copyInnerStructure(copyableCore);
  }

  /// @brief Add edge information to the printer
  /// @param printer Printer visitor gathering edge information
  void printEdgesInformation(Printer *printer) {
    (ReceiverAbstraction<Inputs>::printEdgeInformation(printer), ...);
  }

 private:
  /// @brief Access the ReceiverAbstraction of the type InputDataType to process an element
  /// @tparam InputDataType Type of input data
  template<tool::ContainsConcept<Inputs...> InputDataType>
  void operateReceiver() {
    static std::string const typeStr = hh::tool::typeToStr<InputDataType>();
    std::chrono::time_point<std::chrono::system_clock>
        start = std::chrono::system_clock::now(),
        finish;
    this->lockSlotMutex();
    auto typedReceiver = static_cast<ReceiverAbstraction<InputDataType> *>(this);
    if (!typedReceiver->empty()) {
      std::shared_ptr<InputDataType> data = typedReceiver->getInputData();
      coreTask_->incrementNumberReceivedElements();
      this->unlockSlotMutex();
      callExecuteForAType(data);
    } else { this->unlockSlotMutex(); }
    finish = std::chrono::system_clock::now();
    incrementDequeueExecutionPerInput<InputDataType>(finish - start);
    coreTask_->incrementDequeueExecutionDuration(finish - start);
  }

  /// @brief Call execute function with nullptr
  /// @tparam InputDataType input data type
  template<tool::ContainsConcept<Inputs...> InputDataType>
  void callExecuteForATypeWithNullptr() {
    static std::string const typeStr = hh::tool::typeToStr<InputDataType>();
    std::chrono::time_point<std::chrono::system_clock>
        start = std::chrono::system_clock::now(),
        finish;
    callExecuteForAType<InputDataType>(nullptr);
    finish = std::chrono::system_clock::now();
    incrementDequeueExecutionPerInput<InputDataType>(finish - start);
    coreTask_->incrementDequeueExecutionDuration(finish - start);
  }

  /// @brief Call Execute with a type and log execution time for a type
  /// @tparam InputDataType Type of input data
  /// @param inputDataType Input data type
  template<class InputDataType>
  void callExecuteForAType(std::shared_ptr<InputDataType> inputDataType) {
    static std::string typeStr = hh::tool::typeToStr<InputDataType>();
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    coreTask_->nvtxProfiler()->startRangeExecuting();
    ExecuteAbstraction<InputDataType>::callExecute(inputDataType);
    coreTask_->nvtxProfiler()->endRangeExecuting();
    std::chrono::time_point<std::chrono::system_clock> finish = std::chrono::system_clock::now();
    this->template incrementExecutionPerInput<InputDataType>(finish - start);
    coreTask_->incrementExecutionDuration(finish - start);
    this->nbElementsPerInput_.at(typeStr)++;
  }

 private:
  /// @brief Initialize the data structures for the execution duration per input
  /// @tparam Input Input type to initialize
  template<class Input>
  void initializeMapsExecutionDurationPerInput() {
    static std::string typeStr = hh::tool::typeToStr<Input>();
    this->nbElementsPerInput_[typeStr] = {};
    this->executionDurationPerInput_[typeStr] = {};
    this->dequeueExecutionDurationPerInput_[typeStr] = {};
  }

  /// @brief Increment the execution duration per input
  /// @tparam Input Input type to initialize
  /// @param exec Execution time to add
  template<class Input>
  void incrementExecutionPerInput(std::chrono::nanoseconds const &exec) {
    this->executionDurationPerInput_.at(hh::tool::typeToStr<Input>()) += exec;
  }

  /// @brief Increment the dequeue + execution duration per input
  /// @tparam Input Input type to initialize
  /// @param exec Execution time to add
  template<class Input>
  void incrementDequeueExecutionPerInput(std::chrono::nanoseconds const &exec) {
    static std::string const InputStr = hh::tool::typeToStr<Input>();
    this->dequeueExecutionDurationPerInput_.at(InputStr) += exec;
  }

};
}
}
}
#endif //HEDGEHOG_TASK_INPUTS_MANAGEMENT_ABSTRACTION_H
