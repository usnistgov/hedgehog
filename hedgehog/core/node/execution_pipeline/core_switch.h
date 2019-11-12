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


#ifndef HEDGEHOG_CORE_SWITCH_H
#define HEDGEHOG_CORE_SWITCH_H

#include <ostream>
#include "../../io/queue/sender/core_queue_sender.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Core switch, determine where to divert data to the graphs
/// @tparam GraphInputs Graph input types
template<class ...GraphInputs>
class CoreSwitch : public CoreQueueSender<GraphInputs> ... {
 public:
  /// @brief Default core switch
  CoreSwitch()
      : CoreNode("switch", NodeType::Switch, 1),
        CoreNotifier("switch", NodeType::Switch, 1),
        CoreQueueNotifier("switch", NodeType::Switch, 1),
        CoreQueueSender<GraphInputs>("switch", NodeType::Switch, 1)... {}

  /// @brief Clone a default core switch
  /// @return A new CoreSwitch
  std::shared_ptr<CoreNode> clone() override { return std::make_shared<CoreSwitch<GraphInputs...>>(); }

  /// @brief Send a user node, not possible for a switch, should only be managed by an execution pipeline, throw an
  /// error in every case
  /// @exception std::runtime_error A switch has not nodes
  /// @return Nothing, throw an error
  behavior::Node *node() override {
    std::ostringstream oss;
    oss << "Internal error, a switch has no nodes: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Visit a switch, do nothing
  void visit([[maybe_unused]]AbstractPrinter *) override {}

  /// @brief Add a slot for every input types
  /// @param slot slot to add
  void addSlot(CoreSlot *slot) override { (CoreQueueSender<GraphInputs>::addSlot(slot), ...); }

  /// @brief Remove a slot from every input types
  /// @param slot CoreSlot to remove
  void removeSlot(CoreSlot *slot) override { (CoreQueueSender<GraphInputs>::removeSlot(slot), ...); }

  /// @brief Notify terminated for all input types
  void notifyAllTerminated() override { (CoreQueueSender<GraphInputs>::notifyAllTerminated(), ...); }

  /// @brief Print extra information for the switch
  /// @return string with extra information
  std::string extraPrintingInformation() override {
    std::ostringstream oss;
    oss << "Switch Info: " << std::endl;
    (printSenderInfo<GraphInputs>(oss), ...);
    return oss.str();
  }

 protected:
  /// @brief Duplicate all the edges from this to its copy duplicateNode
  /// @param duplicateNode Node to connect
  /// @param correspondenceMap Correspondence map from base node to copy
  void duplicateEdge(CoreNode *duplicateNode,
                     std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
    (CoreQueueSender<GraphInputs>::duplicateEdge(duplicateNode, correspondenceMap), ...);
  }

 private:
  /// @brief Stream all sender information into oss
  /// @tparam GraphInput Type of Sender to stream
  /// @param oss std::ostringstream to stream into
  template<class GraphInput>
  void printSenderInfo(std::ostringstream &oss) {
    oss << typeid(GraphInput).name() << std::endl;
    for (auto slot : *(static_cast<CoreQueueSender<GraphInput> *>(this)->slots())) {
      oss << "\t" << slot->id() << " / " << slot->name() << std::endl;
    }
  }

  /// @brief Slots accessors, throw an error, a switch does not have any slots
  /// @exception std::runtime_error A switch does not have any slots
  /// @return Nothing, throw an error
  std::set<CoreSlot *> getSlots() override {
    std::ostringstream oss;
    oss << "Runtime error, A switch does not have any slots: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }
};

}
#endif //HEDGEHOG_CORE_SWITCH_H
