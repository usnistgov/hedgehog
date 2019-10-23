//
// Created by anb22 on 6/6/19.
//

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
