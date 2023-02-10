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



#ifndef HEDGEHOG_CORE_GRAPH_H
#define HEDGEHOG_CORE_GRAPH_H

#include <map>
#include <ostream>

#include "../abstractions/base/clonable_abstraction.h"
#include "../abstractions/base/cleanable_abstraction.h"
#include "../abstractions/base/node/graph_node_abstraction.h"

#include "../abstractions/node/graph_inputs_management_abstraction.h"
#include "../abstractions/node/graph_outputs_management_abstraction.h"

#include "../abstractions/base/node/execution_pipeline_node_abstraction.h"

#include "../../tools/traits.h"
#include "../../tools/meta_functions.h"
#include "../../api/graph/scheduler.h"
#include "../../api/printer/printer.h"
#include "../../api/graph/default_scheduler.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration Graph
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class Graph;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog core namespace
namespace core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration CoreExecutionPipeline
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class CoreExecutionPipeline;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Type alias for an GraphInputsManagementAbstraction from the list of template parameters
template<size_t Separator, class...AllTypes>
using GIM = tool::GraphInputsManagementAbstractionTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>;

/// @brief Type alias for an GraphOutputsManagementAbstraction from the list of template parameters
template<size_t Separator, class...AllTypes>
using GOM = tool::GraphOutputsManagementAbstractionTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>;

/// @brief Graph core
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class CoreGraph :
    public abstraction::GraphNodeAbstraction,
    public abstraction::CleanableAbstraction,
    public abstraction::ClonableAbstraction,
    public GIM<Separator, AllTypes...>,
    public GOM<Separator, AllTypes...> {
 private:
  /// Declare CoreExecutionPipeline as friend
  friend CoreExecutionPipeline<Separator, AllTypes...>;
  std::unique_ptr<Scheduler> const
      scheduler_ = nullptr; ///< Scheduler used by the graph

  hh::Graph<Separator, AllTypes...> *const graph_; ///< Graph attached to this core
 public:
  /// @brief Core Graph constructor using the name and the scheduler
  /// @param name Graph name
  /// @param scheduler Graph scheduler
  /// @param graph Graph node attached to this core
  explicit CoreGraph(std::string const &name,
                     std::unique_ptr<Scheduler> scheduler,
                     hh::Graph<Separator, AllTypes...> *graph) :
      GraphNodeAbstraction(name), scheduler_(std::move(scheduler)), graph_(graph) {
    this->source()->registerNode(this);
    this->sink()->registerNode(this);
  }

  /// @brief Copy constructor using an added correspondence map
  /// @param rhs CoreGraph to copy
  /// @param correspondenceMap Correspondence map
  CoreGraph(CoreGraph const &rhs, std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &correspondenceMap)
      : GraphNodeAbstraction(rhs.name()),
        CleanableAbstraction(),
        scheduler_(rhs.scheduler_->create()), graph_(rhs.graph_) {
    this->source()->registerNode(this);
    this->sink()->registerNode(this);
    duplicateInsideNodes(rhs, correspondenceMap);
  }

  /// @brief Default destructor
  ~CoreGraph() override = default;

  /// @brief Connect a core's node as input of the graph for all compatible types
  /// @tparam CoreInputTypes Input node types
  /// @tparam InputCore Type of the core to set as input
  /// @param core Core to set as input
  /// @throw std::runtime if the core is malformed (missing inheritance to SlotAbstraction)
  template<
      class CoreInputTypes,
      tool::CompatibleInputCore<CoreInputTypes, typename GIM<Separator, AllTypes...>::inputs_t> InputCore
  >
  void setInputForAllCommonTypes(InputCore *const core) {
    testRegistered(__FUNCTION__);
    using Input_t = tool::Intersect_t<CoreInputTypes, typename GIM<Separator, AllTypes...>::inputs_t>;
    using Indices = std::make_index_sequence<std::tuple_size_v<Input_t>>;
    registerNodeInsideGraph(core);
    if (dynamic_cast<abstraction::SlotAbstraction *>(core) == nullptr) {
      std::ostringstream oss;
      oss << "The receiver node " << core->name()
          << "has a malformed core (missing inheritance to SlotAbstraction)";
      throw (std::runtime_error(oss.str()));
    }

    callAddInputNodeToGraph<Input_t>(core, Indices{});
  }

  /// @brief Connect a core's node as input of the graph for an compatible input type
  /// @tparam InputType Input node type
  /// @tparam CoreInputTypes Input node types
  /// @tparam InputCore Type of the core to set as input
  /// @param core Core to set as input
  /// @throw std::runtime if the core is malformed (missing inheritance to SlotAbstraction)
  template<
      class InputType,
      class CoreInputTypes,
      tool::CompatibleInputCoreForAType<
          InputType,
          CoreInputTypes,
          typename GIM<Separator, AllTypes...>::inputs_t> InputCore>
  void setInputForACommonType(InputCore *const core) {
    testRegistered(__FUNCTION__);
    using input_t = tool::Intersect_t<CoreInputTypes, typename GIM<Separator, AllTypes...>::inputs_t>;
    using Indices = std::make_index_sequence<std::tuple_size_v<input_t>>;

    testAbstractReceivers<input_t>(core, Indices{});

    registerNodeInsideGraph(core);
    this->template addInputNodeToGraph<InputType>(core);
  }

  /// @brief Connect a core's node as output of the graph for all compatible types
  /// @tparam CoreOutputTypes Output node types
  /// @tparam OutputCore Type of the core to set as output
  /// @param core Core to set as output
  template<
      class CoreOutputTypes,
      tool::CompatibleOutputCore<CoreOutputTypes, typename GOM<Separator, AllTypes...>::outputs_t> OutputCore>
  void setOutputForAllCommonTypes(OutputCore *const core) {
    testRegistered(__FUNCTION__);
    using CommonTypes = tool::Intersect_t<CoreOutputTypes, typename GOM<Separator, AllTypes...>::outputs_t>;
    using Indices = std::make_index_sequence<std::tuple_size_v<CommonTypes>>;
    Indices indices{};
    testAbstractSenders<CoreOutputTypes>(core, indices);
    registerNodeInsideGraph(core);
    callAddOutputNodeToGraph<CommonTypes>(core, indices);
  }

  /// @brief Connect a core's node as output of the graph for a compatible type
  /// @tparam OutputType Output node type
  /// @tparam CoreOutputTypes Output node types
  /// @tparam OutputCore Type of the core to set as output
  /// @param core Core to set as output
  template<class OutputType,
      class CoreOutputTypes,
      tool::CompatibleOutputCoreForAType<
          OutputType, CoreOutputTypes, typename GOM<Separator, AllTypes...
          >::outputs_t> OutputCore>
  void setOutputForACommonType(OutputCore *const core) {
    testRegistered(__FUNCTION__);

    using OutputType_t = std::tuple<OutputType>;
    using IndicesCommonTypes = std::make_index_sequence<std::tuple_size_v<OutputType_t>>;
    IndicesCommonTypes indices{};
    testAbstractSenders<OutputType_t>(core, indices);

    registerNodeInsideGraph(core);
    this->template addOutputNodeToGraph<OutputType>(core);
  }

  /// @brief Connect two nodes together for all common types
  /// @tparam OutputTypesSenderTuple Sender output types
  /// @tparam InputTypeReceiverTuple Receiver Input types
  /// @param senderCore Type of the sender
  /// @param receiverCore Type of the receiver
  template<class OutputTypesSenderTuple, class InputTypeReceiverTuple>
  void addEdgeForAllCommonTypes(NodeAbstraction *const senderCore, NodeAbstraction *const receiverCore) {
    testRegistered(__FUNCTION__);

    using CommonTypes = tool::Intersect_t<OutputTypesSenderTuple, InputTypeReceiverTuple>;

    using IndicesCommonTypes = std::make_index_sequence<std::tuple_size_v<CommonTypes>>;
    IndicesCommonTypes indices{};

    static_assert(
        std::tuple_size_v<CommonTypes> != 0,
        "When adding an edge between two nodes, they should share at least one type."
    );

    testAbstractReceivers<CommonTypes>(receiverCore, indices);
    testAbstractSenders<CommonTypes>(senderCore, indices);

    registerNodeInsideGraph(senderCore);
    registerNodeInsideGraph(receiverCore);

    connectNotifierToSlot(senderCore, receiverCore);
    drawEdges<CommonTypes>(senderCore, receiverCore, indices);
  }

  /// @brief Connect two nodes together for a common type
  /// @tparam CommonType Type used for the edge
  /// @tparam OutputTypesSenderTuple Sender output types
  /// @tparam InputTypeReceiverTuple Receiver Input types
  /// @param senderCore Type of the sender
  /// @param receiverCore Type of the receiver
  template<class CommonType,
      class OutputTypesSenderTuple,
      class InputTypeReceiverTuple>
  void addEdgeForACommonType(NodeAbstraction *const senderCore, NodeAbstraction *const receiverCore) {
    testRegistered(__FUNCTION__);
    using CommonType_t = std::tuple<CommonType>;
    using Indices = std::make_index_sequence<std::tuple_size_v<CommonType_t>>;
    Indices indices{};

    using CommonTypes = tool::Intersect_t<OutputTypesSenderTuple, InputTypeReceiverTuple>;
    static_assert(
        std::tuple_size_v<CommonTypes> != 0,
        "When adding an edge between two nodes, they should share at least one type.");
    static_assert(
        tool::isContainedInTuple_v<CommonType, CommonTypes>,
        "The type should be part of the types shared between the two cores that you want to connect."
    );

    testAbstractReceivers<CommonType_t>(receiverCore, indices);
    testAbstractSenders<CommonType_t>(senderCore, indices);

    registerNodeInsideGraph(senderCore);
    registerNodeInsideGraph(receiverCore);

    connectNotifierToSlot(senderCore, receiverCore);
    drawEdge<CommonType>(senderCore, receiverCore);
  }

  /// @brief Execute the graph
  /// @details Create the needed groups and use the scheduler to create the threads
  /// @param waitForInitialization Wait for internal nodes to be initialized flags
  void executeGraph(bool waitForInitialization) {
    testRegistered(__FUNCTION__);
    this->graphStatus_ = Status::EXEC;
    createInnerGroupsAndLaunchThreads(waitForInitialization);
    auto finishCreationTime = std::chrono::system_clock::now();
    this->graphConstructionDuration(finishCreationTime - this->graphStartCreation());
    this->startExecutionTimeStamp(std::chrono::system_clock::now());
  }

  /// @brief Indicate to the graph that no more input will be sent, trigger the termination of the graph
  void finishPushingData() {
    this->graphStatus_ = Status::TERM;
    testRegistered(__FUNCTION__);
    this->terminateSource();
  }

  /// @brief Wait for the graph to terminate
  /// @details Wait for the inner threads to join
  void waitForTermination() {
    joinThreads();
    std::chrono::time_point<std::chrono::system_clock>
        endExecutionTimeStamp = std::chrono::system_clock::now();
    this->incrementExecutionDuration(endExecutionTimeStamp - this->startExecutionTimeStamp());
  }

  /// @brief Broadcast an input data to all inputs nodes
  /// @tparam CompatibleInputType_t Type of the input data
  /// @param data Data to broadcast
  template<tool::MatchInputTypeConcept<tool::Inputs<Separator, AllTypes...>> CompatibleInputType_t>
  void broadcastAndNotifyAllInputNodes(std::shared_ptr<CompatibleInputType_t> &data) {

    testRegistered(__FUNCTION__);
    this->sendInputDataToSource(data);
  }

  /// @brief Visit the graph
  /// @param printer Printer gathering node information
  void visit(Printer *printer) override {
    if (printer->registerNode(this)) {
      printer->printGraphHeader(this);
      this->printSource(printer);
      this->printSink(printer);

      for (auto insideNodeGroups : *(this->insideNodesAndGroups_)) {
        if (insideNodeGroups.second.size() == 0) {
          if(auto printableNode = dynamic_cast<abstraction::PrintableAbstraction*>(insideNodeGroups.first))
            printableNode->visit(printer);
        } else {
          printer->printGroup(insideNodeGroups.first, insideNodeGroups.second);
        }
      }
      printer->printGraphFooter(this);
    }
  }

  /// @brief Clean the graph
  /// @details Call clean recursively all nodes that can be cleaned
  void cleanGraph() {
    testRegistered(__FUNCTION__);
    if (this->graphStatus_ != Status::TERM) {
      std::unordered_set<behavior::Cleanable *> cleanableSet;
      for (auto insideNodeGroups : *(this->insideNodesAndGroups_)) {
        if (auto cleanableReresentative =
            dynamic_cast<hh::core::abstraction::CleanableAbstraction *>(insideNodeGroups.first)) {
          cleanableReresentative->gatherCleanable(cleanableSet);
          for (auto copy : insideNodeGroups.second) {
            dynamic_cast<hh::core::abstraction::CleanableAbstraction *>(copy)->gatherCleanable(cleanableSet);
          }
        }
      }
      for (auto &cleanableNode : cleanableSet) {
        cleanableNode->clean();
      }
    } else {
      throw std::runtime_error("It is not possible to clean a graph while it is terminating.");
    }
  }

  /// @brief Node ids [nodeId, nodeGroupId] accessor
  /// @return  Node ids [nodeId, nodeGroupId]
  /// @throw std::runtime_error if the node is ill-formed
  [[nodiscard]] std::vector<std::pair<std::string const, std::string const>> ids() const override {
    std::vector<std::pair<std::string const, std::string const>> idInputNodesAndGroups;
    for (auto inputNodes : abstraction::SlotAbstraction::connectedNotifiers()) {
      for (auto inputNotifier : inputNodes->notifiers()) {
        if (auto inputNode = dynamic_cast<NodeAbstraction *>(inputNotifier)) {
          for (auto const &id : inputNode->ids()) { idInputNodesAndGroups.push_back(id); }
        } else {
          throw std::runtime_error("An input node should derive from NodeAbstraction.");
        }
      }
    }
    return idInputNodesAndGroups;
  }

  /// @brief Node accessor
  /// @return Node attached to this core
  [[nodiscard]] behavior::Node *node() const override { return graph_; }

 private:
  /// @brief Wait for the threads to join
  void joinThreads() override { this->scheduler_->joinAll(); }

  /// @brief Set the graph as inside of another graph
  void setInside() override {
    this->graphStatus_ = Status::INS;
    this->disconnectSource();
    this->disconnectSink();
  }

  /// @brief Register a node inside of a graph
  /// @param core Core to register
  /// @throw std::runtime_error Try to modify a graph that has been set as inside of another graph, or set the graph as
  /// inside of itself
  void registerNodeInsideGraph(NodeAbstraction *const core) override {
    if (core->isRegistered()) {
      if (core->belongingGraph() != this) {
        std::ostringstream oss;
        oss << "You can not modify a graph that is connected inside another graph.";
        throw (std::runtime_error(oss.str()));
      }
    } else {
      if (core != this) {
        core->registerNode(this);
        this->insideNodesAndGroups_->insert({core, {}});
      } else {
        std::ostringstream oss;
        oss << "You can not add a graph to itself.";
        throw (std::runtime_error(oss.str()));
      }
    }
  }

  /// @brief Call addInputNodeToGraph for all type-elements of a tuple
  /// @tparam InputTypes Tuple of input types
  /// @tparam Indices Tuple indices
  /// @param core Core to set as input of the graph
  template<class InputTypes, size_t ...Indices>
  void callAddInputNodeToGraph(NodeAbstraction *const core, std::index_sequence<Indices...>) {
    (this->template addInputNodeToGraph<std::tuple_element_t<Indices, InputTypes>>(core), ...);
  }

  /// @brief Call addOutputNodeToGraph for all type-elements of a tuple
  /// @tparam InputTypes Tuple of output types
  /// @tparam Indices Tuple indices
  /// @param core Core to set as output of the graph
  template<class OutputTypes, size_t ...Indices>
  void callAddOutputNodeToGraph(NodeAbstraction *const core, std::index_sequence<Indices...>) {
    (this->template addOutputNodeToGraph<std::tuple_element_t<Indices, OutputTypes>>(core), ...);
  }

  /// @brief Connect a notifier to a slot, used when connecting two nodes
  /// @param senderCore Notifier to connect
  /// @param receiverCore Slot to connect
  /// @throw std::runtime_error The nodes are ill-formed
  void connectNotifierToSlot(NodeAbstraction *senderCore, NodeAbstraction *receiverCore) {
    auto notifierAbstraction = dynamic_cast<abstraction::NotifierAbstraction *>(senderCore);
    auto slotAbstraction = dynamic_cast<abstraction::SlotAbstraction *>(receiverCore);

    if (notifierAbstraction == nullptr) {
      std::ostringstream oss;
      oss << "The sender node " << senderCore->name()
          << "has a malformed core (missing inheritance to NotifierAbstraction)";
      throw (std::runtime_error(oss.str()));
    }

    if (slotAbstraction == nullptr) {
      std::ostringstream oss;
      oss << "The receiver node " << receiverCore->name()
          << "has a malformed core (missing inheritance to SlotAbstraction)";
      throw (std::runtime_error(oss.str()));
    }

    for (auto notifier : notifierAbstraction->notifiers()) {
      for (auto slot : slotAbstraction->slots()) {
        notifier->addSlot(slot);
        slot->addNotifier(notifier);
      }
    }
  }

  /// @brief Do the actual typed connection between a sender and receiver
  /// @tparam CommonType Type to do the connection
  /// @param senderCore  Sender core
  /// @param receiverCore Receiver core
  template<class CommonType>
  void drawEdge(NodeAbstraction *const senderCore, NodeAbstraction *const receiverCore) {
    auto senderAbstraction = dynamic_cast<abstraction::SenderAbstraction<CommonType> *>(senderCore);
    auto receiverAbstraction = dynamic_cast<abstraction::ReceiverAbstraction<CommonType> *>(receiverCore);

    assert(senderAbstraction != nullptr && receiverAbstraction != nullptr);

    for (auto sender : senderAbstraction->senders()) {
      for (auto receiver : receiverAbstraction->receivers()) {
        sender->addReceiver(receiver);
        receiver->addSender(sender);
      }
    }
  }

  /// @brief Do the actual typed connections between a sender and receiver
  /// @tparam CommonTypes Tuple with common types
  /// @tparam Indexes Indexes of CommonTypes tuple
  /// @param senderCore  Sender core
  /// @param receiverCore Receiver core
  template<class CommonTypes, size_t ...Indexes>
  void drawEdges(
      NodeAbstraction *const senderCore,
      NodeAbstraction *const receiverCore,
      std::index_sequence<Indexes...>) {
    (drawEdge<std::tuple_element_t<Indexes, CommonTypes>>(senderCore, receiverCore), ...);
  }

  /// @brief Test a core if it can receives data of specific types
  /// @tparam TupleInputs Tuple of types to test
  /// @tparam Indices Indices of TupleInputs tuple
  /// @param core Core to test
  /// @throw std::runtime_error if core does not inherit from ReceiverAbstraction
  template<class TupleInputs, size_t... Indices>
  void testAbstractReceivers(NodeAbstraction *const core, std::index_sequence<Indices...>) {
    // The core inherits from ReceiverAbstraction for every input common types between the node and the graph
    if (((dynamic_cast<abstraction::ReceiverAbstraction<std::tuple_element_t<Indices, TupleInputs>> *>(core)
        == nullptr) || ... )) {
      std::ostringstream oss;
      oss << "The node " << core->name()
          << " does not have a well defined core (missing inheritance from ReceiverAbstraction).";
      throw std::runtime_error(oss.str());
    }
  }

  /// @brief Test a core if it can send data of specific types
  /// @tparam TupleOutputs Tuple of types to test
  /// @tparam Indices Indices of TupleOutputs tuple
  /// @param core Core to test
  /// @throw std::runtime_error if core does not inherit from SenderAbstraction
  template<class TupleOutputs, size_t... Indices>
  void testAbstractSenders(NodeAbstraction *const core, std::index_sequence<Indices...>) {
    // The core inherits from ReceiverAbstraction for every input common types between the node and the graph
    if (((dynamic_cast<abstraction::SenderAbstraction<std::tuple_element_t<Indices, TupleOutputs>> *>(core)
        == nullptr) || ... )) {
      std::ostringstream oss;
      oss << "The node " << core->name()
          << " does not have a well defined core (missing inheritance from SenderAbstraction).";
      throw std::runtime_error(oss.str());
    }
  }

  /// @brief Test if the graph has been registered
  /// @param funcName Function name calling the registration test
  void testRegistered(auto const &funcName) {
    if (this->isRegistered()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << funcName;
      throw (std::runtime_error(oss.str()));
    }
  }

  /// @brief Create the inner groups and launch the threads inside of a graph
  /// @param waitForInitialization Wait for internal nodes to be initialized flags
  void createInnerGroupsAndLaunchThreads(bool waitForInitialization) override {
    for (auto const &insideNode : *this->insideNodesAndGroups_) {
      if (auto copyableNode = dynamic_cast<abstraction::AnyGroupableAbstraction *>(insideNode.first)) {
        copyableNode->createGroup(*this->insideNodesAndGroups_);
        for (auto &copy : insideNode.second) {
          copy->registerNode(this);
        }
      } else if (auto graph = dynamic_cast<GraphNodeAbstraction *>(insideNode.first)) {
        graph->createInnerGroupsAndLaunchThreads(waitForInitialization);
      } else if (auto ep = dynamic_cast<abstraction::ExecutionPipelineNodeAbstraction *>(insideNode.first)) {
        ep->launchGraphThreads(waitForInitialization);
      }
    }
    launchThreads(waitForInitialization);
  }

  /// @brief Launch the threads with the scheduler
  /// @param waitForInitialization Wait for internal nodes to be initialized flags
  void launchThreads(bool waitForInitialization) {
    std::set<NodeAbstraction *> nodeAbstractions;
    for (auto &insideNode : *this->insideNodesAndGroups_) {
      nodeAbstractions.insert(insideNode.first);
      std::for_each(
          insideNode.second.cbegin(), insideNode.second.cend(),
          [&nodeAbstractions](auto &nodeAbstraction) { nodeAbstractions.insert(nodeAbstraction); });
    }
    scheduler_->spawnThreads(nodeAbstractions, waitForInitialization);
  }

  /// @brief Gather cleanable nodes
  /// @param cleanableSet Mutable set of cleanable ndoes
  void gatherCleanable(std::unordered_set<hh::behavior::Cleanable *> &cleanableSet) override {
    for (auto insideNodeGroups : *(this->insideNodesAndGroups_)) {
      if (auto cleanableReresentative =
          dynamic_cast<hh::core::abstraction::CleanableAbstraction *>(insideNodeGroups.first)) {
        cleanableReresentative->gatherCleanable(cleanableSet);
        for (auto copy : insideNodeGroups.second) {
          if (auto cleanableCopy = dynamic_cast<hh::core::abstraction::CleanableAbstraction *>(copy)) {
            cleanableCopy->gatherCleanable(cleanableSet);
          } else { throw std::runtime_error("A copy of a cleanable node should be a cleanable node."); }
        }
      }
    }
  }

  /// @brief Clone the current graph
  /// @param correspondenceMap Correspondence map to register clone
  /// @return Clone of the grpah
  std::shared_ptr<abstraction::NodeAbstraction> clone(std::map<NodeAbstraction *,
                                                               std::shared_ptr<NodeAbstraction>> &correspondenceMap) override {
    return std::make_shared<CoreGraph<Separator, AllTypes...>>(*this, correspondenceMap);
  }

  /// @brief Clone all of the inside nodes
  /// @param rhs CoreGraph to clone
  /// @param correspondenceMap Map between nodes and their clones
  void duplicateInsideNodes(CoreGraph const &rhs,
                            std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &correspondenceMap) {
    auto originalInsideNodes = *(rhs.insideNodesAndGroups_);
    std::shared_ptr<NodeAbstraction> nodeClone;

    for (auto &originalNode : originalInsideNodes) {
      if (auto originalAsClonable = dynamic_cast<abstraction::ClonableAbstraction *>(originalNode.first)) {
        if (!correspondenceMap.contains(originalNode.first)) {
          nodeClone = originalAsClonable->clone(correspondenceMap);
          storeClone(nodeClone);
          this->registerNodeInsideGraph(nodeClone.get());
          correspondenceMap.insert({originalNode.first, nodeClone});
        }
      }
    }

    this->duplicateSourceEdges(rhs, correspondenceMap);
    this->duplicateSinkEdges(rhs, correspondenceMap);

    for (auto &originalNode : originalInsideNodes) {
      if (auto originalAsClonable = dynamic_cast<abstraction::ClonableAbstraction *>(originalNode.first)) {
        originalAsClonable->duplicateEdge(correspondenceMap);
      }
    }
  }

  /// @brief Duplicate the graph output's edges
  /// @param mapping Map between nodes and their clones
  void duplicateEdge(std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping) override {
    this->duplicateOutputEdges(mapping);
  }

};
}
}
#endif //HEDGEHOG_CORE_GRAPH_H
