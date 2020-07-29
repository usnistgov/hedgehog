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


#ifndef HEDGEHOG_CORE_GRAPH_H
#define HEDGEHOG_CORE_GRAPH_H

#include <ostream>
#include <vector>
#include <filesystem>

#include "../node/core_node.h"
#include "../io/base/sender/core_notifier.h"
#include "../io/base/receiver/core_multi_receivers.h"
#include "../io/graph/receiver/core_graph_multi_receivers.h"
#include "../io/graph/receiver/core_graph_sink.h"
#include "../io/graph/sender/core_graph_source.h"
#include "../../tools/traits.h"
#include "../../tools/helper.h"
#include "../../api/scheduler/default_scheduler.h"
#include "../../api/scheduler/abstract_scheduler.h"
#include "../../api/printer/dot_printer.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration
/// @tparam GraphOutput Graph output type
/// @tparam GraphInputs Graph input types
template<class GraphOutput, class ...GraphInputs>
class Graph;
#endif // DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog core namespace
namespace core {

/// @brief Core associated to the Graph
/// @details Internal representation of a graph in the hedgehog library. Hold nodes that's inside.
/// If the graph is an outer graph, i.e. not inside another graph, it will have two special nodes: source and sink.
/// The source is used to register the input nodes and to send data to them. The sink is used to gather data out of the
/// output nodes, and make them available outside of the graph. When the graph is registered into another graph, the
/// source and the sink are removed and the proper connections are made to the outer graph's nodes.
/// A default scheduler is set to spawn the threads and join them at the end, and uses the OS to schedule threads.
/// @tparam GraphOutput Graph output type
/// @tparam GraphInputs Graph input types
template<class GraphOutput, class ...GraphInputs>
class CoreGraph : public CoreSender<GraphOutput>, public CoreGraphMultiReceivers<GraphInputs...> {
 private:
  Graph<GraphOutput, GraphInputs...> *graph_ = nullptr; ///< User graph
  std::unique_ptr<std::set<CoreNode *>> inputsCoreNodes_ = nullptr; ///< Input node's core
  std::unique_ptr<std::set<CoreSender<GraphOutput> *>> outputCoreNodes_ = nullptr; ///< Output node's core
  std::unique_ptr<AbstractScheduler> scheduler_ = nullptr; ///< Scheduler
  std::shared_ptr<CoreGraphSource<GraphInputs...>> source_ = nullptr; ///< Outer graph's source
  std::shared_ptr<CoreGraphSink<GraphOutput>> sink_ = nullptr; ///< Inner graph's source
  int graphId_ = 0; ///< Graph Id
  int deviceId_ = 0; ///< Device Id used for computation on devices

 public:
  /// @brief CoreGraph constructor
  /// @param graph User graph
  /// @param type Graph's type
  /// @param name Graph's name
  /// @param scheduler Graph's scheduler, by default a DefaultScheduler
  CoreGraph(Graph<GraphOutput, GraphInputs...> *graph, NodeType const type, std::string_view const &name,
            std::unique_ptr<AbstractScheduler> scheduler = std::make_unique<DefaultScheduler>()) :
      CoreNode(name, type, 1),
      CoreNotifier(name, type, 1),
      CoreSlot(name, type, 1),
      CoreReceiver<GraphInputs>(name, type, 1)...,
      CoreSender<GraphOutput>(name, type, 1),
  CoreGraphMultiReceivers<GraphInputs...>(name, type, 1){
    HLOG_SELF(0, "Creating CoreGraph with coreGraph: " << graph << " type: " << (int) type << " and name: " << name)
    if (!scheduler) {
      std::ostringstream oss;
      oss << "Internal error, the graph's scheduler is null, please instantiate an AbstractScheduler.";
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
    this->graph_ = graph;
    this->inputsCoreNodes_ = std::make_unique<std::set<CoreNode *>>();
    this->outputCoreNodes_ = std::make_unique<std::set<CoreSender < GraphOutput> *>>
    ();
    this->source_ = std::make_shared<CoreGraphSource<GraphInputs...>>();
    this->sink_ = std::make_shared<CoreGraphSink<GraphOutput>>();
    this->scheduler_ = std::move(scheduler);
    this->source_->belongingNode(this);
    this->sink_->belongingNode(this);
  }

  /// @brief Core graph copy constructor
  /// @param rhs CoreGraph to copy
  CoreGraph(CoreGraph<GraphOutput, GraphInputs...> const &rhs) :
      CoreNode(rhs.name(), rhs.type(), 1),
      CoreNotifier(rhs.name(), rhs.type(), 1),
      CoreSlot(rhs.name(), rhs.type(), 1),
      CoreReceiver<GraphInputs>(rhs.name(), rhs.type(), 1)...,
      CoreSender<GraphOutput>(rhs.name(), rhs.type(),1),
      CoreGraphMultiReceivers<GraphInputs...>(rhs.name(), rhs.type(),1){
    this->inputsCoreNodes_ = std::make_unique<std::set<CoreNode *>>();
    this->outputCoreNodes_ = std::make_unique<std::set<CoreSender < GraphOutput> *>>
    ();
    this->scheduler_ = rhs.scheduler_->create();
    this->source_ = std::make_shared<CoreGraphSource<GraphInputs...>>();
    this->sink_ = std::make_shared<CoreGraphSink<GraphOutput>>();

    this->source_->belongingNode(this);
    this->sink_->belongingNode(this);

    duplicateInsideNodes(rhs);

    this->hasBeenRegistered(true);
    this->belongingNode(rhs.belongingNode());

    this->graph_ = rhs.graph_;
  }

  /// @brief Clone a core graph calling the graph copy constructor
  /// @return A cloned of this
  std::shared_ptr<CoreNode> clone() override { return std::make_shared<CoreGraph<GraphOutput, GraphInputs...>>(*this); }

  /// @brief Graph's core default destructor
  ~CoreGraph() override {HLOG_SELF(0, "Destructing CoreGraph")}

  /// @brief User graph accessor
  /// @return User graph node
  behavior::Node *node() override { return graph_; }

  /// @brief Device id accessor
  /// @return Device id
  int deviceId() override { return this->deviceId_; }

  /// @brief Graph id accessor
  /// @return Graph id
  int graphId() override { return this->graphId_; }

  /// @brief Input node's cores accessor
  /// @return Input node's cores
  [[nodiscard]] std::unique_ptr<std::set<CoreNode *>> const &inputsCoreNodes() const { return inputsCoreNodes_; }

  /// @brief  Output node's CoreSender accessor
  /// @return Output node's CoreSender
  [[nodiscard]] std::unique_ptr<std::set<CoreSender < GraphOutput> *>> const &outputCoreNodes() const {
    return outputCoreNodes_;
  }

  /// @brief Source accessor
  /// @return Graph's source
  std::shared_ptr<CoreGraphSource<GraphInputs...>> const &source() const { return source_; }

  /// @brief Graph id setter
  /// @param graphId Graph id
  void graphId(size_t graphId) { graphId_ = graphId; }

  /// @brief Device id setter
  /// @param deviceId Graph's device id to set
  void deviceId(int deviceId) override { this->deviceId_ = deviceId; }

  /// @brief Compute the maximum execution time for the graph's inside nodes
  /// @return The maximum execution time for the graph's inside nodes
  [[nodiscard]] std::chrono::duration<double, std::micro> maxExecutionTime() const override {
    std::chrono::duration<double, std::micro>
        ret = std::chrono::duration<double, std::micro>::min(),
        temp{};
    std::shared_ptr<CoreNode> core;
    for (auto const &it : *(this->insideNodes())) {
      core = it.second;
      switch (core->type()) {
        case NodeType::Task:
        case NodeType::Graph:
        case NodeType::StateManager:
        case NodeType::ExecutionPipeline:
          temp = core->maxExecutionTime();
          if (temp > ret) ret = temp;
          break;
        default:
          break;
      }
    }
    return ret;
  }

  /// @brief Compute the minimum execution time for the graph's inside nodes
  /// @return The minimum execution time for the graph's inside nodes
  [[nodiscard]] std::chrono::duration<double, std::micro> minExecutionTime() const override {
    std::chrono::duration<double, std::micro> ret = std::chrono::duration<double, std::micro>::max(), temp{};
    std::shared_ptr<CoreNode> core;
    for (auto const &it : *(this->insideNodes())) {
      core = it.second;
      switch (core->type()) {
        case NodeType::Task:
        case NodeType::Graph:
        case NodeType::StateManager:
        case NodeType::ExecutionPipeline:
          temp = core->minExecutionTime();
          if (temp < ret) ret = temp;
          break;
        default:break;
      }
    }
    return ret;
  }

  /// @brief Compute the maximum wait time for the graph's inside nodes
  /// @return The maximum wait time for the graph's inside nodes
  [[nodiscard]] std::chrono::duration<double, std::micro> maxWaitTime() const override {
    std::chrono::duration<double, std::micro> ret = std::chrono::duration<double, std::micro>::min(), temp{};
    std::shared_ptr<CoreNode> core;
    for (auto const &it : *(this->insideNodes())) {
      core = it.second;
      switch (core->type()) {
        case NodeType::Task:
        case NodeType::Graph:
        case NodeType::StateManager:
        case NodeType::ExecutionPipeline:
          temp = core->maxWaitTime();
          if (temp > ret) ret = temp;
          break;
        default:break;
      }
    }
    return ret;
  }

  /// @brief Compute the minimum wait time for the graph's inside nodes
  /// @return The minimum wait time for the graph's inside nodes
  [[nodiscard]] std::chrono::duration<double, std::micro> minWaitTime() const override {
    std::chrono::duration<double, std::micro> ret = std::chrono::duration<double, std::micro>::max(), temp{};
    std::shared_ptr<CoreNode> core;
    for (auto const &it : *(this->insideNodes())) {
      core = it.second;
      switch (core->type()) {
        case NodeType::Task:
        case NodeType::Graph:
        case NodeType::StateManager:
        case NodeType::ExecutionPipeline:
          temp = core->minWaitTime();
          if (temp < ret) ret = temp;
          break;
        default:break;
      }
    }
    return ret;
  }

  /// @brief Add a directed edge from a compatible "from" node to "to" node.
  /// @tparam UserDefinedSender Sender type that should derive from Sender
  /// @tparam UserDefinedMultiReceiver Receiver type that should derive from MultiReceivers
  /// @tparam Output Sender output type
  /// @tparam Inputs Tuple with MultiReceivers input types
  /// @tparam IsSender Defined if UserDefinedSender is derived from Sender
  /// @tparam IsMultiReceiver Defined if UserDefinedMultiReceiver is derived from MultiReceivers
  /// @param from Node that will send the data
  /// @param to Node that will receiver the data
  template<
      class UserDefinedSender, class UserDefinedMultiReceiver,
      class Output = typename UserDefinedSender::output_t,
      class Inputs = typename UserDefinedMultiReceiver::inputs_t,
      class IsSender = typename std::enable_if_t<
          std::is_base_of_v<
              behavior::Sender<Output>, UserDefinedSender
          >
      >,
      class IsMultiReceiver = typename std::enable_if_t<
          std::is_base_of_v<
              typename helper::HelperMultiReceiversType<Inputs>::type, UserDefinedMultiReceiver
          >
      >
  >
  void addEdge(std::shared_ptr<UserDefinedSender> from, std::shared_ptr<UserDefinedMultiReceiver> to) {
    assert(from != nullptr && to != nullptr);
    static_assert(traits::Contains_v<Output, Inputs>, "The given Receiver cannot be linked to this Sender");
    if (this->isInside()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }

    //Get the associated cores
    auto coreSender = dynamic_cast<CoreSender <Output> *>(std::static_pointer_cast<behavior::Node>(from)->core().get());
    auto coreNotifier = dynamic_cast<CoreNotifier *>(coreSender);
    auto coreSlot = dynamic_cast<CoreSlot *>(std::static_pointer_cast<behavior::Node>(to)->core().get());
    auto
        coreReceiver = dynamic_cast<CoreReceiver<Output> *>(std::static_pointer_cast<behavior::Node>(to)->core().get());

    if (from->core().get() == this || to->core().get() == this) {
      std::ostringstream oss;
      oss << "You can not connect a graph to itself: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }

    if (coreSender->hasBeenRegistered()) {
      if (coreSender->belongingNode() != this) {
        std::ostringstream oss;
        oss << "The Sender node should belong to the graph: " << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }

    if (coreReceiver->hasBeenRegistered()) {
      if (coreReceiver->belongingNode() != this) {
        std::ostringstream oss;
        oss << "The Receiver node should belong to the graph: " << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }

    HLOG_SELF(0,
              "Add edge from " << coreSender->name() << "(" << coreSender->id() << ") to " << coreReceiver->name()
                  << "(" << coreReceiver->id()
                  << ")")

    for (auto r : coreReceiver->receivers()) { coreSender->addReceiver(r); }
    for (CoreSlot *slot : coreSlot->getSlots()) { coreNotifier->addSlot(slot); }
    for (auto s : coreSender->getSenders()) {
      coreReceiver->addSender(s);
      coreSlot->addNotifier(s);
    }

    this->registerNode(std::dynamic_pointer_cast<CoreNode>(from->core()));
    this->registerNode(std::dynamic_pointer_cast<CoreNode>(to->core()));
  }

  /// @brief Set a node as input for the graph
  /// @tparam UserDefinedMultiReceiver Node's type
  /// @tparam InputsMR Tuple of UserDefinedMultiReceiver's input type
  /// @tparam InputsG Tuple of Graph's input type
  /// @tparam isMultiReceiver Defined if UserDefinedMultiReceiver is derived from MultiReceiver
  /// @tparam isInputCompatible Defined if UserDefinedMultiReceiver and Graph (this) are compatible
  /// @param inputNode Node to set as Graph's input
  template<
      class UserDefinedMultiReceiver,
      class InputsMR = typename UserDefinedMultiReceiver::inputs_t,
      class InputsG = typename behavior::MultiReceivers<GraphInputs...>::inputs_t,
      class isMultiReceiver = typename std::enable_if_t<
          std::is_base_of_v<typename helper::HelperMultiReceiversType<InputsMR>::type, UserDefinedMultiReceiver>
      >,
      class isInputCompatible = typename std::enable_if_t<traits::is_included_v<InputsMR, InputsG>>>
  void input(std::shared_ptr<UserDefinedMultiReceiver> inputNode) {
    if (this->isInside()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }

    if (auto inputCoreNode =
        dynamic_cast<typename helper::HelperCoreMultiReceiversType<InputsMR>::type *>(inputNode->core().get())) {
      HLOG_SELF(0, "Set " << inputCoreNode->name() << "(" << inputCoreNode->id() << ") as input")

      if (inputCoreNode->hasBeenRegistered()) {
        if (inputCoreNode->belongingNode() != this) {
          std::ostringstream oss;
          oss << "The node " << inputCoreNode->name() << " belong already to another coreGraph: "
              << __FUNCTION__;
          HLOG_SELF(0, oss.str())
          throw (std::runtime_error(oss.str()));
        }
      }

      //Add it as input of the coreGraph
      this->inputsCoreNodes_->insert(inputCoreNode);
      this->addReceiversToSource(inputCoreNode);
    } else {
      std::ostringstream oss;
      oss << "The node " << inputCoreNode->name() << " is not a multi receiver: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
    this->registerNode(std::static_pointer_cast<CoreNode>(inputNode->core()));
  }

  /// @brief Set a node as output for the graph
  /// @tparam UserDefinedSender Node's type
  /// @tparam IsSender Defined if UserDefinedSender is derived from sender and has the same output as Graph's output
  /// @param outputNode Node to set as Graph's output
  void output(std::shared_ptr<behavior::Sender<GraphOutput>> outputNode) {
    if (this->isInside()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }

    if (auto outputCoreNode = dynamic_cast<CoreSender <GraphOutput> *>(outputNode->core().get())) {
      HLOG_SELF(0, "Set " << outputCoreNode->name() << "(" << outputCoreNode->id() << ") as outputNode")
      if (outputCoreNode->hasBeenRegistered()) {
        if (outputCoreNode->belongingNode() != this) {
          std::ostringstream oss;
          oss
              << "The node " << outputCoreNode->name() << " belong already to another coreGraph: "
              << __FUNCTION__;
          HLOG_SELF(0, oss.str())
          throw (std::runtime_error(oss.str()));
        }
      }
      this->outputCoreNodes_->insert(outputCoreNode);
      for (CoreSender <GraphOutput> *sender : outputCoreNode->getSenders()) {
        this->sink_->addNotifier(sender);
        this->sink_->addSender(sender);
      }
      outputCoreNode->addSlot(this->sink_.get());
      outputCoreNode->addReceiver(this->sink_.get());
    } else {
      std::ostringstream oss;
      oss << "Internal error, the output node is not a valid CoreSender: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }

    this->registerNode(std::static_pointer_cast<CoreNode>(outputNode->core()));
  }

  /// @brief Broadcast data and notify all input nodes
  /// @tparam Input Data input type
  /// @param data Data pushed into the graph, broadcast to all inputs
  template<
      class Input,
      class = typename std::enable_if_t<traits::Contains<Input, GraphInputs...>::value>
  >
  void broadcastAndNotifyToAllInputs(std::shared_ptr<Input> &data) {
    HLOG_SELF(2, "Broadcast data and notify all coreGraph's inputs")
    if (this->isInside()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
    std::static_pointer_cast<CoreQueueSender<Input>>(this->source_)->sendAndNotify(data);
  }

  /// @brief Set the graph as inside, in case of connection to another node
  void setInside() override {
    assert(!this->isInside());
    HLOG_SELF(0, "Set the coreGraph inside")
    CoreNode::setInside();
    // Remove the connection between the sink and the input nodes
    for (CoreNode *inputNode: *(this->inputsCoreNodes_)) {
      if (auto coreSlot = dynamic_cast<CoreSlot *>(inputNode)) {
        ( coreSlot->removeNotifier(
            static_cast<CoreNotifier *>(
                static_cast<CoreQueueSender<GraphInputs> *>(
                    this->source_.get()
                )
            )
        ), ...);

        // Remove the sender connection
        this->removeForAllSenders(inputNode);
      } else {
        std::ostringstream oss;
        oss << "Internal error, the input node is not a slot, when graph is set inside : " << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }

    // Disconnect the sink anf the output nodes
    std::for_each(this->outputCoreNodes_->begin(), this->outputCoreNodes_->end(),
                  [this](CoreSender <GraphOutput> *s) {
                    s->removeSlot(this->sink_.get());
                    s->removeReceiver(this->sink_.get());
                  });

    this->removeInsideNode(this->source_.get());
    this->removeInsideNode(this->sink_.get());
    this->source_ = nullptr;
    this->sink_ = nullptr;
  }

  /// @brief Get ids of input nodes (vector<pair<nodeId, nodeIdCluster>>)
  /// @return Ids of input nodes (vector<pair<nodeId, nodeIdCluster>>)
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> ids() const final {
    std::vector<std::pair<std::string, std::string>> v{};
    for (auto input : *(this->inputsCoreNodes_)) {
      for (std::pair<std::string, std::string> const &innerInput : input->ids()) { v.push_back(innerInput); }
    }
    return v;
  }

  /// @brief Execute the graph
  /// @details Do the duplication of inside nodes and launch the threads
  void executeGraph() {
    HLOG_SELF(2, "Execute the coreGraph")
    if (this->isInside()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
    this->startExecutionTimeStamp(std::chrono::system_clock::now());
    createInnerClustersAndLaunchThreads();
    auto finishCreationTime = std::chrono::system_clock::now();
    this->creationDuration(finishCreationTime - this->creationTimeStamp());
//        std::chrono::duration_cast<std::chrono::microseconds>(
//        finishCreationTime - this->creationTimeStamp())
//        );
  }

  /// @brief Wait for all inside threads to join
  void waitForTermination() {
    HLOG_SELF(2, "Wait for the coreGraph to terminate")
    this->scheduler_->joinAll();
    std::chrono::time_point<std::chrono::system_clock>
        endExecutionTimeStamp = std::chrono::system_clock::now();
    this->executionDuration(endExecutionTimeStamp - this->startExecutionTimeStamp());
//        std::chrono::duration_cast<std::chrono::microseconds>
//                                (endExecutionTimeStamp - this->startExecutionTimeStamp()));
  }

  /// @brief Notify the graph no more input data will be pushed
  void finishPushingData() {
    HLOG_SELF(2, "Indicate finish pushing data")
    if (this->isInside()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
    this->source_->notifyAllTerminated();
  }

  /// @brief Get data out of the graph
  /// @details
  /// - If the graph is not terminated:
  ///     - If an output data is available: the output data is returned,
  ///     - If no output data is available: the call is blocking until an output data is available
  /// - If the graph is terminated : return nullptr
  /// @attention If finishPushingData is not called there is a risk of deadlock
  /// @return An output data or nullptr
  std::shared_ptr<GraphOutput> getBlockingResult() {
    HLOG_SELF(2, "Get blocking data")
    if (this->isInside()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
    std::shared_ptr<GraphOutput> result = nullptr;
    this->sink_->waitForNotification();
    this->sink_->lockUniqueMutex();
    if (!this->sink_->receiverEmpty()) { result = this->sink_->popFront(); }
    this->sink_->unlockUniqueMutex();
    return result;
  }

  /// @brief Create all clusters for inside nodes and launch the threads, not gathered into insideNodesGraph
  /// @param insideNodesGraph not used
  void createCluster([[maybe_unused]]std::shared_ptr<std::multimap<CoreNode *,
                                                                   std::shared_ptr<CoreNode>>> &insideNodesGraph) override {
    createInnerClustersAndLaunchThreads();
  }

  /// @brief Special visit method for a CoreGraph
  /// @param printer Printer visitor to print the CoreGraph
  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    // Test if the coreGraph has already been visited
    if (printer->hasNotBeenVisited(this)) {

      // Print the header of the coreGraph
      printer->printGraphHeader(this);

      // Print the coreGraph information
      printer->printNodeInformation(this);

      if (this->source_ && this->sink_) {
        this->source_->visit(printer);
        this->sink_->visit(printer);
      }

      // Visit all the insides node of the coreGraph
      for (auto it = this->insideNodes()->begin(), end = this->insideNodes()->end(); it != end;
           it = this->insideNodes()->upper_bound(it->first)) {
        if (this->insideNodes()->count(it->first) == 1) {
          it->second->visit(printer);
        } else {
          this->printCluster(printer, it->second);
        }
      }
      // Print coreGraph footer
      printer->printGraphFooter(this);
    }
  }

  //Virtual functions
  //Sender
  /// @brief Add a receiver to the graph, i.e, add a receiver to all output nodes.
  /// @param receiver Receiver to add to all output nodes
  void addReceiver(CoreReceiver<GraphOutput> *receiver) override {
    HLOG_SELF(0, "Add receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (CoreSender <GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
      outputNode->addReceiver(receiver);
    }
  }

  /// @brief Remove a receiver from the graph, i.e, remove a receiver from all output nodes.
  /// @param receiver Receiver to remove from all output nodes
  void removeReceiver(CoreReceiver<GraphOutput> *receiver) override {
    HLOG_SELF(0, "Remove receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (CoreSender <GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
      outputNode->removeReceiver(receiver);
    }
  }

  /// @brief Send a data and notify receivers, not possible for a graph, throws an error in every case
  /// @exception std::runtime_error A graph do not send data
  /// @param ptr ptr to send to receivers
  void sendAndNotify([[maybe_unused]]std::shared_ptr<GraphOutput> ptr) override {
    std::ostringstream oss;
    oss << "Internal error, a graph do not send data: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  //Notifier
  /// @brief Add a slot to a graph, i.e, to all output nodes
  /// @param slot CoreSlot to add to all graph's output nodes
  void addSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Add Slot " << slot->name() << "(" << slot->id() << ")")
    for (CoreSender <GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
      outputNode->addSlot(slot);
    }
  }

  /// @brief Remove a slot from a graph, i.e, from all output nodes
  /// @param slot CoreSlot to remove from all graph's output nodes
  void removeSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Remove Slot " << slot->name() << "(" << slot->id() << ")")
    for (CoreSender <GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
      outputNode->removeSlot(slot);
    }
  }

  /// @brief Notify termination to all connected nodes, not possible for a graph, throw an error in every case
  /// @exception std::runtime_error A graph do not send data
  void notifyAllTerminated() override {
    std::ostringstream oss;
    oss << "Internal error, a graph  do not notify nodes: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Add a notifier to the graph, ie, to all input nodes
  /// @param notifier CoreNotifier to add to all graph's input nodes
  void addNotifier(CoreNotifier *notifier) override {
    HLOG_SELF(0, "Add Notifier " << notifier->name() << "(" << notifier->id() << ")")
    for (CoreNode *inputNode: *(this->inputsCoreNodes_)) {
      if (auto coreSlot = dynamic_cast<CoreSlot *>(inputNode)) {
        coreSlot->addNotifier(notifier);
      } else {
        std::ostringstream oss;
        oss << "Internal error, A graph's input node is not a slot: " << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }
  }

  /// @brief Remove a notifier from the graph, ie, from all input nodes
  /// @param notifier CoreNotifier to remove from all graph's input nodes
  void removeNotifier(CoreNotifier *notifier) override {
    HLOG_SELF(0, "Remove Notifier " << notifier->name() << "(" << notifier->id() << ")")
    for (CoreNode *inputNode: *(this->inputsCoreNodes_)) {
      if (auto coreSlot = dynamic_cast<CoreSlot *>(inputNode)) {
        coreSlot->removeNotifier(notifier);
      } else {
        std::ostringstream oss;
        oss << "Internal error, A graph's input node is not a slot: " << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }
  }

  /// @brief Test notifier for the graph, should not be used, connection is made to the input nodes
  /// @return nothing, throw an error
  bool hasNotifierConnected() override {
    std::ostringstream oss;
    oss << "Internal error, A graph has no notifier connected: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Return the number of input nodes connected, a graph should not have such a connection, throws in every case
  /// @return nothing, throw an error
  [[nodiscard]] size_t numberInputNodes() const override {
    std::ostringstream oss;
    oss << "Internal error, A graph's is not directly connected to the input nodes: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Wake up a graph, wake up all input nodes
  void wakeUp() override {
    HLOG_SELF(2, "Wake up all inputs")
    for (CoreNode *inputNode: *(this->inputsCoreNodes_)) {
      if (auto coreSlot = dynamic_cast<CoreSlot *>(inputNode)) {
        coreSlot->wakeUp();
      } else {
        std::ostringstream oss;
        oss << "Internal error, A graph's input is not a core slot: " << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }
  }

  /// @brief A graph can't wait for notification, throws an error in all case
  /// @exception std::runtime_error A graph do not wait for notification
  /// @return nothing, throw an error
  bool waitForNotification() override {
    std::ostringstream oss;
    oss << "Internal error, a graph is not connected to input nodes, so do not wait for notification: "
        << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Get the senders from the graphs, gather them from the output nodes
  /// @return Set of CoreSender from the graph's output nodes
  [[nodiscard]] std::set<CoreSender < GraphOutput>*> getSenders() override {
    std::set<CoreSender < GraphOutput>*> coreSenders;
    std::set<CoreSender < GraphOutput>*> tempCoreSenders;
    for (CoreSender <GraphOutput> *outputNode : *(this->outputCoreNodes_)) {
      tempCoreSenders = outputNode->getSenders();
      coreSenders.insert(tempCoreSenders.begin(), tempCoreSenders.end());
    }
    return coreSenders;
  }

  /// @brief Get the slots from the graphs, gather them from the input nodes
  /// @return Set of CoreSlot from the graph's input nodes
  [[nodiscard]] std::set<CoreSlot *> getSlots() override {
    std::set<CoreSlot *> coreSlots;
    std::set<CoreSlot *> tempCoreSlots;

    for (CoreNode *mr : *(this->inputsCoreNodes_)) {
      tempCoreSlots = mr->getSlots();
      coreSlots.insert(tempCoreSlots.begin(), tempCoreSlots.end());
    }
    return coreSlots;
  }

  /// @brief Join the threads managed by the graph
  void joinThreads() override {
    HLOG_SELF(2, "Join coreGraph threads")
    this->scheduler_->joinAll();
  }

  /// @brief Create inside nodes' cluster and launch the threads
  void createInnerClustersAndLaunchThreads() {
    HLOG_SELF(0, "Cluster creation")
    std::vector<std::shared_ptr<CoreNode>> insideCoreNodes;
    insideCoreNodes.reserve(this->insideNodes()->size());
    for (auto coreNode : *(this->insideNodes())) { insideCoreNodes.push_back(coreNode.second); }
    for (auto const &insideCoreNode : insideCoreNodes) { insideCoreNode->createCluster(this->insideNodes()); }
    launchThreads();
  }

  /// @brief Launch the threads using the graph's scheduler
  void launchThreads() {
    HLOG_SELF(0, "Launching threads")
    std::vector<std::shared_ptr<CoreNode>> insideCoreNodes;
    insideCoreNodes.reserve(this->insideNodes()->size());
    for (auto coreNode : *(this->insideNodes())) { insideCoreNodes.push_back(coreNode.second); }
    this->scheduler_->spawnThreads(insideCoreNodes);
  }

 private:
  /// @brief Register a node inside the graph
  /// @param coreNode Node to register inside the graph
  void registerNode(const std::shared_ptr<CoreNode> &coreNode) {
    HLOG_SELF(0, "Register coreNode " << coreNode->name() << "(" << coreNode->id() << ")")
    if (!coreNode->hasBeenRegistered()) {
      coreNode->setInside();
      this->addUniqueInsideNode(coreNode);
    }
  }

  /// @brief Duplicate inside nodes, called by CoreExecutionPipeline
  /// @param rhs CoreGraph to duplicate
  void duplicateInsideNodes(CoreGraph<GraphOutput, GraphInputs...> const &rhs) {
    // Inside nodes to copy, MainClusterNode -> ClusterNodes
    std::multimap<CoreNode *, std::shared_ptr<CoreNode>> &originalInsideNodes = *(rhs.insideNodes());
    // Correspondence map, original node -> copy node
    std::map<CoreNode *, std::shared_ptr<CoreNode>> correspondenceMap;
    //Duplicate node
    std::shared_ptr<CoreNode> duplicate;

    // Create all the duplicates and link them to their original node
    for (std::pair<CoreNode *const, std::shared_ptr<CoreNode>> const &originalNode : originalInsideNodes) {
      duplicate = originalNode.second->clone();
      duplicate->belongingNode(this);
      correspondenceMap.insert({originalNode.second.get(), duplicate});
    }

    // Add the duplicate node into the insideNode structure
    for (std::pair<CoreNode *const, std::shared_ptr<CoreNode>> const &originalNode : originalInsideNodes) {
      // Original node
      CoreNode *originalInsideNode = originalNode.second.get();
      // Copy node
      std::shared_ptr<CoreNode> duplicateInsideNode = correspondenceMap.find(originalInsideNode)->second;
      duplicateInsideNode->belongingNode(this);
      this->insideNodes()->insert({duplicateInsideNode.get(), duplicateInsideNode});
    }

    //Do the linkage
    for (std::pair<CoreNode *const, std::shared_ptr<CoreNode>> const &originalNode : originalInsideNodes) {
      CoreNode *originalInsideNode = originalNode.second.get();
      std::shared_ptr<CoreNode> duplicateInsideNode = correspondenceMap.find(originalInsideNode)->second;
      originalInsideNode->duplicateEdge(duplicateInsideNode.get(), correspondenceMap);
    }

    // Duplicate input nodes
    for (CoreNode *originalInputNode : *(rhs.inputsCoreNodes())) {
      auto shInputCoreNode = correspondenceMap.find(originalInputNode)->second;
      auto inputCoreNode = shInputCoreNode.get();
      this->inputsCoreNodes_->insert(inputCoreNode);
      (this->duplicateInputNodes<GraphInputs>(dynamic_cast<CoreReceiver<GraphInputs> *>(inputCoreNode)), ...);
      this->registerNode(shInputCoreNode);
    }

    // Duplicate output nodes
    for (CoreSender <GraphOutput> *originalOutputNode : *(rhs.outputCoreNodes())) {
      auto shOutputCoreNode = correspondenceMap.find(originalOutputNode)->second;
      if (auto outputCoreNode = dynamic_cast<CoreSender <GraphOutput> *>(shOutputCoreNode.get())) {
        this->outputCoreNodes_->insert(outputCoreNode);

        for (CoreSender <GraphOutput> *sender : outputCoreNode->getSenders()) {
          this->sink_->addNotifier(sender);
          this->sink_->addSender(sender);
        }

        outputCoreNode->addSlot(this->sink_.get());
        outputCoreNode->addReceiver(this->sink_.get());

        this->registerNode(std::static_pointer_cast<CoreNode>(shOutputCoreNode));
      } else {
        std::ostringstream oss;
        oss << "Internal error, the output node is not a sender: " << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }
  }

  /// @brief Add receivers to source and do the connection
  /// @tparam InputNodeTypes Node input types
  /// @param inputCoreNode Node to add to the source
  template<class ...InputNodeTypes>
  void addReceiversToSource(CoreMultiReceivers<InputNodeTypes...> *inputCoreNode) {
    //Set Slot/Notifiers
    this->source_->addSlot(inputCoreNode);
    (this->addSourceNotifierInputCoreNode<InputNodeTypes, InputNodeTypes...>(inputCoreNode), ...);
    // If casting not correct, send nullptr that is test inside CoreGraph::addReceiverToSource
    (this->addReceiverToSource<InputNodeTypes>(dynamic_cast<CoreReceiver<InputNodeTypes> *>(inputCoreNode)), ...);
  }

  /// @brief Add an input node to the source
  /// @tparam InputNodeType Specific input type to make the connection
  /// @tparam InputNodeTypes All node's input types
  /// @param inputCoreNode CoreMultiReceivers to connect to the source
  template<class InputNodeType, class ...InputNodeTypes>
  void addSourceNotifierInputCoreNode(CoreMultiReceivers<InputNodeTypes...> *inputCoreNode) {
    if (auto compatibleSourceType = std::dynamic_pointer_cast<CoreQueueSender<InputNodeType>>(this->source_)) {
      inputCoreNode->addNotifier(compatibleSourceType.get());
      compatibleSourceType->addReceiver(inputCoreNode);
    }
  }

  /// @brief If the input core node is compatible, connect it to the source
  /// @tparam InputNodeType Input node's type
  /// @param inputCoreNode Input node to connect to the source
  template<class InputNodeType>
  void addReceiverToSource(CoreReceiver<InputNodeType> *inputCoreNode) {
    if (inputCoreNode) {
      if (auto compatibleSource = dynamic_cast<CoreSender <InputNodeType> *>(this->source_.get())) {
        inputCoreNode->addSender(compatibleSource);
        dynamic_cast<CoreGraphReceiver<InputNodeType> *>(this)->addGraphReceiverInput(inputCoreNode);
      }
    }
  }

  /// @brief Duplicate input nodes, and do the connections for the source compatible type
  /// @tparam InputNodeType Specific Input node type
  /// @param inputCoreNode Input node to connect to the source
  template<class InputNodeType>
  void duplicateInputNodes(CoreReceiver<InputNodeType> *inputCoreNode) {
    if (inputCoreNode) {
      static_cast<CoreGraphReceiver<InputNodeType> *>(this)->addGraphReceiverInput(dynamic_cast<CoreReceiver<
          InputNodeType> *>(inputCoreNode));
      if (auto coreSlot = dynamic_cast<CoreSlot *>(inputCoreNode)) {
        this->source_->addSlot(coreSlot);
        coreSlot->addNotifier(std::static_pointer_cast<CoreQueueSender<InputNodeType>>(this->source_).get());
      } else {
        std::ostringstream oss;
        oss << "Internal error, the inputCoreNode is not a CoreSlot: " << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
      std::static_pointer_cast<CoreQueueSender<InputNodeType>>(this->source_)->addReceiver(dynamic_cast<CoreReceiver<
          InputNodeType> *>(inputCoreNode));
      dynamic_cast<CoreReceiver<InputNodeType> *>(inputCoreNode)->addSender(static_cast<CoreSender <InputNodeType> *>(this->source_.get()));
    }
  }

  /// @brief Specialized method if the input node is in a cluster
  /// @param printer Printer visitor to print the cluster
  /// @param node Main node cluster
  void printCluster(AbstractPrinter *printer, std::shared_ptr<CoreNode> const &node) {
    printer->printClusterHeader(node->coreClusterNode());
    for (auto it = this->insideNodes()->equal_range(node.get()).first;
         it != this->insideNodes()->equal_range(node.get()).second; ++it) {
      printer->printClusterEdge(it->second.get());
      it->second->visit(printer);
    }
    printer->printClusterFooter();
  }
};
}

}
#endif //HEDGEHOG_CORE_GRAPH_H
