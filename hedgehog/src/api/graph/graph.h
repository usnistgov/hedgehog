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

#ifndef HEDGEHOG_GRAPH_H
#define HEDGEHOG_GRAPH_H

#include <ostream>
#include <filesystem>

#include "../../behavior/node.h"
#include "../../behavior/copyable.h"
#include "../../behavior/input_output/multi_senders.h"
#include "../../behavior/input_output/multi_receivers.h"

#include "../../core/nodes/core_graph.h"
#include "../printer/options/color_scheme.h"
#include "../printer/options/structure_options.h"
#include "../printer/options/debug_options.h"
#include "../printer/dot_printer.h"
#include "../printer/jet_color.h"
#include "result_visitor.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// Abstract Execution Pipeline forward declaration
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class AbstractExecutionPipeline;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// Hedgehog graph abstraction.
/// @brief The graph in the Hedgehog library allows the user to create a dataflow. The graph regroups a set of nodes
/// representing parts of the computation. The most important nodes are the tasks (for heavy computation), the state
/// managers (to manage the local state of the computation) and the graph themselves (nested computation).
///
/// If a node is set as input of the graph (Graph::input / Graph::inputs), the data sent to the graph are transmitted to
/// the input nodes. The output nodes (set with Graph::output / Graph::outputs) produce the output data of the graph.
/// Between the input and the output nodes, the nodes need to be connected via edges(Graph::edge / Graph::edges).
///
/// A graph can be duplicated with ah hh::AbstractExecutionPipeline. Useful for mapping data across multiple devices (such as GPUs).
///
/// The difference between the singular and plural method (Graph::input / Graph::inputs, Graph::output / Graph::outputs,
/// Graph::edge / Graph::edges) is/are the type[s] used for the connection: if it is plural the connection is made for
/// all possible types, if it is singular the connection is only made for the user specified type.
///
/// The sequence of operations to use a graph are:
///     - Instantiate a graph [See example]
///     - Populate the graph [See example]
///     - Run the graph (Graph::executeGraph)
///     - Push data into the graph (Graph::pushData)
///     - Indicate that no more data will be pushed (Graph::finishPushingData)
///     - Gather output data with Graph::getBlockingResult can be put in a while loop, the function returns nullptr when the
/// graph is terminated (with Graph::finishPushingData) and all data have been processed.
///     - Wait for the graph to fully terminate (Graph::waitForTermination)
/// @code
/// // Instantiate a graph and its nodes
///  auto g = std::make_shared<hh::Graph<3, int, float, double, int, float, double>>();
///  auto innerInputInt = std::make_shared<IntFloatDoubleTask>();
///  auto innerTaskFloat = std::make_shared<IntFloatDoubleTask>();
///  auto innerOutput = std::make_shared<IntFloatDoubleTask>();
///  auto innerSM = std::make_shared<hh::StateManager<1, int, int>>(std::make_shared<IntState>());
///  auto innerGraph = std::make_shared<IntFloatDoubleGraph>();
///
/// // Create a graph
///  g->input<int>(innerInputInt);
///  g->input<float>(innerTaskFloat);
///  g->inputs(innerSM);
///  g->inputs(innerGraph);
///
///  g->edges(innerInputInt, innerOutput);
///  g->edges(innerSM, innerOutput);
///  g->edges(innerTaskFloat, innerOutput);
///  g->outputs(innerOutput);
///  g->outputs(innerGraph);
///
///  g->executeGraph(); // Execute the graph
///
///  // Push different types of data
///  for (int i = 0; i < 2000; ++i) {
///    g->pushData(std::make_shared<int>(i));
///    g->pushData(std::make_shared<float>(i));
///    g->pushData(std::make_shared<double>(i));
///  }
///
///  // Indicate that no other data will be pushed (trigger termination of the graph)
///  g->finishPushingData();
///
///  // Get the output data
///  while (auto variant = g->getBlockingResult()) {
///    std::visit(hh::ResultVisitor{
///        [](std::shared_ptr<int> &val) { /*Do something with an int*/ },
///        [](std::shared_ptr<float> &val) { /*Do something else with a float*/ },
///        [](std::shared_ptr<double> &val) { /*Do something else again with a double*/ }
///      }, *variant);
///  }
///
///  // Wait for the graph to terminate
///  g->waitForTermination();
/// @endcode
/// The default scheduling method for a graph is to launch all the node's threads and let the OS manages the threads.
/// @attention The conformity rules are: 1) for the input nodes, the node and the graph should share at least one input
/// type, 2) for the output nodes, the node and the graph should share at least one output type and 3) between two nodes
/// an edge can only be drawn for [a] common type[s].
/// @attention If the process wants to push a certain amount of data / get the output and start with new input
/// data, the Graph::pushData and Graph::getBlockingResult can be alternated without the while loop because the graph
/// won't terminate. This technique can only be used if the number of output can be deduced in advance by the end user.
/// Once all processing is complete, then the user must indicate they are done with Graph::finishPushingData, otherwise
/// the graph will deadlock and never terminate.
/// The Graph::cleanGraph method can be used to "clean" the graph nodes, and reset the user nodes attributes between
/// computations.
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class Graph :
    public behavior::Node,
    public tool::BehaviorMultiReceiversTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>,
    public tool::BehaviorMultiSendersTypeDeducer_t<tool::Outputs<Separator, AllTypes...>> {
 private:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  /// @brief Declare CoreExecutionPipeline as friend
  friend core::CoreExecutionPipeline<Separator, AllTypes...>;
  /// @brief Declare AbstractExecutionPipeline as friend
  friend AbstractExecutionPipeline<Separator, AllTypes...>;
#endif //DOXYGEN_SHOULD_SKIP_THIS

  std::shared_ptr<core::CoreGraph<Separator, AllTypes...>> const
      coreGraph_ = nullptr; ///< Core of the graph
  std::unique_ptr<std::set<std::shared_ptr<Node>>>
      nodes_ = nullptr; ///< Set of nodes given by the end user

 public:
  /// Default graph constructor, construct a graph with the name "Graph" and with a default scheduler.
  /// @param name Name of the graph to construct
  /// @param scheduler Scheduler used by the graph (should inherit from hh::Scheduler)
  /// @throw std::runtime_error if the core is not valid, should derives from CoreGraph
  explicit Graph(
      std::string const &name = "Graph",
      std::unique_ptr<Scheduler> scheduler = std::make_unique<DefaultScheduler>()) :
      behavior::Node(std::make_unique<core::CoreGraph<Separator, AllTypes...>>(name, std::move(scheduler), this)),
      coreGraph_(std::dynamic_pointer_cast<core::CoreGraph<Separator, AllTypes...>>(this->core())),
      nodes_(std::make_unique<std::set<std::shared_ptr<Node>>>()) {
    if (coreGraph_ == nullptr) { throw std::runtime_error("The core used by the graph should be a CoreGraph."); }
  }

  /// Default graph destructor
  ~Graph() override = default;

  /// Set an input node and connect the node to the graph's inputs for all common types.
  /// @brief Check if the input node is a valid object, and then connect it to the graph for all common types.
  /// @tparam InputNode_t Type of the input node
  /// @param inputNode Input node to connect
  template<tool::CompatibleInputNode<typename core::GIM<Separator, AllTypes...>::inputs_t> InputNode_t>
  void inputs(std::shared_ptr<InputNode_t> inputNode) {
    auto node = std::static_pointer_cast<Node>(inputNode);
    // Store shared_ptr in case user dereference it while it is still needed
    nodes_->insert(node);
    this->coreGraph_->template setInputForAllCommonTypes<typename InputNode_t::inputs_t>(node->core().get());
  }

  /// Set an input node and connect the node to the graph's input InputDataType.
  /// @brief Check if the input node is a valid object, and then connect it to the graph input for the InputDataType type.
  /// @tparam InputDataType Input type used for the connection between the node and the graph
  /// @tparam InputNode_t Type of the input node
  /// @param inputNode Input node to connect
  template<
      class InputDataType,
      tool::CompatibleInputNodeForAType<InputDataType,
                                        typename core::GIM<Separator, AllTypes...>::inputs_t> InputNode_t>
  void input(std::shared_ptr<InputNode_t> inputNode) {
    auto node = std::static_pointer_cast<Node>(inputNode);
    // Store shared_ptr in case user dereference it while it is still needed
    nodes_->insert(node);
    this->coreGraph_->template setInputForACommonType<InputDataType,
                                                      typename InputNode_t::inputs_t>(node->core().get());
  }

  /// Set an output node and connect the node to the graph's outputs for all common types.
  /// @brief Check if the output node is a valid object, and then connect it to the graph output for all common types.
  /// @tparam OutputNode_t Type of the output node
  /// @param outputNode Output node to connect
  template<tool::CompatibleOutputNode<typename core::GOM<Separator, AllTypes...>::outputs_t> OutputNode_t>
  void outputs(std::shared_ptr<OutputNode_t> outputNode) {
    auto node = std::static_pointer_cast<Node>(outputNode);
    // Store shared_ptr in case user dereference it while it is still needed
    nodes_->insert(node);

    this->coreGraph_->template setOutputForAllCommonTypes<typename OutputNode_t::outputs_t>(node->core().get());
  }

  /// Set an output node and connect the node to the graph's output OutputDataType.
  /// @brief Check if the output node is a valid object, and then connect it to the graph output for the OutputDataType type.
  /// @tparam OutputDataType Output type used for the connection between the node and the graph
  /// @tparam OutputNode_t Type of the output node
  /// @param outputNode Output node to connect
  template<class OutputType,
      tool::CompatibleOutputNodeForAType<OutputType,
                                         typename core::GOM<Separator, AllTypes...>::outputs_t> OutputNode_t>
  void output(std::shared_ptr<OutputNode_t> outputNode) {
    auto node = std::static_pointer_cast<Node>(outputNode);
    // Store shared_ptr in case user dereference it while it is still needed
    nodes_->insert(node);
    this->coreGraph_->template setOutputForACommonType<OutputType,
                                                       typename OutputNode_t::outputs_t>(node->core().get());
  }

  /// Create an edge between two nodes for all common types
  /// @brief Validate the sender and receiver node and if valid, create an edge for all common types between the sender
  /// output types and the receiver input types
  /// @tparam SenderNode_t Type of the sender node
  /// @tparam ReceiverNode_t Type of the receiver node
  /// @param sender Sender node
  /// @param receiver Receiver node
  template<tool::SenderNode SenderNode_t, tool::ReceiverNode ReceiverNode_t>
  void edges(std::shared_ptr<SenderNode_t> sender, std::shared_ptr<ReceiverNode_t> receiver) {
    static_assert(
        std::tuple_size_v<
            tool::Intersect_t<typename SenderNode_t::outputs_t, typename ReceiverNode_t::inputs_t>
        > != 0, "The sender and the receiver nodes should at least share a type."
    );
    auto senderNode = std::static_pointer_cast<Node>(sender);
    auto receiverNode = std::static_pointer_cast<Node>(receiver);
    // Store shared_ptr in case user dereference it while it is still needed
    nodes_->insert(senderNode);
    nodes_->insert(receiverNode);

    this->coreGraph_
        ->template addEdgeForAllCommonTypes<typename SenderNode_t::outputs_t, typename ReceiverNode_t::inputs_t>
            (senderNode->core().get(), receiverNode->core().get());
  }

  /// Create an edge between two nodes for a specific type
  /// @brief Validate the sender and receiver node and if valid, create an edge for the CommonType type
  /// @tparam SenderNode_t Type of the sender node
  /// @tparam ReceiverNode_t Type of the receiver node
  /// @param sender Sender node
  /// @param receiver Receiver node
  template<class CommonType,
      tool::SenderNodeForAType<CommonType> SenderNode_t, tool::ReceiverNodeForAType<CommonType> ReceiverNode_t>
  void edge(std::shared_ptr<SenderNode_t> sender, std::shared_ptr<ReceiverNode_t> receiver) {
    auto senderNode = std::static_pointer_cast<Node>(sender);
    auto receiverNode = std::static_pointer_cast<Node>(receiver);
    // Store shared_ptr in case user dereference it while it is still needed
    nodes_->insert(senderNode);
    nodes_->insert(receiverNode);

    this->coreGraph_->template addEdgeForACommonType<
        CommonType,
        typename SenderNode_t::outputs_t, typename ReceiverNode_t::inputs_t>
        (senderNode->core().get(), receiverNode->core().get());
  }

  /// Execute the graph
  /// @brief Duplicate the nodes in a group and use the scheduler to associate threads to the nodes
  /// @param waitForInitialization Wait for internal nodes to be initialized flags [default = false]
  void executeGraph(bool waitForInitialization = false) { coreGraph_->executeGraph(waitForInitialization); }

  /// Indicate to the graph that no more input data are pushed to the graph
  /// @brief Trigger the termination of the graph, each nodes terminates in a cascaded when (by default) no predecessor
  /// node is connected to the node and the input node queues are empty.
  void finishPushingData() { coreGraph_->finishPushingData(); }

  /// Push data into the graph
  /// @details Each input data is sent to all input nodes that match the input type that is sent.
  /// @tparam CompatibleInputType_t Type on the input data
  /// @param data Data sent to the graph
  template<tool::MatchInputTypeConcept<tool::Inputs<Separator, AllTypes...>> CompatibleInputType_t>
  void pushData(std::shared_ptr<CompatibleInputType_t> data) { this->coreGraph_->broadcastAndNotifyAllInputNodes(data); }

  /// Wait for the graph to terminate
  /// @brief A graph terminate when all the threads it manages are terminated (i.e. when all the nodes are terminated)
  void waitForTermination() { coreGraph_->waitForTermination(); }

  /// Get result data from the graph
  /// @brief Get result from the graph while blocking the main thread. The results are presented under the form of
  /// @code
  /// std::shared_ptr<std::variant<std::shared_ptr<Output1>, std::shared_ptr<Output2>, std::shared_ptr<Output3>>>
  /// @endcode
  /// If the output type is known in advance one can use
  /// @code
  /// std::get<std::shared_ptr<KnownType>>(*result)
  /// @endcode
  /// If multiple types are possible the following code can be used:
  /// @code
  /// std::visit(hh::ResultVisitor{
  ///     [](std::shared_ptr<Output1> &val) { /*Do something with an Output1*/ },
  ///     [](std::shared_ptr<Output2> &val) { /*Do something else with a Output2*/ },
  ///     [](std::shared_ptr<Output3> &val) { /*Do something else again with a Output3*/ }
  ///   }, *result);
  /// @endcode
  /// @return A result of the graph
  auto getBlockingResult() { return coreGraph_->getBlockingResult(); }

  /// Create a dot file representing a snapshot of the state of the graph at the moment of the call, the graph is
  /// saved into the file dotFilePath.
  /// @param dotFilePath Path where the file is stored.
  /// @param colorScheme Color scheme used to color the tasks, either to show difference in execution or in waiting
  /// times, or nothing. The chosen color depends on the colorPicker.
  /// @param structureOptions Show how the graph is represented, with or without input queue size, with or without all
  /// task groups.
  /// @param inputOption Select how the execution should be printed, by input type or gathered
  /// @param debugOption Add debug information on the dot graph.
  /// @param colorPicker Color scheme used to generate the dotfile, JetColor by default.
  /// @param verbose Enable verbose mode: report when dot files are created or overwritten to standard out, default
  /// false.
  void createDotFile(std::filesystem::path const &dotFilePath,
                     ColorScheme colorScheme = ColorScheme::NONE,
                     StructureOptions structureOptions = StructureOptions::NONE,
                     InputOptions inputOption = InputOptions::GATHERED,
                     DebugOptions debugOption = DebugOptions::NONE,
                     std::unique_ptr<ColorPicker> colorPicker = std::make_unique<JetColor>(),
                     bool verbose = false) {
    core::abstraction::GraphNodeAbstraction *core = this->coreGraph_.get();
    DotPrinter
        printer(
        std::filesystem::absolute(dotFilePath), colorScheme, structureOptions, inputOption, debugOption, core,
        std::move(colorPicker), verbose);
    core->visit(&printer);
  }

  /// @brief Set the device id (GPU ID) for all the nodes in a graph
  /// @param deviceId Device Id to set
  void deviceId(int deviceId) { this->coreGraph_->deviceId(deviceId); }

  /// @brief Clean the graph
  /// @brief Call in sequence the clean method in all internal nodes. May be use to reset the attributes of user nodes
  /// between computations.
  void cleanGraph() {
    this->coreGraph_->cleanGraph();
  }
};
}

#endif //HEDGEHOG_GRAPH_H
