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

#ifndef HEDGEHOG_CORE_EXECUTION_PIPELINE_H
#define HEDGEHOG_CORE_EXECUTION_PIPELINE_H

#include <utility>

#include "../../tools/traits.h"
#include "../../api/graph/graph.h"

#include "../abstractions/base/node/execution_pipeline_node_abstraction.h"
#include "../abstractions/node/execution_pipeline_inputs_management_abstraction.h"
#include "../abstractions/node/execution_pipeline_outputs_management_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration AbstractExecutionPipeline
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class AbstractExecutionPipeline;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog core namespace
namespace core {

/// @brief Type alias for an ExecutionPipelineInputsManagementAbstraction from the list of template parameters
template<size_t Separator, class ...AllTypes>
using EPIM =
    tool::ExecutionPipelineInputsManagementAbstractionTypeDeducer_t<tool::Inputs<Separator, AllTypes...>>;

/// @brief Type alias for an ExecutionPipelineOutputsManagementAbstraction from the list of template parameters
template<size_t Separator, class ...AllTypes>
using EPOM =
    tool::ExecutionPipelineOutputsManagementAbstractionTypeDeducer_t<tool::Outputs<Separator, AllTypes...>>;

/// @brief Execution pipeline core
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class CoreExecutionPipeline
    : public abstraction::ExecutionPipelineNodeAbstraction,
      public abstraction::ClonableAbstraction,
      public EPIM<Separator, AllTypes...>,
      public EPOM<Separator, AllTypes...> {
 private:
  AbstractExecutionPipeline<Separator, AllTypes...> *const
      executionPipeline_ = nullptr; ///< Pointer to the execution pipeline node
  std::vector<std::shared_ptr<CoreGraph<Separator, AllTypes...>>>
      coreGraphs_{}; ///< Vector of CoreGraph handled by the execution pipeline
  std::vector<int>
      deviceIds_{}; ///< Device ids matching the core graphs

 public:
  /// @brief Constructor using a user-defined execution pipeline, the base CoreGraph and the deviceIds to determine the
  /// number of graphs in the execution pipeline. The name is set as default as "Execution pipeline"
  /// @param executionPipeline User-defined execution pipeline
  /// @param coreGraph Base CoreGraph
  /// @param deviceIds Device Ids to set to the different graphs in the execution pipeline. The vector size determine
  /// the number of graphs in the execution pipeline.
  /// @param name Name of the execution pipeline
  CoreExecutionPipeline(
      AbstractExecutionPipeline<Separator, AllTypes...> *const &executionPipeline,
      std::shared_ptr<CoreGraph<Separator, AllTypes...>> const coreGraph,
      std::vector<int> deviceIds,
      std::string const name = "Execution pipeline")
      : abstraction::ExecutionPipelineNodeAbstraction(name, executionPipeline),
        EPIM<Separator, AllTypes...>(executionPipeline),
        EPOM<Separator, AllTypes...>(),
        executionPipeline_(executionPipeline),
        deviceIds_(std::move(deviceIds)) {
    initializeCoreExecutionPipeline(coreGraph);
  }

  /// @brief Constructor using a user-defined execution pipeline, the base CoreGraph, and the number of graphs in the
  /// execution pipeline. The name is set as default as "Execution pipeline"
  /// @details The device ids are set in sequence, the base graph as 0 and each clone as previous + 1.
  /// @param executionPipeline USer-defined execution pipeline
  /// @param coreGraph Base CoreGraph
  /// @param numberGraphs Number of graphs in the execution pipeline
  /// @param name Name of the Execution pipeline, default is "Execution pipeline"
  CoreExecutionPipeline(
      AbstractExecutionPipeline<Separator, AllTypes...> *const &executionPipeline,
      std::shared_ptr<CoreGraph<Separator, AllTypes...>> const coreGraph,
      size_t numberGraphs,
      std::string const name = "Execution pipeline")
      : abstraction::ExecutionPipelineNodeAbstraction(name, executionPipeline),
        EPIM<Separator, AllTypes...>(executionPipeline),
        EPOM<Separator, AllTypes...>(),
        executionPipeline_(executionPipeline),
        deviceIds_(std::vector<int>(numberGraphs)) {
    std::iota(deviceIds_.begin(), deviceIds_.end(), 0);
    initializeCoreExecutionPipeline(coreGraph);
  }

  /// @brief Default destructor
  ~CoreExecutionPipeline() override = default;

  /// @brief Do nothing as pre-run step
  void preRun() override {}

  /// @brief Main core execution pipeline logic
  /// @details
  /// - while the execution pipeline runs
  ///     - wait for data or termination
  ///     - if can terminate, break
  ///     - get a piece of data from the queue
  ///     - for each graphs
  ///         - if data need to be sent to the graph, send to the graph
  /// - disconnect the switch from each graphs
  /// - wait for each graph to terminate
  void run() override {
    std::chrono::time_point<std::chrono::system_clock>
        start,
        finish;

    volatile bool isDone;

    using Inputs_t = typename EPIM<Separator, AllTypes...>::inputs_t;
    using Indices = std::make_index_sequence<std::tuple_size_v<Inputs_t>>;

    this->isActive(true);

    // Actual computation loop
    while (!this->canTerminate()) {
      // Wait for a data to arrive or termination
      start = std::chrono::system_clock::now();
      isDone = this->sleep();
      finish = std::chrono::system_clock::now();
      this->incrementWaitDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));

      // If te can terminate break the loop early
      if (isDone) { break; }

      // Operate the connectedReceivers to get a data and send it to execute
      start = std::chrono::system_clock::now();
      this->operateReceivers<Inputs_t>(Indices{});
      finish = std::chrono::system_clock::now();
      this->incrementDequeueExecutionDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start));
    }

    // Do the shutdown phase
    this->postRun();
    // Wake up node that this is linked to
    this->wakeUp();
  }

  /// @brief Post run logic, disconnects the switch and waits for each graph to terminate
  void postRun() override {
    this->disconnectSwitch();
    for (auto &g : this->coreGraphs_) { g->waitForTermination(); }
    this->isActive(false);
  }

  /// @brief  Extra printing information for the execution pipeline
  /// @return An empty string
  [[nodiscard]] std::string extraPrintingInformation() const override { return {}; }

  /// @brief Visit an execution pipeline
  /// @param printer Printer used to gather information
  void visit(Printer *printer) override {
    // Register the node
    if (printer->registerNode(this)) {
      printer->printExecutionPipelineHeader(this, this->coreSwitch());
      EPIM<Separator, AllTypes...>::printEdgesInformation(printer);
      for (auto coreGraph : coreGraphs_) { coreGraph->visit(printer); }
      printer->printExecutionPipelineFooter();
    }
  }

  /// @brief Register the execution pipeline into the belongingGraph
  /// @param belongingGraph Graph to register the execution pipeline into
  void registerNode(abstraction::GraphNodeAbstraction *belongingGraph) override {
    NodeAbstraction::registerNode(belongingGraph);
    EPIM<Separator, AllTypes...>::coreSwitch()->registerNode(belongingGraph);
    for (auto coreGraph : coreGraphs_) { coreGraph->registerNode(belongingGraph); }
  }

  /// @brief Getter to the min max execution duration from the nodes inside the graphs in the execution pipeline
  /// @return Min max execution duration from the nodes inside the graphs in the execution pipeline
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxExecutionDuration() const override {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxExecDuration = {
        std::chrono::nanoseconds::max(), std::chrono::nanoseconds::min()
    };

    for (auto const &graph : coreGraphs_) {
      auto const minMaxGraph = graph->minMaxExecutionDuration();
      minMaxExecDuration.first = std::min(minMaxExecDuration.first, minMaxGraph.first);
      minMaxExecDuration.second = std::max(minMaxExecDuration.second, minMaxGraph.second);
    }
    return minMaxExecDuration;
  }

  /// @brief Getter to the min max wait duration from the nodes inside the graphs in the execution pipeline
  /// @return Min max wait duration from the nodes inside the graphs in the execution pipeline
  [[nodiscard]] std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxWaitDuration() const override {
    std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds> minMaxWaitDuration = {
        std::chrono::nanoseconds::max(), std::chrono::nanoseconds::min()
    };

    for (auto const &graph : coreGraphs_) {
      auto const minMaxGraph = graph->minMaxWaitDuration();
      minMaxWaitDuration.first = std::min(minMaxWaitDuration.first, minMaxGraph.first);
      minMaxWaitDuration.second = std::max(minMaxWaitDuration.second, minMaxGraph.second);
    }
    return minMaxWaitDuration;
  }

 private:
  /// @brief Initialize the execution pipeline
  /// @details Register, duplicate and connect the graph given in parameter
  /// @param coreGraph Base core graph
  /// @throw std::runtime_error If the graph is not valid, in the right state or the device ids are not valid
  void initializeCoreExecutionPipeline(std::shared_ptr<CoreGraph<Separator, AllTypes...>> const coreGraph) {
    if (coreGraph == nullptr) {
      throw std::runtime_error("An execution pipeline should be created with a valid graph (!= nullptr).");
    }
    if (coreGraph->graphStatus() != abstraction::GraphNodeAbstraction::INIT) {
      throw std::runtime_error("An execution pipeline should be created with a graph that is not already part of "
                               "another graph or being executed.");
    }
    if (deviceIds_.empty()) {
      throw std::runtime_error(
          "An execution pipeline should be created with a valid number of graph clones (!= 0) and with valid graph ids.");
    }
    coreGraphs_.reserve(deviceIds_.size());
    registerAndDuplicateGraph(coreGraph);
    connectCoreGraphs();
    setInitialized();
  }

  /// @brief Register and duplicate the coreGraph in the execution pipeline
  /// @details The CoreGraph is duplicated n times by calling clone() on it, all of the graphs are initialized and
  /// registered in the execution pipeline.
  /// @param coreGraph CoreGraph to duplicate and register
  void registerAndDuplicateGraph(std::shared_ptr<CoreGraph<Separator, AllTypes...>> const coreGraph) {
    coreGraph->deviceId(deviceIds_.at(0));
    coreGraph->graphId(0);
    coreGraph->setInside();
    coreGraphs_.push_back(coreGraph);
    for (size_t graphId = 1; graphId < this->deviceIds_.size(); ++graphId) {
      std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> correspondenceMap;
      auto clone = std::dynamic_pointer_cast<CoreGraph<Separator, AllTypes...>>(
          coreGraph->clone(correspondenceMap)
      );
      clone->deviceId(deviceIds_.at(graphId));
      clone->graphId(graphId);
      clone->setInside();
      coreGraphs_.push_back(clone);
    }
  }

  /// @brief Connect all of the core graphs into the execution pipeline (connection to the switch and output of the
  /// execution pipeline)
  void connectCoreGraphs() {
    for (auto &coreGraph : coreGraphs_) {
      EPIM<Separator, AllTypes...>::connectGraphToSwitch(coreGraph);
      EPOM<Separator, AllTypes...>::registerGraphOutputNodes(coreGraph);
    }
  }

  /// @brief Operate the receivers for all input types
  /// @tparam InputTypes Input data types
  /// @tparam Indices Indices to travers the tuple of input types
  template<class InputTypes, size_t ...Indices>
  void operateReceivers(std::index_sequence<Indices...>) {
    (this->template operateReceiver<std::tuple_element_t<Indices, InputTypes>>(), ...);
  }

  /// @brief Operate the receiver for a specific input type
  /// @details
  /// - Lock the mutex
  /// - If there is an input data available
  ///     - get the data
  ///     - unlock the mutex
  ///     - for each graph
  ///     - if the data can be send to the graph, send it
  /// - else, unlock the mutex
  /// @tparam Input
  template<class Input>
  void operateReceiver() {
    auto typedReceiver = static_cast<abstraction::ReceiverAbstraction<Input> *>(this);
    if (!typedReceiver->empty()) {
      std::shared_ptr<Input> data = nullptr;
      typedReceiver->getInputData(data);
      for (auto coreGraph : this->coreGraphs_) {
        if (EPIM<Separator, AllTypes...>::callSendToGraph(data, coreGraph->graphId())) {
          while(!std::static_pointer_cast<abstraction::ReceiverAbstraction<Input>>(coreGraph)->receive(data)) {
            cross_platform_yield();
          }
          std::static_pointer_cast<abstraction::SlotAbstraction>(coreGraph)->wakeUp();
        }
      }
    }
  }

  /// @brief Create graph's inner groups and launch graph's threads
  /// @param waitForInitialization Wait for internal nodes to be initialized flags
  void launchGraphThreads(bool waitForInitialization) override {
    for (auto coreGraph : this->coreGraphs_) {
      coreGraph->createInnerGroupsAndLaunchThreads(waitForInitialization);
    }
  }

  /// @brief Get the exec pipeline id
  /// @details In fact get switch id, they share the same id
  /// @return Execution pipeline id
  [[nodiscard]] std::string id() const override { return this->coreSwitch()->id(); }

  /// @brief Clone method, to duplicate an execution pipeline when it is part of another graph in an execution pipeline
  /// @param correspondenceMap Correspondence map of belonging graph's node
  /// @return Clone of this execution pipeline
  std::shared_ptr<abstraction::NodeAbstraction> clone(
      std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &correspondenceMap) override {
    return std::make_shared<CoreExecutionPipeline>(
        this->executionPipeline_,
        std::dynamic_pointer_cast<CoreGraph<Separator, AllTypes...>>(this->coreGraphs_[0]->clone(correspondenceMap)),
        this->deviceIds_);
  }

  /// @brief Duplicate the execution pipeline edge
  /// @param mapping Correspondence map of belonging graph's node
  void duplicateEdge(std::map<NodeAbstraction *, std::shared_ptr<NodeAbstraction>> &mapping) override {
    this->duplicateOutputEdges(mapping);
  }

  /// @brief Accessor to the execution duration per input
  /// @return A Map where the key is the type as string, and the value is the associated duration
  /// @throw std::runtime_error Not defined for ep
  [[nodiscard]] std::map<std::string, std::chrono::nanoseconds> const &executionDurationPerInput() const final {
    throw std::runtime_error("The execution per in put is not defined on an Execution Pipeline.");
  }

  /// @brief Accessor to the number of elements per input
  /// @return A Map where the key is the type as string, and the value is the associated number of elements received
  /// @throw std::runtime_error Not defined for ep
  [[nodiscard]] std::map<std::string, std::size_t> const &nbElementsPerInput() const final {
    throw std::runtime_error("The nb of elements per in put is not defined on an Execution Pipeline.");
  }

  /// @brief Accessor to the dequeue + execution duration per input
  /// @return Map in which the key is the type and the value is the duration
  /// @throw std::runtime_error Not defined for ep
  [[nodiscard]] std::map<std::string, std::chrono::nanoseconds> const &dequeueExecutionDurationPerInput() const final {
    throw std::runtime_error("The execution per in put is not defined on an Execution Pipeline.");
  }
};

}
}

#endif //HEDGEHOG_CORE_EXECUTION_PIPELINE_H
