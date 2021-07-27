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
#include <utility>



#ifndef HEDGEHOG_CORE_EXECUTION_PIPELINE_H
#define HEDGEHOG_CORE_EXECUTION_PIPELINE_H

#include "core_switch.h"
#include "../core_task.h"
#include "../../../api/graph.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief AbstractExecutionPipeline
/// @tparam GraphOutput Graph output type
/// @tparam GraphInputs Graph input types
template<class GraphOutput, class ...GraphInputs>
class AbstractExecutionPipeline;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog core namespace
namespace core {

/// @brief Execution Pipeline core
/// @details Duplicate and hold the copies of the graph given at construction. Associate also the device ids and the
///  graph ids to the copies. The graph and the copies are plug to the execution pipeline's switch, that will divert the
/// data coming to the execution pipeline to the different graphs.
/// @tparam GraphOutput Graph's output type
/// @tparam GraphInputs Graph's input types
template<class GraphOutput, class ...GraphInputs>
class CoreExecutionPipeline : public CoreTask<GraphOutput, GraphInputs...> {
 private:
  AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline_ = nullptr; ///< User's execution pipeline
  size_t numberGraphs_ = 0; ///< Total number of graphs in the execution pipeline
  std::vector<int> deviceIds_ = {}; ///< Device id's value to set to the different graphs into the execution pipeline

 protected:
  std::shared_ptr<CoreSwitch<GraphInputs...>> coreSwitch_; ///< Switch use to divert the data to the graphs
  std::vector<std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>>> epGraphs_ = {}; ///< Core Copies of the graphs
  ///< (actual memory is stored here)

 public:
  /// @brief Deleted Default constructor
  CoreExecutionPipeline() = delete;

  /// @brief The core execution pipeline constructor
  /// @param name Execution pipeline name
  /// @param executionPipeline User's execution pipeline
  /// @param coreBaseGraph Base graph to duplicate
  /// @param numberGraphs Number of graphs in the execution pipeline
  /// @param deviceIds Device ids to set to the different graphs
  /// @param automaticStart True if the graphs have to run automatically, else False
  CoreExecutionPipeline(std::string_view const &name,
                        AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline,
                        std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> coreBaseGraph,
                        size_t numberGraphs,
                        std::vector<int> const &deviceIds,
                        bool automaticStart)
      : CoreTask<GraphOutput, GraphInputs...>(name, 1, NodeType::ExecutionPipeline, nullptr, automaticStart),
        executionPipeline_(executionPipeline),
        numberGraphs_(numberGraphs),
        deviceIds_(deviceIds),
        coreSwitch_(std::make_shared<CoreSwitch<GraphInputs...>>()) {
    if (this->numberGraphs_ == 0) { this->numberGraphs_ = 1; }

    if (coreBaseGraph->isInside() || this->isInside()) {
      std::ostringstream oss;
      oss << "You can not modify a graph that is connected inside another graph: " << __FUNCTION__;
      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
    if (numberGraphs_ != deviceIds.size()) {
      std::ostringstream oss;
      oss
          << "The number of device Ids do not correspond to the number of coreGraph duplications you sent to the "
             "execution pipeline \""
          << name
          << "\". Even if you do not associate the graphs duplicates to a specific device, please set the deviceIds "
             "to the map to the number of duplicates specified."
          << std::endl;

      HLOG_SELF(0, oss.str())
      throw (std::runtime_error(oss.str()));
    }
    epGraphs_.reserve(this->numberGraphs_);

    coreBaseGraph->graphId(0);
    coreBaseGraph->deviceId(this->deviceIds_[0]);
    connectGraphToEP(coreBaseGraph);

    this->duplicateGraphs();
  }

  /// @brief Default destructor
  virtual ~CoreExecutionPipeline() = default;

  /// @brief Return the user's node
  /// @return User's node: CoreExecutionPipeline::executionPipeline_
  [[nodiscard]] behavior::Node *node() override { return this->executionPipeline_; }

  /// @brief Device ids accessor
  /// @return Device ids
  [[nodiscard]] std::vector<int> const &deviceIds() const { return deviceIds_; }

  /// @brief Number of execution pipeline's graphs accessor
  /// @return Number of execution pipeline's graphs
  [[nodiscard]] size_t numberGraphs() const { return numberGraphs_; }

  /// @brief Execution pipeline id, i.e switch id accessor
  /// @return Execution pipeline id, i.e switch id
  [[nodiscard]] std::string id() const override { return this->coreSwitch_->id(); }

  /// @brief User execution pipeline accessor
  /// @return User execution pipeline
  AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline() const {
    return executionPipeline_;
  }

  /// @brief Get a device id, not possible for an execution pipeline, throw an error in every case
  /// @exception std::runtime_error An execution pipeline does not have a device id
  /// @return Nothing, throw an error
  int deviceId() override {
    std::ostringstream oss;
    oss << "Internal error, an execution pipeline has not device id: " << __FUNCTION__;
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Execution pipeline's senders accessor
  /// @details Gather senders from all execution pipeline's graphs
  /// @return A set composed by execution pipeline's senders
  std::set<CoreSender<GraphOutput> *>
  getSenders() override {
    std::set<CoreSender<GraphOutput> *>
        res = {},
        senders = {};
    for (auto epGraph : this->epGraphs_) {
      senders.clear();
      senders = epGraph->getSenders();
      mergeSenders(res, senders);
    }
    return res;
  }

  /// @brief Return the core of the base graph
  /// @return Base graph's core
  std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> baseCoreGraph() {
    return this->epGraphs_.empty() ? nullptr : this->epGraphs_[0];
  }

  /// @brief Add a receiver to the execution pipeline, to all inside graphs
  /// @param receiver Receiver to add to the execution pipeline
  void addReceiver(CoreReceiver<GraphOutput> *receiver) override {
    for (CoreReceiver<GraphOutput> *r : receiver->receivers()) {
      if (auto coreQueueReceiver = dynamic_cast<CoreQueueReceiver<GraphOutput> *>(r)) {
        this->destinations()->insert(coreQueueReceiver);
      } else {
        std::ostringstream oss;
        oss << "Internal error, a receiver added to an execution pipeline is not a coreQueueReceiver : "
            << __FUNCTION__;
        HLOG_SELF(0, oss.str())
        throw (std::runtime_error(oss.str()));
      }
    }
    for (auto epGraph : this->epGraphs_) {
      connectGraphsOutputToReceiver(epGraph.get(), receiver);
    }
  }

  /// @brief Add a slot to a execution pipeline, i.e. to all inside graphs
  /// @param slot CoreSlot to add to all execution pipeline
  void addSlot(CoreSlot *slot) override {
    for (auto graph : this->epGraphs_) { graph->addSlot(slot); }
  }

  /// @brief Special visit method for an execution pipeline, visit also all inside graphs
  /// @param printer Printer visitor to print the CoreExecutionPipeline
  void visit(AbstractPrinter *printer) override {
    if (printer->hasNotBeenVisited(this)) {
      printer->printExecutionPipelineHeader(this, coreSwitch_.get());
      for (auto graph: epGraphs_) {
        (this->printEdgeSwitchGraphs<GraphInputs>(printer, graph.get()), ...);
        graph->visit(printer);
      }
      printer->printExecutionPipelineFooter();
    }
  }

  /// @brief Create inner graphs clusters and launch the threads
  void createCluster([[maybe_unused]]std::shared_ptr<std::multimap<CoreNode *,
                                                                   std::shared_ptr<CoreNode>>> &) override {
    for (std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> epGraph : this->epGraphs_) {
      epGraph->createInnerClustersAndLaunchThreads();
    }
  }

  /// @brief Return the maximum execution time of all inside graphs
  /// @return Maximum execution time of all inside graphs
  [[nodiscard]] std::chrono::nanoseconds maxExecutionTime() const override {
    std::chrono::nanoseconds ret = std::chrono::nanoseconds::min();
    for (auto graph: epGraphs_) {
      std::chrono::nanoseconds temp = graph->maxExecutionTime();
      if (temp > ret) ret = temp;
    }
    return ret;
  }

  /// @brief Return the minimum execution time of all inside graphs
  /// @return Minimum execution time of all inside graphs
  [[nodiscard]] std::chrono::nanoseconds minExecutionTime() const override {
    std::chrono::nanoseconds ret = std::chrono::nanoseconds::max();
    for (auto graph: epGraphs_) {
      std::chrono::nanoseconds temp = graph->minExecutionTime();
      if (temp < ret) ret = temp;
    }
    return ret;
  }

  /// @brief Return the maximum wait time of all inside graphs
  /// @return Maximum wait time of all inside graphs
  [[nodiscard]] std::chrono::nanoseconds maxWaitTime() const override {
    std::chrono::nanoseconds ret = std::chrono::nanoseconds::min();
    for (auto graph: epGraphs_) {
      std::chrono::nanoseconds temp = graph->maxWaitTime();
      if (temp > ret) ret = temp;
    }
    return ret;
  }

  /// @brief Return the minimum wait time of all inside graphs
  /// @return Minimum wait time of all inside graphs
  [[nodiscard]] std::chrono::nanoseconds minWaitTime() const override {
    std::chrono::nanoseconds ret = std::chrono::nanoseconds::max();
    for (auto graph: epGraphs_) {
      std::chrono::nanoseconds temp = graph->minWaitTime();
      if (temp < ret) ret = temp;
    }
    return ret;
  }

 protected:
  /// @brief Can terminate for the ep, specialised to not call user's defined one
  /// @param lock Node's mutex
  /// @return True if the node is terminated, else False
  bool callCanTerminate(bool lock) override {
    bool result;
    if (lock) { this->lockUniqueMutex(); }
    result = !this->hasNotifierConnected() && this->receiversEmpty();
    HLOG_SELF(2, "callCanTerminate: " << std::boolalpha << result)
    if (lock) { this->unlockUniqueMutex(); }
    return result;
  };

 private:
  /// @brief Duplicate the graphs and link it to the switch
  void duplicateGraphs() {
    for (size_t numberGraph = 1; numberGraph < this->numberGraphs(); ++numberGraph) {
      auto graphDuplicate =
          std::dynamic_pointer_cast<CoreGraph<GraphOutput, GraphInputs...>>(this->baseCoreGraph()->clone());
      graphDuplicate->graphId(numberGraph);
      graphDuplicate->deviceId(this->deviceIds_[numberGraph]);
      connectGraphToEP(graphDuplicate);
    }
  }

  /// @brief Add data and notification link between switch and one inside graph
  /// @tparam GraphInput Graph input type
  /// @param graph graph to link
  template<class GraphInput>
  void addEdgeSwitchGraph(std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> &graph) {
    auto coreSender = std::static_pointer_cast<CoreSender<GraphInput>>(this->coreSwitch_);
    auto coreNotifier = std::static_pointer_cast<CoreNotifier>(coreSender);
    auto coreSlot = std::static_pointer_cast<CoreSlot>(graph);
    auto coreReceiver = std::static_pointer_cast<CoreReceiver<GraphInput>>(graph);

    for (auto r : coreReceiver->receivers()) { coreSender->addReceiver(r); }
    for (auto s : coreSender->getSenders()) {
      coreReceiver->addSender(s);
      coreSlot->addNotifier(s);
    }
    for (CoreSlot *slot : coreSlot->getSlots()) { coreNotifier->addSlot(slot); }
  }

  /// @brief Connect a graph to the switch and register it
  /// @param coreGraph Graph to connect to the execution pipeline
  void connectGraphToEP(std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> &coreGraph) {
    coreGraph->setInside();
    coreGraph->belongingNode(this);
    coreGraph->hasBeenRegistered(true);
    (addEdgeSwitchGraph<GraphInputs>(coreGraph), ...);
    this->epGraphs_.push_back(coreGraph);
  }

  /// @brief Connect a CoreReceiver to all output of a graph
  /// @param graph CoreGraph to be connected
  /// @param coreReceiver Receiver to connect
  void connectGraphsOutputToReceiver(CoreGraph<GraphOutput, GraphInputs...> *graph,
                                     CoreReceiver<GraphOutput> *coreReceiver) {
    for (CoreReceiver<GraphOutput> *receiver : coreReceiver->receivers()) { graph->addReceiver(receiver); }
  }

  /// @brief Add the graph's senders to the super set
  /// @param superSet Set where to merge the graph's sender
  /// @param graphSenders Senders to add to the set
  void mergeSenders(std::set<CoreSender<GraphOutput> *
  > &superSet, std::set<CoreSender<GraphOutput> *> &graphSenders) {
    for (CoreSender<GraphOutput> *sender : graphSenders) { superSet.insert(sender); }
  }

  /// @brief Print an edge for GraphInput input from the switch to all graph's input node
  /// @tparam GraphInput Edge Graph input type to print
  /// @param printer Printer object to visit the execution pipeline and print the edge
  /// @param graph Graph to take the node's from
  template<class GraphInput>
  void printEdgeSwitchGraphs(AbstractPrinter *printer, CoreGraph<GraphOutput, GraphInputs...> *graph) {
    for (CoreNode *graphInputNode : *(graph->inputsCoreNodes())) {
      if (auto *coreQueueReceiver = dynamic_cast<CoreQueueReceiver<GraphInput> *>(graphInputNode)) {
        printer->printEdgeSwitchGraphs(coreQueueReceiver,
                                       this->id(),
                                       traits::type_name<GraphInput>(),
                                       coreQueueReceiver->queueSize(),
                                       coreQueueReceiver->maxQueueSize(),
                                       traits::is_managed_memory_v<GraphInput>);
      }
    }
  }
};

}
}
#endif //HEDGEHOG_CORE_EXECUTION_PIPELINE_H
