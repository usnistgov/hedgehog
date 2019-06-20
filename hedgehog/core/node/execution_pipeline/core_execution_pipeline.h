#include <utility>

//
// Created by anb22 on 6/10/19.
//

#ifndef HEDGEHOG_CORE_EXECUTION_PIPELINE_H
#define HEDGEHOG_CORE_EXECUTION_PIPELINE_H

#include "core_switch.h"
#include "../core_task.h"
#include "../../../api/graph.h"

template<class GraphOutput, class ...GraphInputs>
class AbstractExecutionPipeline;

template<class GraphOutput, class ...GraphInputs>
class CoreExecutionPipeline : public CoreTask<GraphOutput, GraphInputs...> {
 public:
  AbstractExecutionPipeline<GraphOutput, GraphInputs...> *
      executionPipeline_ = nullptr;

  size_t
      numberGraphs_ = 0;

  std::vector<std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>>>
      epGraphs_ = {};

  std::vector<int> deviceIds_ = {};

  std::shared_ptr<CoreSwitch<GraphInputs...>>
      coreSwitch_;

  CoreExecutionPipeline() = delete;
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
        coreSwitch_(std::make_shared<CoreSwitch<GraphInputs...>>("switch", NodeType::Switch, 1)) {
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!Begin creation EP!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (this->numberGraphs_ == 0) { this->numberGraphs_ = 1; }

    if (coreBaseGraph->isInside() || this->isInside()) {
      HLOG_SELF(0, "You can't play with an inner Graph!")
      exit(42);
    }
    if (numberGraphs_ != deviceIds.size()) {
      std::ostringstream oss;
      oss
          << "The number of device Id do not correspond to the number of coreGraph duplication you ask for the execution pipeline \""
          << name
          << "\". Even if you do not associate the gr+aphs duplicates to a specific device, please set the deviceIds correctly."
          << std::endl;

      std::cerr << oss.str();
      HLOG_SELF(0, "ERROR: CoreExecutionPipeline " << __PRETTY_FUNCTION__ << " " << oss.str())
      exit(42);
    }
    epGraphs_.reserve(this->numberGraphs_);

    coreBaseGraph->graphId(0);
    connectGraphToEP(coreBaseGraph);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    std::cout << "SWITCH: " << std::endl;
//    std::cout << "\t" << *(this->coreSwitch_) << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    this->duplicateGraphs();
  }

  virtual ~CoreExecutionPipeline() = default;

  Node *node() override { return this->executionPipeline_; }

  std::vector<int> const &deviceIds() const {
    return deviceIds_;
  }

  size_t numberGraphs() const {
    return numberGraphs_;
  }

  std::set<CoreSender<GraphOutput> *> getSenders() override {
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

  std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> baseCoreGraph() {
    return this->epGraphs_.empty() ? nullptr : this->epGraphs_[0];
  }

  void addReceiver(CoreReceiver<GraphOutput> *receiver) override {
    for (auto epGraph : this->epGraphs_) { connectGraphsOutputToReceiver(epGraph.get(), receiver); }
  }

  void addSlot(CoreSlot *slot) override {
    for (auto graph : this->epGraphs_) { graph->addSlot(slot); }
  }

  void visit(AbstractPrinter *printer) override {
    if (printer->hasNotBeenVisited(this)) {
      printer->printExecutionPipelineHeader(this->name(), CoreNode::id(), this->id());
      for (auto graph: epGraphs_) {
        (this->printEdgeSwitchGraphs<GraphInputs>(printer, graph.get()), ...);
        graph->visit(printer);
      }
      printer->printExecutionPipelineFooter();
    }
  }

  AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline() const {
    return executionPipeline_;
  }

  std::string id() const override {
    return "switch" + CoreNode::id();
  }

  int deviceId() override {
    HLOG_SELF(0, "ERROR: DeviceId called for an execution pipeline!: " << __PRETTY_FUNCTION__)
    exit(42);
  }

  void createCluster([[maybe_unused]]std::shared_ptr<std::multimap<CoreNode *,
                                                                   std::shared_ptr<CoreNode>>> &insideNodesGraph) override {
    for (std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> epGraph : this->epGraphs_) {
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//      using namespace std::chrono_literals;
//      std::this_thread::sleep_for(2s);
//      std::cout << "Spawning threads: " << epGraph->id() << " gid: " << epGraph->graphId() << std::endl;
//      for (std::pair<CoreNode *const, std::shared_ptr<CoreNode>> const &duplicatNode : *(epGraph->insideNodes())) {
//        std::cout << "\t " << duplicatNode.second->name() << " / " << duplicatNode.second->id() << std::endl;
//      }
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      epGraph->createInnerClustersAndLaunchThreads();
    }
  }


 private:
  void duplicateGraphs() {
    for (size_t numberGraph = 1; numberGraph < this->numberGraphs(); ++numberGraph) {
      auto graphDuplicate =
          std::dynamic_pointer_cast<CoreGraph<GraphOutput, GraphInputs...>>(this->baseCoreGraph()->clone());
      graphDuplicate->graphId(numberGraph);
      graphDuplicate->deviceId(this->deviceIds_[numberGraph]);
      connectGraphToEP(graphDuplicate);
    }
  }

  template<class GraphInput>
  void addEdgeSwitchGraph(std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> &graph) {
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    std::cout << "Connect switch " <<this->coreSwitch_->id() << " to graph " << graph->id() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto coreSender = std::static_pointer_cast<CoreSender<GraphInput>>(this->coreSwitch_);
    auto coreNotifier = std::static_pointer_cast<CoreNotifier>(coreSender);

    auto coreSlot = std::static_pointer_cast<CoreSlot>(graph);
    auto coreReceiver = std::static_pointer_cast<CoreReceiver<GraphInput>>(graph);

    for (auto r : coreReceiver->receivers()) {
      coreSender->addReceiver(r);
    }

    for (auto s : coreSender->getSenders()) {
      coreReceiver->addSender(s);
      coreSlot->addNotifier(s);
    }

    for (CoreSlot *slot : coreSlot->getSlots()) {
      coreNotifier->addSlot(slot);
    }
  }

  void connectGraphToEP(std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>> &coreGraph) {
    coreGraph->setInside();
    coreGraph->belongingNode(this);
    coreGraph->hasBeenRegistered(true);
    (addEdgeSwitchGraph<GraphInputs>(coreGraph), ...);
    this->epGraphs_.push_back(coreGraph);
  }

  void connectGraphsOutputToReceiver(CoreGraph<GraphOutput, GraphInputs...> *graph,
                                     CoreReceiver<GraphOutput> *coreReceiver) {
    for (CoreReceiver<GraphOutput> *receiver : coreReceiver->receivers()) {
      graph->addReceiver(receiver);
    }
  }

  void mergeSenders(std::set<CoreSender<GraphOutput> *> &superSet, std::set<CoreSender<GraphOutput> *> &graphSenders) {
    for (CoreSender<GraphOutput> *sender : graphSenders) {
      superSet.insert(sender);
    }
  }

  template<class GraphInput>
  void printEdgeSwitchGraphs(AbstractPrinter *printer, CoreGraph<GraphOutput, GraphInputs...> *graph) {

    for (CoreNode *graphInputNode : *(graph->inputsCoreNodes())) {
      printer->printEdgeSwitchGraphs(graphInputNode,
                                     this->id(),
                                     HedgehogTraits::type_name<GraphInput>());
    }
  }
};

#endif //HEDGEHOG_CORE_EXECUTION_PIPELINE_H
