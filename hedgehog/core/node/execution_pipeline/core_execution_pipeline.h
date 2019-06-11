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

  CoreGraph<GraphOutput, GraphInputs...> *
      baseCoreGraph_ = nullptr;

  std::vector<std::shared_ptr<CoreGraph<GraphOutput, GraphInputs...>>>
      epGraphs_ = {};

  std::vector<int> deviceIds_ = {};

  CoreSwitch<GraphInputs...> *
      coreSwitch_ = nullptr;

  CoreExecutionPipeline() = delete;
  CoreExecutionPipeline(std::string_view const &name,
                        AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline,
                        std::shared_ptr<Graph<GraphOutput, GraphInputs...>> baseGraph,
                        size_t numberGraphs,
                        std::vector<int> const &deviceIds,
                        bool automaticStart)
      : CoreTask<GraphOutput, GraphInputs...>(name, 1, NodeType::ExecutionPipeline, nullptr, automaticStart),
        executionPipeline_(executionPipeline),
        numberGraphs_(numberGraphs),
        baseCoreGraph_(dynamic_cast<CoreGraph<GraphOutput, GraphInputs...> *>(baseGraph->core())),
        deviceIds_(deviceIds),
        coreSwitch_(new CoreSwitch<GraphInputs...>("switch", NodeType::Switch, 1)) {

    if (this->numberGraphs_ == 0) { this->numberGraphs_ = 1; }
    if (this->baseCoreGraph_->isInside() || this->isInside()) {
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

    baseCoreGraph_->setInside();
    baseCoreGraph_->graphId(0);

    epGraphs_.reserve(this->numberGraphs_ - 1);

    this->addUniqueInsideNode(baseGraph);

    // TODO: Create copies and loop
    connectGraphToEP(this->baseCoreGraph_);

  }

  virtual ~CoreExecutionPipeline() {
    delete coreSwitch_;
  };

  Node *node() override { return this->executionPipeline_; }

  std::set<CoreSender<GraphOutput> *> getSenders() override {
    std::set<CoreSender<GraphOutput> *>
        res;

    std::set<CoreSender<GraphOutput> *> senders = this->baseCoreGraph_->getSenders();
    mergeSenders(res, senders);

    for (auto epGraph : this->epGraphs_) {
      senders.clear();
      senders = epGraph->getSenders();
      mergeSenders(res, senders);
    }

    return res;
  }

  void addReceiver(CoreReceiver<GraphOutput> *receiver) override {
    connectGraphsOutputToReceiver(baseCoreGraph_, receiver);

    for (auto epGraph : this->epGraphs_) {
      connectGraphsOutputToReceiver(epGraph.get(), receiver);
    }
  }

  void addSlot(CoreSlot *slot) override {
    this->baseCoreGraph_->addSlot(slot);

    for (auto graph : this->epGraphs_) {
      graph->addSlot(slot);
    }
  }

  void visit(AbstractPrinter *printer) override {
    if (printer->hasNotBeenVisited(this)) {
      printer->printExecutionPipelineHeader(this->name(), CoreNode::id(), this->id());
      (this->printEdgeSwitchGraphs<GraphInputs>(printer, this->baseCoreGraph_), ...);
      for (auto graph: epGraphs_) {
        (this->printEdgeSwitchGraphs<GraphInputs>(printer, graph.get()), ...);
      }
      this->baseCoreGraph_->visit(printer);
      printer->printExecutionPipelineFooter();
    }
  }

  AbstractExecutionPipeline<GraphOutput, GraphInputs...> *executionPipeline() const {
    return executionPipeline_;
  }

  std::string id() const override {
    return "switch" + CoreNode::id();
  }

  size_t deviceId() override {
    HLOG_SELF(0, "ERROR: DeviceId called for an execution pipeline!: " << __PRETTY_FUNCTION__)
    exit(42);
  }

  void copyWholeNode(std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>> &insideNodesGraph) override {
    insideNodesGraph->insert(std::begin(*(this->insideNodes())), std::end(*(this->insideNodes())));
    this->baseCoreGraph_->copyInnerNodesAndLaunchThreads();
    for (auto epGraph : this->epGraphs_) {
      epGraph->copyInnerNodesAndLaunchThreads();
    }
  }

 private:

  template<class GraphInput>
  void addEdge(CoreGraph<GraphOutput, GraphInputs...> *graph) {
    auto coreSender = static_cast<CoreSender<GraphInput> *>(this->coreSwitch_);
    auto coreNotifier = static_cast<CoreNotifier *>(coreSender);

    auto coreSlot = static_cast<CoreSlot *>(graph);
    auto coreReceiver = static_cast<CoreReceiver<GraphInput> *>(graph);

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

  void connectGraphToEP(CoreGraph<GraphOutput, GraphInputs...> *graph) {
    (addEdge<GraphInputs>(graph), ...);
  }

  template<class GraphInput>
  void connectSwitchSender(CoreSwitchSender<GraphInput> *sender, CoreReceiver<GraphInput> *receiver) {
    receiver->addSender(sender);
    dynamic_cast<CoreMultiReceivers<GraphInputs...> *>(receiver)->addNotifier(sender);
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
