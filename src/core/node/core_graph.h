//
// Created by 775backup on 2019-04-08.
//

#ifndef HEDGEHOG_CORE_GRAPH_H
#define HEDGEHOG_CORE_GRAPH_H

#include <ostream>
#include <vector>

#include "../node/core_node.h"

#include "../io/base/sender/core_notifier.h"
#include "../io/base/receiver/core_multi_receivers.h"

#include "../../tools/traits.h"
#include "../../tools/helper.h"
#include "../../tools/scheduler/default_scheduler.h"
#include "../scheduler/abstract_scheduler.h"
#include "../io/graph/receiver/core_graph_multi_receivers.h"
#include "../io/graph/receiver/core_graph_sink.h"
#include "../io/graph/sender/core_graph_source.h"

template<class GraphOutput, class ...GraphInputs>
class Graph;

template<class GraphOutput, class ...GraphInputs>
class CoreGraph : public CoreSender<GraphOutput>, public CoreGraphMultiReceivers<GraphInputs...> {
 private:
  Graph<GraphOutput, GraphInputs...> *graph_ = nullptr;
  std::unique_ptr<std::set<CoreMultiReceivers<GraphInputs...> *>> inputsCoreNodes_ = nullptr;
  std::unique_ptr<std::set<CoreSender<GraphOutput> *>> outputCoreNodes_ = nullptr;
  std::unique_ptr<AbstractScheduler> scheduler_ = nullptr;
  std::shared_ptr<CoreGraphSource<GraphInputs...>> source_ = nullptr;
  std::shared_ptr<CoreGraphSink<GraphOutput>> sink_ = nullptr;

 public:
  CoreGraph(Graph<GraphOutput, GraphInputs...> *graph, NodeType const type, std::string_view const &name)
      :
      CoreNode(name, type, 1),
      CoreNotifier(name, type, 1),
      CoreSlot(name, type, 1),
      CoreReceiver<GraphInputs>(name, type, 1)...,
      CoreSender<GraphOutput>(name, type,
  1),
  CoreGraphMultiReceivers<GraphInputs...>(name, type,
  1){
    HLOG_SELF(0, "Creating CoreGraph with graph: " << graph << " type: " << (int) type << " and name: " << name)
    this->graph_ = graph;
    this->inputsCoreNodes_ = std::make_unique<std::set<CoreMultiReceivers<GraphInputs...> *>>();
    this->outputCoreNodes_ = std::make_unique<std::set<CoreSender < GraphOutput> *>>
    ();
    //Todo Create default Scheduler, make it available as parameter of the graph
    this->scheduler_ = std::make_unique<DefaultScheduler>();
    this->source_ = std::make_shared<CoreGraphSource<GraphInputs...>>();
    this->sink_ = std::make_shared<CoreGraphSink<GraphOutput>>();
    this->registerNode(std::static_pointer_cast<Node>(this->sink_));
    this->registerNode(std::static_pointer_cast<Node>(this->source_));
  }

  ~CoreGraph() override {
    HLOG_SELF(0, "Destructing CoreGraph")
    this->graph_ = nullptr;
  }

  Node *getNode() override {
    return graph_;
  }

  std::chrono::duration<uint64_t, std::micro> const maxExecutionTime() const override {
    std::chrono::duration<uint64_t, std::micro> ret = std::chrono::duration<uint64_t, std::micro>::min(), temp{};
    CoreNode *core;
    for (auto const &it : *(this->insideNodes())) {
      core = it.second->getCore();
      switch (core->type()) {
        case NodeType::Task:
        case NodeType::Graph:temp = core->maxExecutionTime();
          if (temp > ret) ret = temp;
          break;
        default:break;
      }
    }
    return ret;
  };

  std::chrono::duration<uint64_t, std::micro> const minExecutionTime() const override {
    std::chrono::duration<uint64_t, std::micro> ret = std::chrono::duration<uint64_t, std::micro>::max(), temp{};
    CoreNode *core;
    for (auto const &it : *(this->insideNodes())) {
      core = it.second->getCore();
      switch (core->type()) {
        case NodeType::Task:
        case NodeType::Graph:temp = core->minExecutionTime();
          if (temp < ret) ret = temp;
          break;
        default:break;
      }
    }
    return ret;
  };

  std::chrono::duration<uint64_t, std::micro> const maxWaitTime() const override {
    std::chrono::duration<uint64_t, std::micro> ret = std::chrono::duration<uint64_t, std::micro>::min(), temp{};
    CoreNode *core;
    for (auto const &it : *(this->insideNodes())) {
      core = it.second->getCore();
      switch (core->type()) {
        case NodeType::Task:
        case NodeType::Graph:temp = core->maxWaitTime();
          if (temp > ret) ret = temp;
          break;
        default:break;
      }
    }
    return ret;
  };

  std::chrono::duration<uint64_t, std::micro> const minWaitTime() const override {
    std::chrono::duration<uint64_t, std::micro> ret = std::chrono::duration<uint64_t, std::micro>::max(), temp{};
    CoreNode *core;
    for (auto const &it : *(this->insideNodes())) {
      core = it.second->getCore();
      switch (core->type()) {
        case NodeType::Task:
        case NodeType::Graph:temp = core->minWaitTime();
          if (temp < ret) ret = temp;
          break;
        default:break;
      }
    }
    return ret;
  };

  template<
      class UserDefinedSender, class UserDefinedMultiReceiver,
      class Output = typename UserDefinedSender::output_t,
      class Inputs = typename UserDefinedMultiReceiver::inputs_t,
      class IsSender = typename std::enable_if<
          std::is_base_of_v<
              Sender<Output>, UserDefinedSender
          >
      >::type,
      class IsMultiReceiver = typename std::enable_if<
          std::is_base_of_v<
              typename Helper::HelperMultiReceiversType<Inputs>::type, UserDefinedMultiReceiver
          >
      >::type
  >
  void addEdge(std::shared_ptr<UserDefinedSender> from, std::shared_ptr<UserDefinedMultiReceiver> to) {
    assert(from != nullptr && to != nullptr);
    if (this->isInside()) {
      HLOG_SELF(0, "You can't play with an inner graph!")
      exit(42);
    }
    static_assert(HedgehogTraits::contains_v<Output, Inputs>, "The given Receiver cannot be linked to this Sender");

    //Get the associated cores
    auto coreSender = dynamic_cast<CoreSender <Output> *>(std::static_pointer_cast<Node>(from)->getCore());
    auto coreNotifier = dynamic_cast<CoreNotifier *>(coreSender);
    auto coreSlot = dynamic_cast<CoreSlot *>(std::static_pointer_cast<Node>(to)->getCore());
    auto coreReceiver = dynamic_cast<CoreReceiver<Output> *>(std::static_pointer_cast<Node>(to)->getCore());

    if (from->getCore() == this || to->getCore() == this) {
      HLOG_SELF(0, "You can't connectMemoryManager the graph to itself!")
      exit(42);
    }

    if (coreSender->hasBeenRegistered()) {
      if (coreSender->belongingNode() != this) {
        HLOG_SELF(0, "The Sender node should belong to the graph.")
        exit(42);
      }
    }

    if (coreReceiver->hasBeenRegistered()) {
      if (coreReceiver->belongingNode() != this) {
        HLOG_SELF(0, "The Receiver node should belong to the graph.")
        exit(42);
      }
    }

    HLOG_SELF(0,
              "Add edge from " << coreSender->name() << "(" << coreSender->id() << ") to " << coreReceiver->name()
                               << "(" << coreReceiver->id()
                               << ")")

    for (auto r : coreReceiver->getReceivers()) {
      coreSender->addReceiver(r);
    }
    for (auto s : coreSender->getSenders()) {
      coreReceiver->addSender(s);
      coreSlot->addNotifier(s);
    }

    for (CoreSlot *slot : coreSlot->getSlots()) {
      coreNotifier->addSlot(slot);
    }

    this->registerNode(std::dynamic_pointer_cast<Node>(from));
    this->registerNode(std::dynamic_pointer_cast<Node>(to));

  }

  void input(std::shared_ptr<MultiReceivers<GraphInputs...>> inputNode) {
    if (this->isInside()) {
      HLOG_SELF(0, "You can't play with an inner graph!")
      exit(42);
    }

    auto inputCoreNode = dynamic_cast<CoreMultiReceivers<GraphInputs...> *>(inputNode->getCore());
    HLOG_SELF(0, "Set " << inputCoreNode->name() << "(" << inputCoreNode->id() << ") as input")

    if (inputCoreNode->hasBeenRegistered()) {
      if (inputCoreNode->belongingNode() != this) {
        HLOG_SELF(0, "The node " << inputCoreNode->name() << " belong already to another graph!")
        exit(42);
      }
    }

    //Add it as input of the graph
    this->inputsCoreNodes_->insert(inputCoreNode);
    //Add it as input for each graph receiver
    (static_cast<CoreGraphReceiver<GraphInputs> *>(this)->addGraphReceiverInput(static_cast<CoreReceiver<GraphInputs> *>(inputCoreNode)), ...);

    this->source_->addSlot(inputCoreNode);
    (inputCoreNode->addNotifier(std::static_pointer_cast<CoreTaskSender<GraphInputs>>(this->source_).get()), ...);

    (std::static_pointer_cast<CoreTaskSender<GraphInputs>>(this->source_)->addReceiver(static_cast<CoreReceiver<
        GraphInputs> *>(inputCoreNode)), ...);
    (static_cast<CoreReceiver<GraphInputs> *>(inputCoreNode)->addSender(static_cast<CoreSender <GraphInputs> *>(this->source_.get())), ...);

    this->registerNode(std::static_pointer_cast<Node>(inputNode));
  }

  void output(std::shared_ptr<Sender<GraphOutput>> output) {
    if (this->isInside()) {
      HLOG_SELF(0, "You can't play with an inner graph!")
      exit(42);
    }

    auto outputCoreNode = dynamic_cast<CoreSender <GraphOutput> *>(output->getCore());

    HLOG_SELF(0, "Set " << outputCoreNode->name() << "(" << outputCoreNode->id() << ") as output")

    if (outputCoreNode->hasBeenRegistered()) {
      if (outputCoreNode->belongingNode() != this) {
        HLOG_SELF(0, "The node " << outputCoreNode->name() << " belong already to another graph!")
        exit(42);
      }
    }

    this->outputCoreNodes_->insert(outputCoreNode);

    for (CoreSender <GraphOutput> *sender : outputCoreNode->getSenders()) {
      this->sink_->addNotifier(sender);
      this->sink_->addSender(sender);
    }

    outputCoreNode->addSlot(this->sink_.get());
    outputCoreNode->addReceiver(this->sink_.get());

    this->registerNode(std::static_pointer_cast<Node>(output));
  }

  template<
      class Input,
      class = std::enable_if_t<HedgehogTraits::Contains<Input, GraphInputs...>::value>
  >
  void broadcastAndNotifyToAllInputs(std::shared_ptr<Input> &data) {
    HLOG_SELF(2, "Broadcast data and notify all graph's inputs")
    if (this->isInside()) {
      HLOG_SELF(0, "You can't play with an inner graph!")
      exit(42);
    }
    std::static_pointer_cast<CoreTaskSender<Input>>(this->source_)->sendAndNotify(data);
  }

  void setInside() override {
    HLOG_SELF(0, "Set the graph inside")
    CoreNode::setInside();

    for (CoreMultiReceivers<GraphInputs...> *inputNode: *(this->inputsCoreNodes_)) {
      (
          static_cast<CoreSlot *>(inputNode)
              ->removeNotifier(
                  static_cast<CoreNotifier *>(
                      static_cast<CoreTaskSender<GraphInputs> *>(this->source_.get())
                  )
              ), ...);
      (static_cast<CoreReceiver<GraphInputs> *>(inputNode)->removeSender(static_cast<CoreTaskSender<GraphInputs> *>(this->source_.get())), ...);
    }

    std::for_each(this->outputCoreNodes_->begin(),
                  this->outputCoreNodes_->end(),
                  [this](CoreSender <GraphOutput> *s) {
                    s->removeSlot(this->sink_.get());
                    s->removeReceiver(this->sink_.get());
                  });

    this->removeInsideNode(this->source_->id());
    this->removeInsideNode(this->sink_->id());
    this->source_ = nullptr;
    this->sink_ = nullptr;
  }

  std::vector<std::pair<std::string, std::string>> ids() const final {
    std::vector<std::pair<std::string, std::string>> v{};
    for (auto input : *(this->inputsCoreNodes_)) {
      for (std::pair<std::string, std::string> const &innerInput : input->ids()) {
        v.push_back(innerInput);
      }
    }
    return v;
  }

  void executeGraph() {
    HLOG_SELF(2, "Execute the graph")
    if (this->isInside()) {
      HLOG_SELF(0, "You can not invoke the method executeGraph with an inner graph.")
      exit(42);
    }
    this->startExecutionTimeStamp(std::chrono::high_resolution_clock::now());
    copyInnerNodesAndLaunchThreads();
    auto finishCreationTime = std::chrono::high_resolution_clock::now();
    this->creationDuration(std::chrono::duration_cast<std::chrono::microseconds>(
        finishCreationTime - this->creationTimeStamp()));
  }

  void waitForTermination() {
    HLOG_SELF(2, "Wait for the graph to terminate")
    this->scheduler_->joinAll();
    std::chrono::time_point<std::chrono::high_resolution_clock>
        endExecutionTimeStamp = std::chrono::high_resolution_clock::now();
    this->executionDuration(std::chrono::duration_cast<std::chrono::microseconds>
                                (endExecutionTimeStamp - this->startExecutionTimeStamp()));
  }

  void finishPushingData() {
    HLOG_SELF(2, "Indicate finish pushing data")
    if (this->isInside()) {
      HLOG_SELF(0, "You can not invoke the method executeGraph with an inner graph.")
      exit(42);
    }
    this->source_->notifyAllTerminated();
  }

  std::shared_ptr<GraphOutput> getBlockingResult() {
    HLOG_SELF(2, "Get blocking data")

    if (this->isInside()) {
      HLOG_SELF(0, "You can not invoke the method executeGraph with an inner graph.")
      exit(42);
    }

    std::shared_ptr<GraphOutput> result = nullptr;

    this->sink_->waitForNotification();

    this->sink_->lockUniqueMutex();
    if (!this->sink_->receiverEmpty()) { result = this->sink_->popFront(); }

    this->sink_->unlockUniqueMutex();
    return result;
  }

  void copyWholeNode([[maybe_unused]]std::shared_ptr<std::multimap<std::string,
                                                                   std::shared_ptr<Node>>> &insideNodesGraph) override {

    copyInnerNodesAndLaunchThreads();
  }

  friend std::ostream &operator<<(std::ostream &os, CoreGraph const &core) {
    os << " insideNodes_: " << std::endl;

    for (auto &key : *(core.insideNodes_)) {
      os << key.first << " : " << *(key.second.get()->getCore()) << std::endl;
    }
    return os;
  }

  void printCluster(AbstractPrinter *printer, CoreNode const *node) {
    printer->printClusterHeader(node->clusterId());
    auto id = node->id();
    for (std::multimap<std::string, std::shared_ptr<Node>>::iterator it = this->insideNodes()->equal_range(id).first;
         it != this->insideNodes()->equal_range(id).second; ++it) {
      printer->printClusterEdge(it->second->getCore());
      it->second->getCore()->visit(printer);
    }
    printer->printClusterFooter();
  }

  void visit(AbstractPrinter *printer) override {
    HLOG_SELF(1, "Visit")
    //Test if the graph has already been visited
    if (printer->hasNotBeenVisited(this)) {

//      Print the header of the graph
      printer->printGraphHeader(this);

      //      Print the graph information
      printer->printNodeInformation(this);

      //      Visit all the insides node of the graph
      for (std::multimap<std::string, std::shared_ptr<Node>>::const_iterator it = this->insideNodes()->begin(),
               end = this->insideNodes()->end(); it != end; it = this->insideNodes()->upper_bound(it->first)) {
        if (this->insideNodes()->count(it->first) == 1) {
          it->second->getCore()->visit(printer);
        } else {
          this->printCluster(printer, it->second->getCore());
        }
      }
//      Print graph footer
      printer->printGraphFooter(this);
    }
  }

  //Virtual functions
  //Sender
  void addReceiver(CoreReceiver<GraphOutput> *receiver) override {
    HLOG_SELF(0, "Add receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (CoreSender <GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
      outputNode->addReceiver(receiver);
    }
  }

  void removeReceiver(CoreReceiver<GraphOutput> *receiver) override {
    HLOG_SELF(0, "Remove receiver " << receiver->name() << "(" << receiver->id() << ")")
    for (CoreSender <GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
      outputNode->addReceiver(receiver);
    }
  }

  void sendAndNotify([[maybe_unused]]std::shared_ptr<GraphOutput> ptr) override {
    HLOG_SELF(0, "Shouldn't been called yet ?")
    exit(42);
  }

  //Notifier
  void addSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Add Slot " << slot->name() << "(" << slot->id() << ")")
    for (CoreSender <GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
      outputNode->addSlot(slot);
    }
  }

  void removeSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Remove Slot " << slot->name() << "(" << slot->id() << ")")
    for (CoreSender <GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
      outputNode->removeSlot(slot);
    }
  }

  void notifyAllTerminated() override {
    HLOG_SELF(0, "Shouldn't been called yet ?")
    exit(42);
  }

  void addNotifier(CoreNotifier *notifier) override {
    HLOG_SELF(0, "Add Notifier " << notifier->name() << "(" << notifier->id() << ")")
    for (CoreMultiReceivers<GraphInputs...> *inputNode: *(this->inputsCoreNodes_)) {
      inputNode->addNotifier(notifier);
    }
  }

  void removeNotifier(CoreNotifier *notifier) override {
    HLOG_SELF(0, "Remove Notifier " << notifier->name() << "(" << notifier->id() << ")")
    for (CoreMultiReceivers<GraphInputs...> *inputNode: *(this->inputsCoreNodes_)) {
      inputNode->removeNotifier(notifier);
    }
  }

  bool hasNotifierConnected() override {
    HLOG_SELF(0, "Shouldn't been called yet ?")
    exit(42);
  }

  size_t numberInputNodes() const override {
    HLOG_SELF(0, "Shouldn't been called yet ?")
    exit(42);
  }

  void wakeUp() override {
    HLOG_SELF(2, "Wake up all inputs")
    for (CoreMultiReceivers<GraphInputs...> *inputNode: *(this->inputsCoreNodes_)) {
      inputNode->wakeUp();
    }
  }

  void waitForNotification() override {
    HLOG_SELF(0, "Shouldn't been called yet ?")
    exit(42);
  }

  std::set<CoreSender < GraphOutput>*>
  getSenders() override {
    std::set<CoreSender < GraphOutput>*> coreSenders;
    std::set<CoreSender < GraphOutput>*> tempCoreSenders;

    for (CoreSender <GraphOutput> *outputNode : *(this->outputCoreNodes_)) {
      tempCoreSenders = outputNode->getSenders();
      coreSenders.insert(tempCoreSenders.begin(), tempCoreSenders.end());
    }
    return coreSenders;
  }

  std::set<CoreSlot *> getSlots() override {
    std::set<CoreSlot *> coreSlots;
    std::set<CoreSlot *> tempCoreSlots;

    for (CoreMultiReceivers<GraphInputs...> *mr : *(this->inputsCoreNodes_)) {
      tempCoreSlots = mr->getSlots();
      coreSlots.insert(tempCoreSlots.begin(), tempCoreSlots.end());
    }
    return coreSlots;
  }

  void joinThreads() override {
    HLOG_SELF(2, "Join graph threads")
    this->scheduler_->joinAll();
  }

 protected:
  void copyInnerNodesAndLaunchThreads() {
    HLOG_SELF(0, "Copy inside nodes")
    std::vector<std::shared_ptr<Node>> copyNodes;

    for (const std::pair<std::string, std::shared_ptr<Node>> &insideNodes : *(this->insideNodes().get())) {
      copyNodes.push_back(insideNodes.second);
    }

    for (std::shared_ptr<Node> const &node : copyNodes) {
      CoreNode *coreNode = node->getCore();
      coreNode->copyWholeNode(this->insideNodes());
    }

    this->scheduler_->spawnThreads(this->insideNodes());
  }

 private:

  void registerNode(const std::shared_ptr<Node> &node) {
    HLOG_SELF(0, "Register node " << node->getCore()->name() << "(" << node->getCore()->id() << ")")
    if (!node->getCore()->hasBeenRegistered()) {
      node->getCore()->setInside();
      this->addUniqueInsideNode(node);
    }
  }

};

#endif //HEDGEHOG_CORE_GRAPH_H
