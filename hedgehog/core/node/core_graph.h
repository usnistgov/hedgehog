//
// Created by 775backup on 2019-04-08.
//

#ifndef HEDGEHOG_CORE_GRAPH_H
#define HEDGEHOG_CORE_GRAPH_H

#include <ostream>
#include <vector>
#include <filesystem>

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
#include "../../tools/printers/dot_printer.h"

template<class GraphOutput, class ...GraphInputs>
class Graph;

template<class GraphOutput, class ...GraphInputs>
class CoreGraph : public CoreSender<GraphOutput>, public CoreGraphMultiReceivers<GraphInputs...> {
 private:
  Graph<GraphOutput, GraphInputs...> *graph_ = nullptr;
  std::unique_ptr<std::set<CoreNode *>> inputsCoreNodes_ = nullptr;
  std::unique_ptr<std::set<CoreSender<GraphOutput> *>> outputCoreNodes_ = nullptr;
  std::unique_ptr<AbstractScheduler> scheduler_ = nullptr;
  std::shared_ptr<CoreGraphSource<GraphInputs...>> source_ = nullptr;
  std::shared_ptr<CoreGraphSink<GraphOutput>> sink_ = nullptr;
  int graphId_ = 0;
  int deviceId_ = 0;

 public:
  CoreGraph(Graph<GraphOutput, GraphInputs...> *graph, NodeType const type, std::string_view const &name) :
	  CoreNode(name, type, 1),
	  CoreNotifier(name, type, 1),
	  CoreSlot(name, type, 1),
	  CoreReceiver<GraphInputs>(name, type, 1)...,
	  CoreSender<GraphOutput>(name, type, 1),
  CoreGraphMultiReceivers<GraphInputs...>(name, type, 1){
	HLOG_SELF(0, "Creating CoreGraph with coreGraph: " << graph << " type: " << (int) type << " and name: " << name)
	this->graph_ = graph;
	this->inputsCoreNodes_ = std::make_unique<std::set<CoreNode *>>();
	this->outputCoreNodes_ = std::make_unique<std::set<CoreSender < GraphOutput> *>>();
	//Todo Create default Scheduler, make it available as parameter of the coreGraph
	this->scheduler_ = std::make_unique<DefaultScheduler>();
	this->source_ = std::make_shared<CoreGraphSource<GraphInputs...>>();
	this->sink_ = std::make_shared<CoreGraphSink<GraphOutput>>();
	this->source_->belongingNode(this);
	this->sink_->belongingNode(this);
  }

  CoreGraph(CoreGraph<GraphOutput, GraphInputs...> const &rhs) :
	  CoreNode(rhs.name(), rhs.type(), 1),
	  CoreNotifier(rhs.name(), rhs.type(), 1),
	  CoreSlot(rhs.name(), rhs.type(), 1),
	  CoreReceiver<GraphInputs>(rhs.name(), rhs.type(), 1)...,
	  CoreSender<GraphOutput>(rhs.name(), rhs.type(),1),
	  CoreGraphMultiReceivers<GraphInputs...>(rhs.name(), rhs.type(),1){
	this->inputsCoreNodes_ = std::make_unique<std::set<CoreNode *>>();
	this->outputCoreNodes_ = std::make_unique<std::set<CoreSender < GraphOutput> *>>();
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

  ~CoreGraph() override {HLOG_SELF(0, "Destructing CoreGraph")}

  std::shared_ptr<CoreNode> clone() override { return std::make_shared<CoreGraph<GraphOutput, GraphInputs...>>(*this); }

  Node *node() override { return graph_; }

  int deviceId() override { return this->deviceId_; }

  int graphId() override { return this->graphId_; }

  std::chrono::duration<uint64_t, std::micro> const maxExecutionTime() const override {
	std::chrono::duration<uint64_t, std::micro> ret = std::chrono::duration<uint64_t, std::micro>::min(), temp{};
	std::shared_ptr<CoreNode> core;
	for (auto const &it : *(this->insideNodes())) {
	  core = it.second;
	  switch (core->type()) {
		case NodeType::Task:
		case NodeType::Graph:temp = core->maxExecutionTime();
		  if (temp > ret) ret = temp;
		  break;
		default:break;
	  }
	}
	return ret;
  }

  std::chrono::duration<uint64_t, std::micro> const minExecutionTime() const override {
	std::chrono::duration<uint64_t, std::micro> ret = std::chrono::duration<uint64_t, std::micro>::max(), temp{};
	std::shared_ptr<CoreNode> core;
	for (auto const &it : *(this->insideNodes())) {
	  core = it.second;
	  switch (core->type()) {
		case NodeType::Task:
		case NodeType::Graph:temp = core->minExecutionTime();
		  if (temp < ret) ret = temp;
		  break;
		default:break;
	  }
	}
	return ret;
  }

  std::chrono::duration<uint64_t, std::micro> const maxWaitTime() const override {
	std::chrono::duration<uint64_t, std::micro> ret = std::chrono::duration<uint64_t, std::micro>::min(), temp{};
	std::shared_ptr<CoreNode> core;
	for (auto const &it : *(this->insideNodes())) {
	  core = it.second;
	  switch (core->type()) {
		case NodeType::Task:
		case NodeType::Graph:temp = core->maxWaitTime();
		  if (temp > ret) ret = temp;
		  break;
		default:break;
	  }
	}
	return ret;
  }

  std::chrono::duration<uint64_t, std::micro> const minWaitTime() const override {
	std::chrono::duration<uint64_t, std::micro> ret = std::chrono::duration<uint64_t, std::micro>::max(), temp{};
	std::shared_ptr<CoreNode> core;
	for (auto const &it : *(this->insideNodes())) {
	  core = it.second;
	  switch (core->type()) {
		case NodeType::Task:
		case NodeType::Graph:temp = core->minWaitTime();
		  if (temp < ret) ret = temp;
		  break;
		default:break;
	  }
	}
	return ret;
  }

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
	  HLOG_SELF(0, "You can't play with an inner coreGraph!")
	  exit(42);
	}
	static_assert(HedgehogTraits::contains_v<Output, Inputs>, "The given Receiver cannot be linked to this Sender");

	//Get the associated cores
	auto coreSender = dynamic_cast<CoreSender<Output> *>(std::static_pointer_cast<Node>(from)->core().get());
	auto coreNotifier = dynamic_cast<CoreNotifier *>(coreSender);
	auto coreSlot = dynamic_cast<CoreSlot *>(std::static_pointer_cast<Node>(to)->core().get());
	auto coreReceiver = dynamic_cast<CoreReceiver<Output> *>(std::static_pointer_cast<Node>(to)->core().get());

	if (from->core().get() == this || to->core().get() == this) {
	  HLOG_SELF(0, "You can't connectMemoryManager the coreGraph to itself!")
	  exit(42);
	}

	if (coreSender->hasBeenRegistered()) {
	  if (coreSender->belongingNode() != this) {
		HLOG_SELF(0, "The Sender node should belong to the coreGraph.")
		exit(42);
	  }
	}

	if (coreReceiver->hasBeenRegistered()) {
	  if (coreReceiver->belongingNode() != this) {
		HLOG_SELF(0, "The Receiver node should belong to the coreGraph.")
		exit(42);
	  }
	}

	HLOG_SELF(0,
			  "Add edge from " << coreSender->name() << "(" << coreSender->id() << ") to " << coreReceiver->name()
							   << "(" << coreReceiver->id()
							   << ")")

	for (auto r : coreReceiver->receivers()) { coreSender->addReceiver(r); }
	for (CoreSlot *slot : coreSlot->getSlots()) { coreNotifier->addSlot(slot); }
	for (auto s : coreSender->getSenders()) { coreReceiver->addSender(s); coreSlot->addNotifier(s); }

	this->registerNode(std::dynamic_pointer_cast<CoreNode>(from->core()));
	this->registerNode(std::dynamic_pointer_cast<CoreNode>(to->core()));
  }

  template<
	  class UserDefinedMultiReceiver,
	  class InputsMR = typename UserDefinedMultiReceiver::inputs_t,
	  class InputsG = typename MultiReceivers<GraphInputs...>::inputs_t,
	  class isMultiReceiver = typename std::enable_if<
		  std::is_base_of_v<typename Helper::HelperMultiReceiversType<InputsMR>::type, UserDefinedMultiReceiver>
	  >::type,
	  class isInputCompatible = std::enable_if<HedgehogTraits::is_included_v<InputsMR, InputsG>>>
  void input(std::shared_ptr<UserDefinedMultiReceiver> inputNode) {
	if (this->isInside()) {
	  HLOG_SELF(0, "You can't play with an inner coreGraph!")
	  exit(42);
	}

	auto inputCoreNode =
		dynamic_cast<typename Helper::HelperCoreMultiReceiversType<InputsMR>::type *>(inputNode->core().get());
	HLOG_SELF(0, "Set " << inputCoreNode->name() << "(" << inputCoreNode->id() << ") as input")

	if (inputCoreNode->hasBeenRegistered()) {
	  if (inputCoreNode->belongingNode() != this) {
		HLOG_SELF(0, "The node " << inputCoreNode->name() << " belong already to another coreGraph!")
		exit(42);
	  }
	}

	//Add it as input of the coreGraph
	this->inputsCoreNodes_->insert(inputCoreNode);
	this->addReceiversToSource(inputCoreNode);
	this->registerNode(std::static_pointer_cast<CoreNode>(inputNode->core()));
  }

  void output(std::shared_ptr<Sender<GraphOutput>> outputNode) {
	if (this->isInside()) {
	  HLOG_SELF(0, "You can't play with an inner coreGraph!")
	  exit(42);
	}

	auto outputCoreNode = dynamic_cast<CoreSender<GraphOutput> *>(outputNode->core().get());

	HLOG_SELF(0, "Set " << outputCoreNode->name() << "(" << outputCoreNode->id() << ") as outputNode")

	if (outputCoreNode->hasBeenRegistered()) {
	  if (outputCoreNode->belongingNode() != this) {
		HLOG_SELF(0, "The node " << outputCoreNode->name() << " belong already to another coreGraph!")
		exit(42);
	  }
	}

	this->outputCoreNodes_->insert(outputCoreNode);

	for (CoreSender<GraphOutput> *sender : outputCoreNode->getSenders()) {
	  this->sink_->addNotifier(sender);
	  this->sink_->addSender(sender);
	}

	outputCoreNode->addSlot(this->sink_.get());
	outputCoreNode->addReceiver(this->sink_.get());

	this->registerNode(std::static_pointer_cast<CoreNode>(outputNode->core()));
  }

  template<
	  class Input,
	  class = std::enable_if_t<HedgehogTraits::Contains<Input, GraphInputs...>::value>
  >
  void broadcastAndNotifyToAllInputs(std::shared_ptr<Input> &data) {
	HLOG_SELF(2, "Broadcast data and notify all coreGraph's inputs")
	if (this->isInside()) {
	  HLOG_SELF(0, "You can't play with an inner coreGraph!")
	  exit(42);
	}
	std::static_pointer_cast<CoreQueueSender<Input>>(this->source_)->sendAndNotify(data);
  }


  void graphId(size_t graphId) { graphId_ = graphId; }

  void setInside() override {
	assert(!this->isInside());
	HLOG_SELF(0, "Set the coreGraph inside")
	CoreNode::setInside();

	for (CoreNode *inputNode: *(this->inputsCoreNodes_)) {
	  (
		  dynamic_cast<CoreSlot *>(inputNode)->removeNotifier(
			  static_cast<CoreNotifier *>(
				  static_cast<CoreQueueSender<GraphInputs> *>(
					  this->source_.get()
				  )
			  )
		  ), ...);
	  this->removeForAllSenders(inputNode);
	}

	std::for_each(this->outputCoreNodes_->begin(), this->outputCoreNodes_->end(),
				  [this](CoreSender<GraphOutput> *s) {
					s->removeSlot(this->sink_.get());
					s->removeReceiver(this->sink_.get());
				  });

	this->removeInsideNode(this->source_.get());
	this->removeInsideNode(this->sink_.get());
	this->source_ = nullptr;
	this->sink_ = nullptr;
  }

  std::unique_ptr<std::set<CoreNode *>> const &inputsCoreNodes() const { return inputsCoreNodes_; }

  std::vector<std::pair<std::string, std::string>> ids() const final {
	std::vector<std::pair<std::string, std::string>> v{};
	for (auto input : *(this->inputsCoreNodes_)) {
	  for (std::pair<std::string, std::string> const &innerInput : input->ids()) { v.push_back(innerInput); }
	}
	return v;
  }

  void executeGraph() {
	HLOG_SELF(2, "Execute the coreGraph")
	if (this->isInside()) {
	  HLOG_SELF(0, "You can not invoke the method executeGraph with an inner coreGraph.")
	  exit(42);
	}
	this->startExecutionTimeStamp(std::chrono::high_resolution_clock::now());
	createInnerClustersAndLaunchThreads();
	auto finishCreationTime = std::chrono::high_resolution_clock::now();
	this->creationDuration(std::chrono::duration_cast<std::chrono::microseconds>(
		finishCreationTime - this->creationTimeStamp()));
  }

  void waitForTermination() {
	HLOG_SELF(2, "Wait for the coreGraph to terminate")
	this->scheduler_->joinAll();
	std::chrono::time_point<std::chrono::high_resolution_clock>
		endExecutionTimeStamp = std::chrono::high_resolution_clock::now();
	this->executionDuration(std::chrono::duration_cast<std::chrono::microseconds>
								(endExecutionTimeStamp - this->startExecutionTimeStamp()));
  }

  void finishPushingData() {
	HLOG_SELF(2, "Indicate finish pushing data")
	if (this->isInside()) {
	  HLOG_SELF(0, "You can not invoke the method executeGraph with an inner coreGraph.")
	  exit(42);
	}
	this->source_->notifyAllTerminated();
  }

  std::shared_ptr<GraphOutput> getBlockingResult() {
	HLOG_SELF(2, "Get blocking data")
	if (this->isInside()) {
	  HLOG_SELF(0, "You can not invoke the method executeGraph with an inner coreGraph.")
	  exit(42);
	}

	std::shared_ptr<GraphOutput> result = nullptr;
	this->sink_->waitForNotification();
	this->sink_->lockUniqueMutex();
	if (!this->sink_->receiverEmpty()) { result = this->sink_->popFront(); }
	this->sink_->unlockUniqueMutex();
	return result;
  }

  void createCluster(
	  [[maybe_unused]]std::shared_ptr<std::multimap<CoreNode *,
													std::shared_ptr<CoreNode>>> &insideNodesGraph) override {
	createInnerClustersAndLaunchThreads();
  }

  std::unique_ptr<std::set<CoreSender < GraphOutput> *>> const &
  outputCoreNodes() const { return outputCoreNodes_; }

  void printCluster(AbstractPrinter *printer, std::shared_ptr<CoreNode> const &node) {
	printer->printClusterHeader(node->coreClusterNode());
	for (std::multimap<CoreNode *, std::shared_ptr<CoreNode>>::iterator
			 it = this->insideNodes()->equal_range(node.get()).first;
		 it != this->insideNodes()->equal_range(node.get()).second; ++it) {
	  printer->printClusterEdge(it->second.get());
	  it->second->visit(printer);
	}
	printer->printClusterFooter();
  }

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
	  for (std::multimap<CoreNode *, std::shared_ptr<CoreNode>>::const_iterator it = this->insideNodes()->begin(),
			   end = this->insideNodes()->end(); it != end; it = this->insideNodes()->upper_bound(it->first)) {
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
  void addReceiver(CoreReceiver<GraphOutput> *receiver) override {
	HLOG_SELF(0, "Add receiver " << receiver->name() << "(" << receiver->id() << ")")
	for (CoreSender<GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
	  outputNode->addReceiver(receiver);
	}
  }

  void removeReceiver(CoreReceiver<GraphOutput> *receiver) override {
	HLOG_SELF(0, "Remove receiver " << receiver->name() << "(" << receiver->id() << ")")
	for (CoreSender<GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
	  outputNode->removeReceiver(receiver);
	}
  }

  void sendAndNotify([[maybe_unused]]std::shared_ptr<GraphOutput> ptr) override {
	HLOG_SELF(0, "Shouldn't been called yet ?")
	exit(42);
  }

  //Notifier
  void addSlot(CoreSlot *slot) override {
	HLOG_SELF(0, "Add Slot " << slot->name() << "(" << slot->id() << ")")
	for (CoreSender<GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
	  outputNode->addSlot(slot);
	}
  }

  void removeSlot(CoreSlot *slot) override {
	HLOG_SELF(0, "Remove Slot " << slot->name() << "(" << slot->id() << ")")
	for (CoreSender<GraphOutput> *outputNode: *(this->outputCoreNodes_)) {
	  outputNode->removeSlot(slot);
	}
  }

  void notifyAllTerminated() override {
	HLOG_SELF(0, "Shouldn't been called yet ?")
	exit(42);
  }

  void addNotifier(CoreNotifier *notifier) override {
	HLOG_SELF(0, "Add Notifier " << notifier->name() << "(" << notifier->id() << ")")
	for (CoreNode *inputNode: *(this->inputsCoreNodes_)) {
	  dynamic_cast<CoreSlot *>(inputNode)->addNotifier(notifier);
	}
  }

  void removeNotifier(CoreNotifier *notifier) override {
	HLOG_SELF(0, "Remove Notifier " << notifier->name() << "(" << notifier->id() << ")")
	for (CoreNode *inputNode: *(this->inputsCoreNodes_)) {
	  dynamic_cast<CoreSlot *>(inputNode)->removeNotifier(notifier);
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
	for (CoreNode *inputNode: *(this->inputsCoreNodes_)) {
	  dynamic_cast<CoreSlot *>(inputNode)->wakeUp();
	}
  }

  bool waitForNotification() override {
	HLOG_SELF(0, "Shouldn't been called yet ?")
	exit(42);
  }

  std::set<CoreSender < GraphOutput>*>
  getSenders() override {
	std::set<CoreSender < GraphOutput>*> coreSenders;
	std::set<CoreSender < GraphOutput>*> tempCoreSenders;

	for (CoreSender<GraphOutput> *outputNode : *(this->outputCoreNodes_)) {
	  tempCoreSenders = outputNode->getSenders();
	  coreSenders.insert(tempCoreSenders.begin(), tempCoreSenders.end());
	}

	return coreSenders;
  }

  std::set<CoreSlot *> getSlots() override {
	std::set<CoreSlot *> coreSlots;
	std::set<CoreSlot *> tempCoreSlots;

	for (CoreNode *mr : *(this->inputsCoreNodes_)) {
	  tempCoreSlots = mr->getSlots();
	  coreSlots.insert(tempCoreSlots.begin(), tempCoreSlots.end());
	}
	return coreSlots;
  }

  void joinThreads() override {
	HLOG_SELF(2, "Join coreGraph threads")
	this->scheduler_->joinAll();
  }

  void deviceId(int deviceId) override {
    this->deviceId_ = deviceId;
  }

  void createInnerClustersAndLaunchThreads() {
	HLOG_SELF(0, "Cluster creation")
	std::vector<std::shared_ptr<CoreNode>> insideCoreNodes;
	insideCoreNodes.reserve(this->insideNodes()->size());
	for (auto coreNode : *(this->insideNodes())) { insideCoreNodes.push_back(coreNode.second); }
	for (auto const &insideCoreNode : insideCoreNodes) { insideCoreNode->createCluster(this->insideNodes()); }
	launchThreads();
  }

  void launchThreads() {
	HLOG_SELF(0, "Launching threads")
	std::vector<std::shared_ptr<CoreNode>> insideCoreNodes;
	insideCoreNodes.reserve(this->insideNodes()->size());
	for (auto coreNode : *(this->insideNodes())) { insideCoreNodes.push_back(coreNode.second); }
	this->scheduler_->spawnThreads(insideCoreNodes);
  }

  std::shared_ptr<CoreGraphSource<GraphInputs...>> const &source() const { return source_; }

 private:
  void registerNode(const std::shared_ptr<CoreNode> &coreNode) {
	HLOG_SELF(0, "Register coreNode " << coreNode->name() << "(" << coreNode->id() << ")")
	if (!coreNode->hasBeenRegistered()) {
	  coreNode->setInside();
	  this->addUniqueInsideNode(coreNode);
	}
  }

  void duplicateInsideNodes(CoreGraph<GraphOutput, GraphInputs...> const &rhs) {
	std::multimap<CoreNode *, std::shared_ptr<CoreNode>> &originalInsideNodes = *(rhs.insideNodes());
	std::map<CoreNode *, std::shared_ptr<CoreNode>>
		correspondenceMap;

	std::shared_ptr<CoreNode> duplicate;
	// Create all the duplicates and link them to their original node
	for (std::pair<CoreNode *const, std::shared_ptr<CoreNode>> const &originalNode : originalInsideNodes) {
	  duplicate = originalNode.second->clone();
	  duplicate->belongingNode(this);
	  correspondenceMap.insert({originalNode.second.get(), duplicate});
	}

	// Add the duplicate node into the insideNode structure
	for (std::pair<CoreNode *const, std::shared_ptr<CoreNode>> const &originalNode : originalInsideNodes) {
	  CoreNode
		  *originalInsideNode = originalNode.second.get();

	  std::shared_ptr<CoreNode>
		  duplicateInsideNode = correspondenceMap.find(originalInsideNode)->second;

	  duplicateInsideNode->belongingNode(this);
	  this->insideNodes()->insert({duplicateInsideNode.get(), duplicateInsideNode});
	}

	//Do the linkage
	for (std::pair<CoreNode *const, std::shared_ptr<CoreNode>> const &originalNode : originalInsideNodes) {
	  CoreNode *originalInsideNode = originalNode.second.get();
	  std::shared_ptr<CoreNode> duplicateInsideNode = correspondenceMap.find(originalInsideNode)->second;
	  originalInsideNode->duplicateEdge(duplicateInsideNode.get(), correspondenceMap);
	}

	//Set Input/Output Node
	for (CoreNode *originalInputNode : *(rhs.inputsCoreNodes())) {
	  auto shInputCoreNode = correspondenceMap.find(originalInputNode)->second;
	  auto inputCoreNode = shInputCoreNode.get();

	  this->inputsCoreNodes_->insert(inputCoreNode);
	  (this->duplicateInputNodes<GraphInputs>(dynamic_cast<CoreReceiver<GraphInputs> *>(inputCoreNode)), ...);
	  this->registerNode(shInputCoreNode);
	}

	for (CoreSender<GraphOutput> *originalOutputNode : *(rhs.outputCoreNodes())) {
	  auto shOutputCoreNode = correspondenceMap.find(originalOutputNode)->second;
	  auto outputCoreNode = dynamic_cast<CoreSender<GraphOutput> *>(shOutputCoreNode.get());
	  this->outputCoreNodes_->insert(dynamic_cast<CoreSender<GraphOutput> *>(outputCoreNode));

	  for (CoreSender<GraphOutput> *sender : outputCoreNode->getSenders()) {
		this->sink_->addNotifier(sender);
		this->sink_->addSender(sender);
	  }

	  outputCoreNode->addSlot(this->sink_.get());
	  outputCoreNode->addReceiver(this->sink_.get());

	  this->registerNode(std::static_pointer_cast<CoreNode>(shOutputCoreNode));
	}
  }

  template<class ...InputNodeTypes>
  void addReceiversToSource(CoreMultiReceivers<InputNodeTypes...> *inputCoreNode) {
	//Set Slot/Notifiers
	this->source_->addSlot(inputCoreNode);
//	(inputCoreNode->addNotifier(std::static_pointer_cast<CoreQueueSender<InputNodeTypes>>(this->source_).get()), ...);
	(this->addSourceNotifierInputCoreNode<InputNodeTypes, InputNodeTypes...>(inputCoreNode), ...);
	(this->addReceiverToSource<InputNodeTypes>(dynamic_cast<CoreReceiver<InputNodeTypes> *>(inputCoreNode)), ...);
  }

  template<class InputNodeType, class ...InputNodeTypes>
  void addSourceNotifierInputCoreNode(CoreMultiReceivers<InputNodeTypes...> *inputCoreNode){
    if(auto compatibleSourceType = std::dynamic_pointer_cast<CoreQueueSender<InputNodeType>>(this->source_)){
	  inputCoreNode->addNotifier(compatibleSourceType.get());
	  compatibleSourceType->addReceiver(inputCoreNode);
    }
  }

  template<class InputNodeType>
  void addReceiverToSource(CoreReceiver<InputNodeType> *inputCoreNode) {
    if(inputCoreNode) {
//	  std::static_pointer_cast<CoreQueueSender<InputNodeType>>(this->source_)->addReceiver(inputCoreNode);
//	inputCoreNode->addSender(static_cast<CoreSender<InputNodeType> *>(this->source_.get()));
	  if (auto compatibleSource = dynamic_cast<CoreSender<InputNodeType> *>(this->source_.get())) {
		inputCoreNode->addSender(compatibleSource);
		dynamic_cast<CoreGraphReceiver<InputNodeType> *>(this)->addGraphReceiverInput(inputCoreNode);
	  }
//	  	dynamic_cast<CoreGraphReceiver<InputNodeType> *>(this)->addGraphReceiverInput(inputCoreNode);
	}
  }

  template<class InputNodeType>
  void duplicateInputNodes(CoreReceiver<InputNodeType> *inputCoreNode) {
	if (inputCoreNode) {
	  static_cast<CoreGraphReceiver<InputNodeType> *>(this)->addGraphReceiverInput(dynamic_cast<CoreReceiver<
		  InputNodeType> *>(inputCoreNode));
	  this->source_->addSlot(dynamic_cast<CoreSlot *>(inputCoreNode));
	  dynamic_cast<CoreSlot *>(inputCoreNode)->addNotifier(std::static_pointer_cast<CoreQueueSender<InputNodeType>>(this->source_).get());
	  std::static_pointer_cast<CoreQueueSender<InputNodeType>>(this->source_)->addReceiver(dynamic_cast<CoreReceiver<
		  InputNodeType> *>(inputCoreNode));
	  dynamic_cast<CoreReceiver<InputNodeType> *>(inputCoreNode)->addSender(static_cast<CoreSender<InputNodeType> *>(this->source_.get()));
	}
  }
};

#endif //HEDGEHOG_CORE_GRAPH_H
