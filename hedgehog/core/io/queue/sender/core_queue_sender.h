//
// Created by anb22 on 5/8/19.
//

#ifndef HEDGEHOG_CORE_QUEUE_SENDER_H
#define HEDGEHOG_CORE_QUEUE_SENDER_H

#include "core_queue_notifier.h"
#include "../receiver/core_queue_receiver.h"
#include "../../base/sender/core_sender.h"
#include "../../../../tools/traits.h"

template<class NodeOutput>
class CoreQueueSender : public CoreSender<NodeOutput>, public virtual CoreQueueNotifier {
  std::shared_ptr<std::set<CoreQueueReceiver<NodeOutput> *>> destinations_ = nullptr;

 public:
  CoreQueueSender(std::string_view const &name, NodeType const type, size_t const numberThreads)
	  : CoreQueueNotifier(name, type, numberThreads),
		CoreSender<NodeOutput>(name, type, numberThreads) {
	HLOG_SELF(0, "Creating CoreQueueSender with type: " << (int) type << " and name: " << name)
	destinations_ = std::make_shared<std::set<CoreQueueReceiver<NodeOutput> *>>();
  }

  ~CoreQueueSender() override {HLOG_SELF(0, "Destructing CoreQueueSender")}

  virtual std::shared_ptr<std::set<CoreQueueReceiver<NodeOutput> *>> const &destinations() const {
	return destinations_;
  }

  void addReceiver(CoreReceiver<NodeOutput> *receiver) override {
	HLOG_SELF(0, "Add receiver " << receiver->name() << "(" << receiver->id() << ")")
	for (auto queueReceiver: receiver->receivers()) {
	  auto r = dynamic_cast<CoreQueueReceiver<NodeOutput> *>(queueReceiver);
	  assert(r != nullptr);
	  this->destinations_->insert(r);
	}
  }

  void removeReceiver(CoreReceiver<NodeOutput> *receiver) override {
	HLOG_SELF(0, "Remove receiver " << receiver->name() << "(" << receiver->id() << ")")
	for (auto queueReceiver: receiver->receivers()) {
	  auto r = dynamic_cast<CoreQueueReceiver<NodeOutput> *>(queueReceiver);
	  assert(r != nullptr);
	  this->destinations_->erase(r);
	}
  }

  void sendAndNotify(std::shared_ptr<NodeOutput> ptr) final {
	for (CoreQueueReceiver<NodeOutput> *receiver : *(this->destinations_)) {
	  HLOG_SELF(2, "Send data to " << receiver->name() << "(" << receiver->id() << ")")
	  receiver->receive(ptr);
	  HLOG_SELF(2, "Wake up " << receiver->name() << "(" << receiver->id() << ")")
	  receiver->queueSlot()->wakeUp();
	}
  }

  void visit(AbstractPrinter *printer) override {
	HLOG_SELF(1, "Visit")
	for (CoreQueueReceiver<NodeOutput> *receiver : *(this->destinations())) {
	  if (receiver->type() != NodeType::Switch || receiver->type() != NodeType::ExecutionPipeline) {
		printer->printEdge(this,
						   receiver,
						   HedgehogTraits::type_name<NodeOutput>(),
						   receiver->queueSize(),
						   receiver->maxQueueSize(),
						   HedgehogTraits::is_managed_memory_v<NodeOutput>);
	  }
	}
  }

  std::set<CoreSender<NodeOutput> *> getSenders() override { return {this}; }

  void copyInnerStructure(CoreQueueSender<NodeOutput> *rhs) {
	HLOG_SELF(0, "Copy Cluster CoreQueueSender information from " << rhs->name() << "(" << rhs->id() << ")")
	this->destinations_ = rhs->destinations_;
	CoreQueueNotifier::copyInnerStructure(rhs);
  }

 protected:
  void duplicateEdge(CoreNode *duplicateNode,
					 std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
	for (CoreQueueReceiver<NodeOutput> *originalReceiver : *(this->destinations())) {
	  auto nodeReceiverFound = correspondenceMap.find(static_cast<CoreNode *>(originalReceiver));
	  if (nodeReceiverFound != correspondenceMap.end()) {
		if (nodeReceiverFound->second->id() == this->id()) {
		  std::cerr << "Receiver found is the same as the original receiver" << std::endl;
		  exit(42);
		}
		connectSenderToReceiverDuplication(duplicateNode, nodeReceiverFound->second.get());
	  }
	}
  }

 private:
  void connectSenderToReceiverDuplication(CoreNode *coreNodeSender, CoreNode *coreNodeReceiver) {
	auto coreReceiver = dynamic_cast<CoreReceiver<NodeOutput> *>(coreNodeReceiver);
	auto coreSlot = dynamic_cast<CoreSlot *>(coreNodeReceiver);
	auto coreNotifier = dynamic_cast<CoreNotifier *>(coreNodeSender);

	for (auto r : coreReceiver->receivers()) {
	  dynamic_cast<CoreQueueSender<NodeOutput> *>(coreNodeSender)->addReceiver(r);
	}

	for (CoreSlot *slot : coreSlot->getSlots()) { coreNotifier->addSlot(slot); }

	for (auto s : dynamic_cast<CoreQueueSender<NodeOutput> *>(coreNodeSender)->getSenders()) {
	  coreReceiver->addSender(s);
	  coreSlot->addNotifier(s);
	}
  }
};

#endif //HEDGEHOG_CORE_QUEUE_SENDER_H
