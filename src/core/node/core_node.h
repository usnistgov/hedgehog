//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_CORE_NODE_H
#define HEDGEHOG_CORE_NODE_H

#include <iostream>
#include <sstream>
#include <memory>
#include <map>
#include <set>
#include <chrono>
#include <cmath>

#include "../../tools/printers/abstract_printer.h"
#include "../../behaviour/node.h"
#include "../../tools/logger.h"

enum struct NodeType { Graph, Task, StateManager, Sink, Source };

class CoreNode {
 private:
  std::string_view
      name_ = nullptr;

  NodeType const
      type_;

  bool
      isInside_ = false,
      hasBeenRegistered_ = false,
      isInCluster_ = false,
      isActive_ = true;

  CoreNode *
      belongingNode_ = nullptr;

  std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>>
      insideNodes_ = nullptr;

  uint8_t
      threadId_ = 0;

  size_t
      numberThreads_ = 1;

  std::string
      clusterId_ = {},
      extraPrintingInformation_ = "";

  std::chrono::duration<uint64_t, std::micro>
      creationDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(),
      executionDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(),
      waitDuration_ = std::chrono::duration<uint64_t, std::micro>::zero();

  std::chrono::time_point<
      std::chrono::high_resolution_clock
  > const
      creationTimeStamp_ = std::chrono::high_resolution_clock::now();

  std::chrono::time_point<std::chrono::high_resolution_clock>
      startExecutionTimeStamp_ = std::chrono::high_resolution_clock::now();

 public:
  CoreNode() = delete;

  CoreNode(std::string_view const &name, NodeType const type, size_t numberThreads)
      : name_(name), type_(type), isActive_(true), clusterId_(this->id()) {
    numberThreads_ = numberThreads == 0 ? 1 : numberThreads;
    HLOG_SELF(0,
              "Creating CoreNode with type: " << (int) type << ", name: " << name << " and number of Threads: "
                                              << this->numberThreads_);
    this->insideNodes_ = std::make_shared<std::multimap<std::string, std::shared_ptr<Node>>>();
  }

  virtual ~CoreNode() {
    HLOG_SELF(0, "Destructing CoreNode")
  };

  std::string id() const {
    std::stringstream ss{};
    ss << "x" << this;
    return ss.str();
  }

  virtual std::vector<std::pair<std::string, std::string>> ids() const {
    return {{this->id(), this->clusterId()}};
  }

  std::string_view const &name() const { return name_; }
  NodeType type() const { return type_; }
  bool isInside() const { return isInside_; }
  std::string const &clusterId() const { return clusterId_; }
  uint8_t threadId() const { return threadId_; }
  size_t numberThreads() const { return numberThreads_; }
  CoreNode *belongingNode() const { return belongingNode_; }
  std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>> const &insideNodes() const { return insideNodes_; }
  std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>> &insideNodes() { return insideNodes_; }
  std::chrono::duration<uint64_t, std::micro> const &executionTime() const { return executionDuration_; }
  std::chrono::duration<uint64_t, std::micro> const &waitTime() const { return waitDuration_; }
  bool isInCluster() const { return this->isInCluster_; }
  bool isActive() const { return isActive_; }

  std::chrono::time_point<std::chrono::high_resolution_clock> const &creationTimeStamp() const { return creationTimeStamp_; }
  virtual std::chrono::duration<uint64_t,
                                std::micro> const maxExecutionTime() const { return this->executionDuration_; };
  virtual std::chrono::duration<uint64_t,
                                std::micro> const minExecutionTime() const { return this->executionDuration_; };
  virtual std::chrono::duration<uint64_t, std::micro> const maxWaitTime() const { return this->waitDuration_; };
  virtual std::chrono::duration<uint64_t, std::micro> const minWaitTime() const { return this->waitDuration_; };
  std::chrono::duration<uint64_t, std::micro> const &creationDuration() const { return creationDuration_; }

  std::chrono::duration<uint64_t, std::micro> const meanExecTimeCluster() const {
    auto ret = this->executionTime();
    if (this->isInCluster()) {
      std::chrono::duration<uint64_t, std::micro> sum = std::chrono::duration<uint64_t, std::micro>::zero();
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->clusterId()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->clusterId()).second; ++it) {
        sum += it->second->getCore()->executionTime();
      }
      ret = sum / this->numberThreads();
    }
    return ret;
  };

  std::chrono::duration<uint64_t, std::micro> const meanWaitTimeCluster() const {
    auto ret = this->waitTime();
    if (this->isInCluster()) {
      std::chrono::duration<uint64_t, std::micro> sum = std::chrono::duration<uint64_t, std::micro>::zero();
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->clusterId()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->clusterId()).second; ++it) {
        sum += it->second->getCore()->waitTime();
      }
      ret = sum / this->numberThreads();
    }
    return ret;
  };

  uint64_t stdvExecTimeCluster() const {
    auto ret = 0;
    if (this->isInCluster()) {
      auto mean = this->meanExecTimeCluster().count(), meanSquare = mean * mean;
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->clusterId()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->clusterId()).second; ++it) {
        ret += (uint64_t) std::pow(it->second->getCore()->executionTime().count(), 2) - meanSquare;
      }
      ret /= this->numberThreads();
      ret = (uint64_t) std::sqrt(ret);
    }
    return ret;
  };

  uint64_t stdvWaitTimeCluster() const {
    auto ret = 0;
    if (this->isInCluster()) {
      auto mean = this->meanWaitTimeCluster().count(), meanSquare = mean * mean;
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->clusterId()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->clusterId()).second; ++it) {
        ret += (uint64_t) std::pow(it->second->getCore()->waitTime().count(), 2) - meanSquare;
      }
      ret /= this->numberThreads();
      ret = (uint64_t) std::sqrt(ret);
    }
    return ret;
  };

  size_t numberActiveThreadInCluster() const {
    size_t ret = 0;
    if (this->isInCluster()) {
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->clusterId()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->clusterId()).second; ++it) {
        ret += it->second->getCore()->isActive() ? 1 : 0;
      }
    } else {
      ret = this->isActive() ? 1 : 0;
    }
    return ret;
  }


  bool hasBeenRegistered() const { return hasBeenRegistered_; }
  virtual void setInside() { this->isInside_ = true; }
  void setInCluster() { this->isInCluster_ = true; }
  void threadId(uint8_t threadId) { threadId_ = threadId; }
  void clusterId(std::string const &clusterId) { clusterId_ = clusterId; }
  void numberThreads(size_t numberThreads) { numberThreads_ = numberThreads; }
  void belongingNode(CoreNode *belongingNode) { belongingNode_ = belongingNode; }
  void hasBeenRegistered(bool hasBeenRegistered) { hasBeenRegistered_ = hasBeenRegistered; }
  void extraPrintingInformation(std::string const &extraInformation) { extraPrintingInformation_ = extraInformation; }
  void isActive(bool isActive) { isActive_ = isActive; }
  void creationDuration(std::chrono::duration<uint64_t, std::micro> const &creationDuration) {
    creationDuration_ = creationDuration;
  }

  virtual void run() {}
  virtual Node *getNode() = 0;
  virtual void copyWholeNode(std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>> &insideNodesGraph) = 0;
  virtual void visit(AbstractPrinter *printer) = 0;

  void removeInsideNode(std::string const &id) {
    HLOG_SELF(0, "Remove inside node " << id << ")")
    this->insideNodes()->erase(id);
  }

  std::string const &extraPrintingInformation() const { return extraPrintingInformation_; }
  std::chrono::time_point<std::chrono::high_resolution_clock> const &startExecutionTimeStamp() const { return startExecutionTimeStamp_; }

  void startExecutionTimeStamp(std::chrono::time_point<std::chrono::high_resolution_clock> const &startExecutionTimeStamp) {
    startExecutionTimeStamp_ = startExecutionTimeStamp;
  }

  std::chrono::duration<uint64_t, std::micro> const &executionDuration() const {
    return executionDuration_;
  }
  void executionDuration(std::chrono::duration<uint64_t, std::micro> const &executionDuration) {
    executionDuration_ = executionDuration;
  }
  virtual void joinThreads() {}

  void copyInnerStructure(CoreNode *rhs) {
    this->isInside_ = rhs->isInside();
    this->belongingNode_ = rhs->belongingNode();
    this->hasBeenRegistered_ = rhs->hasBeenRegistered();
    this->isInCluster_ = rhs->isInCluster();
    this->numberThreads_ = rhs->numberThreads();
    this->clusterId_ = rhs->clusterId();
  }

 protected:
  void addUniqueInsideNode(const std::shared_ptr<Node> &node) {
    HLOG_SELF(0, "Add InsideNode " << node->getCore()->name() << "(" << node->getCore()->id() << ")")
    if (insideNodes_->find(node->getCore()->id()) == insideNodes_->end()) {
      node->getCore()->belongingNode(this);
      node->getCore()->hasBeenRegistered(true);
      insideNodes_->insert({node->getCore()->id(), node});
    }
  };

  void incrementWaitDuration(std::chrono::duration<uint64_t, std::micro> const &wait) {
    this->waitDuration_ += wait;
  }

  void incrementExecutionDuration(std::chrono::duration<uint64_t, std::micro> const &exec) {
    this->executionDuration_ += exec;
  }

 private:
  void isInside(bool isInside) { isInside_ = isInside; }
};

#endif //HEDGEHOG_CORE_NODE_H
