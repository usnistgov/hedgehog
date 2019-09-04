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
#include <vector>

#include "../../tools/printers/abstract_printer.h"
#include "../../behaviour/node.h"
#include "../../tools/logger.h"

enum struct NodeType { Graph, Task, StateManager, Sink, Source, ExecutionPipeline, Switch };

class CoreSlot;

class CoreNode {
 private:
  std::string_view
      name_ = "";

  NodeType const
      type_;

  bool
      isInside_ = false,
      hasBeenRegistered_ = false,
      isInCluster_ = false,
      isActive_ = false;

  CoreNode *
      belongingNode_ = nullptr;

  std::shared_ptr<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>>
      insideNodes_ = nullptr;

  int
      threadId_ = 0;

  size_t
      numberThreads_ = 1;

  CoreNode *
      coreClusterNode_ = nullptr;

  std::chrono::duration<uint64_t, std::micro>
      creationDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(),
      executionDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(),
      waitDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(),
      memoryWaitDuration_ = std::chrono::duration<uint64_t, std::micro>::zero();

  std::chrono::time_point<std::chrono::high_resolution_clock> const
      creationTimeStamp_ = std::chrono::high_resolution_clock::now();

  std::chrono::time_point<std::chrono::high_resolution_clock>
      startExecutionTimeStamp_ = std::chrono::high_resolution_clock::now();

 public:
  CoreNode() = delete;

  CoreNode(std::string_view const &name, NodeType const type, size_t numberThreads)
      : name_(name), type_(type), isActive_(false), coreClusterNode_(this) {
    numberThreads_ = numberThreads == 0 ? 1 : numberThreads;
    HLOG_SELF(0,
              "Creating CoreNode with type: " << (int) type << ", name: " << name << " and number of Threads: "
                                              << this->numberThreads_)
    this->insideNodes_ = std::make_shared<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>>();
  }

  virtual ~CoreNode() {
    HLOG_SELF(0, "Destructing CoreNode")
  }

  virtual std::shared_ptr<CoreNode> clone() = 0;  // Virtual constructor (copying)

  virtual std::string id() const {
    std::stringstream ss{};
    ss << "x" << this;
    return ss.str();
  }

  virtual std::vector<std::pair<std::string, std::string>> ids() const {
    return {{this->id(), this->coreClusterNode()->id()}};
  }

  std::string_view const &name() const { return name_; }
  NodeType type() const { return type_; }
  bool isInside() const { return isInside_; }
  CoreNode *coreClusterNode() const { return coreClusterNode_; }
  int threadId() const { return threadId_; }
  size_t numberThreads() const { return numberThreads_; }
  CoreNode *belongingNode() const { return belongingNode_; }

  std::shared_ptr<std::multimap<CoreNode *,
                                std::shared_ptr<CoreNode>>> const &insideNodes() const { return insideNodes_; }
  std::shared_ptr<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>> &insideNodes() { return insideNodes_; }

  std::chrono::duration<uint64_t, std::micro> const &executionTime() const { return executionDuration_; }
  std::chrono::duration<uint64_t, std::micro> const &waitTime() const { return waitDuration_; }
  std::chrono::duration<uint64_t, std::micro> const &memoryWaitTime() const { return memoryWaitDuration_; }
  bool isInCluster() const { return this->isInCluster_; }
  bool isActive() const { return isActive_; }

  virtual int graphId() { return this->belongingNode()->graphId(); }
  virtual int deviceId() { return this->belongingNode()->deviceId(); }
  virtual void deviceId(int deviceId) { this->belongingNode()->deviceId(deviceId); }

  std::chrono::time_point<std::chrono::high_resolution_clock> const &creationTimeStamp() const {
    return creationTimeStamp_; }
  virtual std::chrono::duration<uint64_t, std::micro> const maxExecutionTime() const {
    return this->executionDuration_; }
  virtual std::chrono::duration<uint64_t, std::micro> const minExecutionTime() const {
    return this->executionDuration_; }
  virtual std::chrono::duration<uint64_t, std::micro> const maxWaitTime() const { return this->waitDuration_; }
  virtual std::chrono::duration<uint64_t, std::micro> const minWaitTime() const { return this->waitDuration_; }
  std::chrono::duration<uint64_t, std::micro> const &creationDuration() const { return creationDuration_; }

  std::chrono::duration<uint64_t, std::micro> const meanExecTimeCluster() const {
    auto ret = this->executionTime();
    if (this->isInCluster()) {
      std::chrono::duration<uint64_t, std::micro> sum = std::chrono::duration<uint64_t, std::micro>::zero();
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        sum += it->second->executionTime();
      }
      ret = sum / this->numberThreads();
    }
    return ret;
  }
  std::chrono::duration<uint64_t, std::micro> const meanWaitTimeCluster() const {
    auto ret = this->waitTime();
    if (this->isInCluster()) {
      std::chrono::duration<uint64_t, std::micro> sum = std::chrono::duration<uint64_t, std::micro>::zero();
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        sum += it->second->waitTime();
      }
      ret = sum / this->numberThreads();
    }
    return ret;
  }
  std::chrono::duration<uint64_t, std::micro> const meanMemoryWaitTimeCluster() const {
    auto ret = this->memoryWaitTime();
    if (this->isInCluster()) {
      std::chrono::duration<uint64_t, std::micro> sum = std::chrono::duration<uint64_t, std::micro>::zero();
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        sum += it->second->memoryWaitTime();
      }
      ret = sum / this->numberThreads();
    }
    return ret;
  }


  uint64_t stdvExecTimeCluster() const {
    auto ret = 0;
    if (this->isInCluster()) {
      auto mean = this->meanExecTimeCluster().count(), meanSquare = mean * mean;
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        ret += (uint64_t) std::pow(it->second->executionTime().count(), 2) - meanSquare;
      }
      ret /= this->numberThreads();
      ret = (uint64_t) std::sqrt(ret);
    }
    return ret;
  }
  uint64_t stdvWaitTimeCluster() const {
    auto ret = 0;
    if (this->isInCluster()) {
      auto mean = this->meanWaitTimeCluster().count(), meanSquare = mean * mean;
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        ret += (uint64_t) std::pow(it->second->waitTime().count(), 2) - meanSquare;
      }
      ret /= this->numberThreads();
      ret = (uint64_t) std::sqrt(ret);
    }
    return ret;
  }

  uint64_t stdvMemoryWaitTimeCluster() const {
   auto ret = 0;
    if (this->isInCluster()) {
      auto mean = this->meanMemoryWaitTimeCluster().count(), meanSquare = mean * mean;
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        ret += (uint64_t) std::pow(it->second->memoryWaitTime().count(), 2) - meanSquare;
      }
      ret /= this->numberThreads();
      ret = (uint64_t) std::sqrt(ret);
    }
   return ret;
  }

  std::pair<uint64_t, uint64_t> minmaxWaitTimeCluster() const {
    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
    uint64_t val = 0;
    if (this->isInCluster()) {
      auto mean = this->meanWaitTimeCluster().count(), meanSquare = mean * mean;
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        val = it->second->waitTime().count();
        if (val < min) { min = val; }
        if (val > max) { max = val; }
      }
    }
    else {
      min = this->meanWaitTimeCluster().count();
      max = min;
    }
    return {min, max};
  }

  std::pair<uint64_t, uint64_t> minmaxMemoryWaitTimeCluster() const {
    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
    uint64_t val = 0;
    if (this->isInCluster()) {
      auto mean = this->meanMemoryWaitTimeCluster().count(), meanSquare = mean * mean;
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        val = it->second->memoryWaitTime().count();
        if (val < min) { min = val; }
        if (val > max) { max = val; }
      }
    }
    else {
      min = this->meanMemoryWaitTimeCluster().count();
      max = min;
    }
    return {min, max};
  }

  std::pair<uint64_t, uint64_t> minmaxExecTimeCluster() const {
    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
    uint64_t val = 0;
    if (this->isInCluster()) {
      auto mean = this->meanWaitTimeCluster().count(), meanSquare = mean * mean;
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        val = it->second->executionTime().count();
        if (val < min) { min = val; }
        if (val > max) { max = val; }
      }
    } else {
      min = this->meanExecTimeCluster().count();
      max = min;
    }

    return {min, max};
  }



  size_t numberActiveThreadInCluster() const {
    size_t ret = 0;
    if (this->isInCluster()) {
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        ret += it->second->isActive() ? 1 : 0;
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
  void coreClusterNode(CoreNode *coreClusterNode) { coreClusterNode_ = coreClusterNode; }
  void numberThreads(size_t numberThreads) { numberThreads_ = numberThreads; }
  void belongingNode(CoreNode *belongingNode) { belongingNode_ = belongingNode; }
  void hasBeenRegistered(bool hasBeenRegistered) { hasBeenRegistered_ = hasBeenRegistered; }
  void isActive(bool isActive) { isActive_ = isActive; }
  void creationDuration(std::chrono::duration<uint64_t, std::micro> const &creationDuration) {
    creationDuration_ = creationDuration;
  }

  virtual void preRun() {}
  virtual void run() {}
  virtual void postRun() {}

  virtual Node *node() = 0;
  virtual void createCluster([[maybe_unused]]std::shared_ptr<std::multimap<CoreNode *,
                                                                           std::shared_ptr<CoreNode>>> &insideNodesGraph) {};
  virtual void visit(AbstractPrinter *printer) = 0;

  virtual std::set<CoreSlot *> getSlots() = 0;

  void removeInsideNode(CoreNode *coreNode) {
    HLOG_SELF(0, "Remove inside node " << coreNode->id() << ")")
    this->insideNodes()->erase(coreNode);
  }

  virtual std::string extraPrintingInformation() { return node()->extraPrintingInformation(); }
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
    this->coreClusterNode(rhs->coreClusterNode());
  }

  virtual void duplicateEdge(CoreNode *, std::map<CoreNode *, std::shared_ptr<CoreNode>> &) = 0;

  virtual void removeForAllSenders(CoreNode*) {
    std::cerr << "Nope" << std::endl;
  };

  void incrementWaitForMemoryDuration(std::chrono::duration<uint64_t, std::micro> const &memoryWait) {
    this->memoryWaitDuration_ += memoryWait;
  }

 protected:
  void addUniqueInsideNode(const std::shared_ptr<CoreNode> &coreNode) {
    HLOG_SELF(0, "Add InsideNode " << coreNode->name() << "(" << coreNode->id() << ")")
    if (insideNodes_->find(coreNode.get()) == insideNodes_->end()) {
      coreNode->belongingNode(this);
      coreNode->hasBeenRegistered(true);
      insideNodes_->insert({coreNode.get(), coreNode});
    }
  }

  void incrementWaitDuration(std::chrono::duration<uint64_t, std::micro> const &wait) {
    this->waitDuration_ += wait;
  }

  void incrementExecutionDuration(std::chrono::duration<uint64_t, std::micro> const &exec) {
    this->executionDuration_ += exec;
  }


  void isInside(bool isInside) { isInside_ = isInside; }
};

#endif //HEDGEHOG_CORE_NODE_H
