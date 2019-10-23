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

#include "../../api/printer/abstract_printer.h"
#include "../../behavior/node.h"
#include "../../tools/logger.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Hedgehog node's type
enum struct NodeType { Graph, Task, StateManager, Sink, Source, ExecutionPipeline, Switch };

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief CoreSlot forward declaration
class CoreSlot;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Main Hedgehog core abstraction
class CoreNode {
 private:
  bool
      isInside_ = false, ///< True if the node is inside a graph, else False
      hasBeenRegistered_ = false, ///< True if the node has been registered into a graph, else False
      isCudaRelated_ = false, ///< True if the node is related with CUDA, else False
      isInCluster_ = false, ///< True if the node is in cluster, else False
      isActive_ = false; ///< True if the node is active, else False

  int threadId_ = 0; ///< Thread id, used to debug only

  size_t numberThreads_ = 1; ///< Number of threads associated to the node

  std::string_view name_ = ""; ///< Node name

  NodeType const type_; ///< Node type

  CoreNode
      *belongingNode_ = nullptr, ///< Pointer to the belonging node, a graph, does not store memory, just for reference
      *coreClusterNode_ = nullptr; ///< Pointer to the main cluster node, does not store memory, just for reference

  std::shared_ptr<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>>
      insideNodes_ = nullptr; ///< Map of inside nodes [Main Cluster Node -> Node]

  std::chrono::duration<uint64_t, std::micro>
      creationDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(), ///< Node creation duration
      executionDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(), ///< Node execution duration
      waitDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(), ///< Node wait duration
      memoryWaitDuration_ = std::chrono::duration<uint64_t, std::micro>::zero(); ///< Node memory wait duration

  std::chrono::time_point<std::chrono::high_resolution_clock> const
      creationTimeStamp_ = std::chrono::high_resolution_clock::now(); ///< Node creation timestamp

  std::chrono::time_point<std::chrono::high_resolution_clock>
      startExecutionTimeStamp_ = std::chrono::high_resolution_clock::now(); ///< Node begin execution timestamp

 public:
  /// @brief Deleted default constructor
  CoreNode() = delete;

  /// @brief Core node only constructor
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Node number of threads
  CoreNode(std::string_view const &name, NodeType const type, size_t numberThreads)
      : isActive_(false), name_(name), type_(type), coreClusterNode_(this) {
    numberThreads_ = numberThreads == 0 ? 1 : numberThreads;
    HLOG_SELF(0,
              "Creating CoreNode with type: " << (int) type << ", name: " << name << " and number of Threads: "
                                              << this->numberThreads_)
    this->insideNodes_ = std::make_shared<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>>();
  }

  /// @brief Default virtual destructor
  virtual ~CoreNode() {
    HLOG_SELF(0, "Destructing CoreNode")
  }

  /// @brief Virtual constructor for copy
  /// @return A copy of the current node
  virtual std::shared_ptr<CoreNode> clone() = 0;  // Virtual constructor (copying)

  // Accessor

  /// @brief Unique Id accessor
  /// @return Unique Id
  [[nodiscard]] virtual std::string id() const {
    std::stringstream ss{};
    ss << "x" << this;
    return ss.str();
  }

  /// @brief Input node ids [nodeId, nodeIdCluster] accessor
  /// @return  Input node ids [nodeId, nodeIdCluster]
  [[nodiscard]] virtual std::vector<std::pair<std::string, std::string>> ids() const {
    return {{this->id(), this->coreClusterNode()->id()}};
  }

  /// @brief Node name accessor
  /// @return Node name
  [[nodiscard]] std::string_view const &name() const { return name_; }

  /// @brief Node type accessor
  /// @return Node type
  [[nodiscard]] NodeType type() const { return type_; }

  /// @brief Node inside property accessor
  /// @return Node inside property
  [[nodiscard]] bool isInside() const { return isInside_; }

  /// @brief Node registration property accessor
  /// @return Node registration property
  [[nodiscard]] bool hasBeenRegistered() const { return hasBeenRegistered_; }

  /// @brief Main cluster core node link to this node accessor
  /// @return Main cluster core node link to this node
  [[nodiscard]] CoreNode *coreClusterNode() const { return coreClusterNode_; }

  /// @brief Thread id accessor
  /// @return Thread id
  [[nodiscard]] int threadId() const { return threadId_; }

  /// @brief Number of threads associated accessor
  /// @return Number of threads associated
  [[nodiscard]] size_t numberThreads() const { return numberThreads_; }

  /// @brief Belonging node accessor
  /// @return Belonging node
  [[nodiscard]] CoreNode *belongingNode() const { return belongingNode_; }

  /// @brief Inside node accessor
  /// @return Inside node
  [[nodiscard]] std::shared_ptr<std::multimap<CoreNode *,
                                              std::shared_ptr<CoreNode>>> const &insideNodes() const {
    return insideNodes_;
  }

  /// @brief Inside nodes accessor
  /// @return Inside nodes
  [[nodiscard]] std::shared_ptr<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>> &insideNodes() {
    return insideNodes_;
  }

  /// @brief Execution time accessor
  /// @return Execution time
  [[nodiscard]] std::chrono::duration<uint64_t, std::micro> const &executionTime() const { return executionDuration_; }

  /// @brief Wait time accessor
  /// @return Wait time
  [[nodiscard]] std::chrono::duration<uint64_t, std::micro> const &waitTime() const { return waitDuration_; }

  /// @brief Memory wait time accessor
  /// @return Memory wait time
  [[nodiscard]] std::chrono::duration<uint64_t, std::micro> const &memoryWaitTime() const {
    return memoryWaitDuration_;
  }

  /// @brief In cluster property accessor
  /// @return In cluster property
  [[nodiscard]] bool isInCluster() const { return this->isInCluster_; }

  /// @brief Is active property accessor
  /// @return Is active property
  [[nodiscard]] bool isActive() const { return isActive_; }

  /// @brief Is related to CUDA, used to have a green background on the dot file
  /// @return True if CUDA related, else False
  [[nodiscard]] bool isCudaRelated() const { return isCudaRelated_; }

  /// @brief Graph id accessor
  /// @return Graph id
  [[nodiscard]] virtual int graphId() { return this->belongingNode()->graphId(); }

  /// @brief Device id accessor
  /// @return Device id
  [[nodiscard]] virtual int deviceId() { return this->belongingNode()->deviceId(); }

  /// @brief Maximum execution time accessor
  /// @return Maximum execution time
  [[nodiscard]] virtual std::chrono::duration<uint64_t, std::micro> maxExecutionTime() const {
    return this->executionDuration_;
  }

  /// @brief Minimum execution time accessor
  /// @return Minimum execution time
  [[nodiscard]] virtual std::chrono::duration<uint64_t, std::micro> minExecutionTime() const {
    return this->executionDuration_;
  }

  /// @brief Maximum waiting time accessor
  /// @return Maximum waiting time
  [[nodiscard]] virtual std::chrono::duration<uint64_t, std::micro> maxWaitTime() const { return this->waitDuration_; }

  /// @brief Minimum waiting time accessor
  /// @return Minimum waiting time
  [[nodiscard]] virtual std::chrono::duration<uint64_t, std::micro> minWaitTime() const { return this->waitDuration_; }

  /// @brief Creation timestamp accessor
  /// @return Creation timestamp
  [[nodiscard]] std::chrono::time_point<std::chrono::high_resolution_clock> const &creationTimeStamp() const {
    return creationTimeStamp_;
  }

  /// @brief Execution start timestamp accessor
  /// @return Execution start timestamp
  [[nodiscard]] std::chrono::time_point<std::chrono::high_resolution_clock> const &startExecutionTimeStamp() const {
    return startExecutionTimeStamp_;
  }

  /// @brief Creation duration accessor
  /// @return Creation duration
  [[nodiscard]] std::chrono::duration<uint64_t,
                                      std::micro> const &creationDuration() const { return creationDuration_; }

  /// @brief Execution duration accessor
  /// @return Execution duration
  [[nodiscard]] std::chrono::duration<uint64_t, std::micro> const &executionDuration() const {
    return executionDuration_;
  }

  /// @brief Compute and return the mean execution time for all tasks in the node cluster
  /// @return The mean execution time for all tasks in the node cluster
  [[nodiscard]] std::chrono::duration<uint64_t, std::micro> meanExecTimeCluster() const {
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

  /// @brief Compute and return the mean wait time for all tasks in the node cluster
  /// @return The mean wait time for all tasks in the node cluster
  [[nodiscard]] std::chrono::duration<uint64_t, std::micro> meanWaitTimeCluster() const {
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

  /// @brief Compute and return the mean memory wait time for all tasks in the node cluster
  /// @return The mean memory wait time for all tasks in the node cluster
  [[nodiscard]] std::chrono::duration<uint64_t, std::micro> meanMemoryWaitTimeCluster() const {
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

  /// @brief Compute and return the standard deviation execution time for all tasks in the node cluster
  /// @return The standard deviation execution time for all tasks in the node cluster
  [[nodiscard]] uint64_t stdvExecTimeCluster() const {
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

  /// @brief Compute and return the standard deviation wait time for all tasks in the node cluster
  /// @return The standard deviation wait time for all tasks in the node cluster
  [[nodiscard]] uint64_t stdvWaitTimeCluster() const {
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

  /// @brief Compute and return the standard deviation memory wait time for all tasks in the node cluster
  /// @return The standard deviation memory wait time for all tasks in the node cluster
  [[nodiscard]] uint64_t stdvMemoryWaitTimeCluster() const {
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

  /// @brief Compute and return the min and max wait time for all tasks in the node cluster
  /// @return The min and max mean wait time for all tasks in the node cluster
  [[nodiscard]] std::pair<uint64_t, uint64_t> minmaxWaitTimeCluster() const {
    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
    uint64_t val = 0;
    if (this->isInCluster()) {
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        val = it->second->waitTime().count();
        if (val < min) { min = val; }
        if (val > max) { max = val; }
      }
    } else {
      min = this->meanWaitTimeCluster().count();
      max = min;
    }
    return {min, max};
  }

  /// @brief Compute and return the min and max memory wait time for all tasks in the node cluster
  /// @return The min and max memory mean wait time for all tasks in the node cluster
  [[nodiscard]] std::pair<uint64_t, uint64_t> minmaxMemoryWaitTimeCluster() const {
    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
    uint64_t val = 0;
    if (this->isInCluster()) {
      for (auto it = this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).first;
           it != this->belongingNode()->insideNodes()->equal_range(this->coreClusterNode()).second; ++it) {
        val = it->second->memoryWaitTime().count();
        if (val < min) { min = val; }
        if (val > max) { max = val; }
      }
    } else {
      min = this->meanMemoryWaitTimeCluster().count();
      max = min;
    }
    return {min, max};
  }

  /// @brief Compute and return the min and max execution time for all tasks in the node cluster
  /// @return The min and max execution wait time for all tasks in the node cluster
  [[nodiscard]] std::pair<uint64_t, uint64_t> minmaxExecTimeCluster() const {
    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
    uint64_t val = 0;
    if (this->isInCluster()) {
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

  /// @brief Compute and return the number of active nodes in a cluster
  /// @return The number of active nodes in a cluster
  [[nodiscard]] size_t numberActiveThreadInCluster() const {
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

  /// @brief Extra printing information accessor
  /// @return Extra printing information
  [[nodiscard]] virtual std::string extraPrintingInformation() { return node()->extraPrintingInformation(); }

  // Setter
  /// @brief Execution timestamp setter
  /// @param startExecutionTimeStamp Execution timestamp to set
  void startExecutionTimeStamp(
      std::chrono::time_point<std::chrono::high_resolution_clock> const &startExecutionTimeStamp) {
    startExecutionTimeStamp_ = startExecutionTimeStamp;
  }
  /// @brief Device id setter
  /// @param deviceId Device id
  virtual void deviceId(int deviceId) { this->belongingNode()->deviceId(deviceId); }

  /// @brief Set the node as inside, (inside a graph)
  virtual void setInside() { this->isInside_ = true; }

  /// @brief Set the task as part of a cluster
  void setInCluster() { this->isInCluster_ = true; }

  /// @brief Set the thread id
  /// @param threadId Thread id to set
  void threadId(uint8_t threadId) { threadId_ = threadId; }

  /// @brief Set the main cluster node to associate to this node
  /// @param coreClusterNode Main cluster node to associate to this node
  void coreClusterNode(CoreNode *coreClusterNode) { coreClusterNode_ = coreClusterNode; }

  /// @brief Name node setter
  /// @param name Name node to set
  void name(std::string_view const &name) { name_ = name; }

  /// @brief Number of threads setter
  /// @param numberThreads Number of threads
  void numberThreads(size_t numberThreads) { numberThreads_ = numberThreads; }

  /// @brief Belonging node setter
  /// @param belongingNode Belonging node
  void belongingNode(CoreNode *belongingNode) { belongingNode_ = belongingNode; }

  /// @brief Has been registered property setter
  /// @param hasBeenRegistered Has been registered property
  void hasBeenRegistered(bool hasBeenRegistered) { hasBeenRegistered_ = hasBeenRegistered; }

  /// @brief Is active property setter
  /// @param isActive Is active property
  void isActive(bool isActive) { isActive_ = isActive; }

  /// @brief Is CUDA related property setter
  /// @param isCudaRelated CUDA related property to set
  void isCudaRelated(bool isCudaRelated) { isCudaRelated_ = isCudaRelated; }

  /// @brief Set the node as being inside another one
  /// @param isInside True if the node is inside another one, else False
  void isInside(bool isInside) { isInside_ = isInside; }

  /// @brief Creation duration setter
  /// @param creationDuration Creation duration to set
  void creationDuration(std::chrono::duration<uint64_t, std::micro> const &creationDuration) {
    creationDuration_ = creationDuration;
  }

  /// @brief Execution duration setter
  /// @param executionDuration Execution duration
  void executionDuration(std::chrono::duration<uint64_t, std::micro> const &executionDuration) {
    executionDuration_ = executionDuration;
  }

  /// @brief Add wait for memory duration to total duration
  /// @param memoryWait Duration to add to the memory duration
  void incrementWaitForMemoryDuration(std::chrono::duration<uint64_t, std::micro> const &memoryWait) {
    this->memoryWaitDuration_ += memoryWait;
  }

  // Virtual
  /// @brief Method defining what to do before the run
  virtual void preRun() {}

  /// @brief Run method, main execution
  virtual void run() {}

  /// @brief Method defining what to do after the run
  virtual void postRun() {}

  /// @brief Define how to create a cluster for the node, by default do nothing
  virtual void createCluster(std::shared_ptr<std::multimap<CoreNode *, std::shared_ptr<CoreNode>>> &) {};

  /// @brief Define what is done when the thread is joined
  virtual void joinThreads() {}

  /// @brief Duplicate all of the edges from this to its copy duplicateNode
  /// @param duplicateNode Node to connect
  /// @param correspondenceMap Correspondence  map from base node to copy
  virtual void duplicateEdge(
      CoreNode *duplicateNode, std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) = 0;

  /// @brief User's node accessor
  /// @return User's node
  virtual behavior::Node *node() = 0;

  /// @brief Abstract visit method for printing mechanism
  /// @param printer Printer visitor to print the node
  virtual void visit(AbstractPrinter *printer) = 0;

  /// @brief Slots accessor for the node
  /// @return Slots from the node
  virtual std::set<CoreSlot *> getSlots() = 0;

  // Public method
  /// @brief Remove a node from the registered inside nodes
  /// @param coreNode Node to remove from the inside nodes
  void removeInsideNode(CoreNode *coreNode) {
    HLOG_SELF(0, "Remove inside node " << coreNode->id() << ")")
    this->insideNodes()->erase(coreNode);
  }


  /// @brief Copy inner structure from rhs nodes to this
  /// @param rhs Node to copy to this
  void copyInnerStructure(CoreNode *rhs) {
    this->isInside_ = rhs->isInside();
    this->belongingNode_ = rhs->belongingNode();
    this->hasBeenRegistered_ = rhs->hasBeenRegistered();
    this->isInCluster_ = rhs->isInCluster();
    this->isCudaRelated_ = rhs->isCudaRelated();
    this->numberThreads_ = rhs->numberThreads();
    this->coreClusterNode(rhs->coreClusterNode());
  }

 protected:
  /// @brief Add a node to the inside nodes
  /// @param coreNode Node to add to the inside nodes
  void addUniqueInsideNode(const std::shared_ptr<CoreNode> &coreNode) {
    HLOG_SELF(0, "Add InsideNode " << coreNode->name() << "(" << coreNode->id() << ")")
    if (insideNodes_->find(coreNode.get()) == insideNodes_->end()) {
      coreNode->belongingNode(this);
      coreNode->hasBeenRegistered(true);
      insideNodes_->insert({coreNode.get(), coreNode});
    }
  }

  /// @brief Increment wait duration
  /// @param wait Duration to add to the wait duration
  void incrementWaitDuration(std::chrono::duration<uint64_t, std::micro> const &wait) { this->waitDuration_ += wait; }

  /// @brief Increment execution duration
  /// @param exec Duration to add to the execution duration
  void incrementExecutionDuration(std::chrono::duration<uint64_t, std::micro> const &exec) {
    this->executionDuration_ += exec;
  }
};

}
#endif //HEDGEHOG_CORE_NODE_H
