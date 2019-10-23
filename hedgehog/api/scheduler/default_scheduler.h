//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_DEFAULT_SCHEDULER_H
#define HEDGEHOG_DEFAULT_SCHEDULER_H

#include <thread>
#include <vector>
#include <algorithm>
#include "abstract_scheduler.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Default scheduler for Hedgehog graph
class DefaultScheduler : public AbstractScheduler {
 private:
  std::unique_ptr<std::vector<std::thread>>
      threads_ = nullptr; ///< List of threads spawned
  std::unique_ptr<std::vector<std::shared_ptr<core::CoreNode>>>
      innerGraphs_ = nullptr; ///< List of inner graphs, all graphs could have it owns AbstractScheduler

 public:
  /// @brief Default constructor
  DefaultScheduler() {
    this->threads_ = std::make_unique<std::vector<std::thread>>();
    this->innerGraphs_ = std::make_unique<std::vector<std::shared_ptr<core::CoreNode>>>();
  }

  /// @brief Definition of virtual constructor
  /// @return New instance of DefaultScheduler
  [[nodiscard]] std::unique_ptr<AbstractScheduler> create() const override {
    return std::make_unique<DefaultScheduler>();
  }

  /// @brief Default destructor
  ~DefaultScheduler() override = default;

  /// @brief Spawn the threads for all graph's inside node
  /// @param insideCores Graph's inside nodes
  void spawnThreads(std::vector<std::shared_ptr<core::CoreNode>> &insideCores) override {
    // Iterate over inside nodes
    for (auto &core : insideCores) {
      // If the node is not a graph
      if (core->type() != core::NodeType::Graph) {
        // Create the thread for the run
        threads_->emplace_back(&core::CoreNode::run, core);
        // If the node is a graph
      } else { innerGraphs_->push_back(core); }
    }
  }

  /// @brief Wait for all inside nodes to join and join the threads of all inside graphs
  void joinAll() override {
    std::for_each(threads_->begin(), threads_->end(), [](std::thread &t) { t.join(); });
    for (std::shared_ptr<core::CoreNode> &innerGraph : *(this->innerGraphs_)) { innerGraph->joinThreads(); }
  }

};
}
#endif //HEDGEHOG_DEFAULT_SCHEDULER_H
