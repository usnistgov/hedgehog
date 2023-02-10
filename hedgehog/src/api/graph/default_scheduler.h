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



#ifndef HEDGEHOG_DEFAULT_SCHEDULER_H
#define HEDGEHOG_DEFAULT_SCHEDULER_H

#include <thread>
#include <sstream>
#include <vector>

#include "scheduler.h"
#include "../../core/abstractions/base/node/task_node_abstraction.h"
#include "../../core/abstractions/base/node/execution_pipeline_node_abstraction.h"
#include "../../core/abstractions/base/node/graph_node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {

/// Default scheduler use in Hedgehog graph
/// @brief By default, each node that needs a thread gets a thread and management of these threads is left to the OS.
class DefaultScheduler : public Scheduler {
 private:
  std::unique_ptr<std::vector<std::thread>> const
      threads_ = nullptr; ///< Vector of threads for the graph nodes

  std::unique_ptr<std::vector<core::abstraction::GraphNodeAbstraction *>>
      innerGraphs_ = nullptr; ///< Scheduler's graph

 public:
  /// Default constructor
  DefaultScheduler() :
      threads_(std::make_unique<std::vector<std::thread>>()),
      innerGraphs_(std::make_unique<std::vector<core::abstraction::GraphNodeAbstraction *>>()) {}

  /// Default destructor
  ~DefaultScheduler() override = default;

  /// @brief Definition of virtual constructor
  /// @return New instance of DefaultScheduler
  [[nodiscard]] std::unique_ptr<Scheduler> create() const override { return std::make_unique<DefaultScheduler>(); }

  /// @brief Spawn the threads for all graph's nodes
  /// @param cores Graph's inside nodes
  /// @param waitForInitialization Wait for internal nodes to be initialized flags
  /// @throw std::runtime_error if a thread cannot be created, or if the core is malformed
  void spawnThreads(std::set<core::abstraction::NodeAbstraction *> const &cores, bool waitForInitialization) override {
    std::vector<core::abstraction::TaskNodeAbstraction *> taskExec{};

    for (auto &core : cores) {
      if (auto exec = dynamic_cast<core::abstraction::TaskNodeAbstraction *>(core)) {
        try {
          threads_->emplace_back(&core::abstraction::TaskNodeAbstraction::run, exec);
          taskExec.push_back(exec);
        } catch (std::exception const &e) {
          std::ostringstream oss;
          oss << "Can not create thread for node \"" << core->name() << "\" because of error: " << e.what();
          throw std::runtime_error(oss.str());
        }
      } else if (auto graph = dynamic_cast<core::abstraction::GraphNodeAbstraction *>(core)) {
        innerGraphs_->push_back(graph);
      } else {
        std::ostringstream oss;
        oss
            << "Node " << core->name() << "/" << core->id()
            << " does not derive from the right abstraction to be handled properly by the default scheduler.";
        throw std::runtime_error(oss.str());
      }
    }

    /// If asked, wait for all internals to be initialized before returning
    if (waitForInitialization) {
      while (!std::all_of(taskExec.cbegin(), taskExec.cend(),
                          [](auto const &exec) { return exec->isInitialized(); })) {}
    }

  }

  /// Wait for all inside nodes to join and join the threads of all inside graphs
  void joinAll() override {
    std::for_each(threads_->begin(), threads_->end(), [](std::thread &t) { t.join(); });
    for (core::abstraction::GraphNodeAbstraction *innerGraph : *(this->innerGraphs_)) {
      innerGraph->joinThreads();
    }
  }

};
}
#endif //HEDGEHOG_DEFAULT_SCHEDULER_H
