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
