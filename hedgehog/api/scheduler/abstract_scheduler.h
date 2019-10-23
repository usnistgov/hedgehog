//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_ABSTRACT_SCHEDULER_H
#define HEDGEHOG_ABSTRACT_SCHEDULER_H

#include <map>
#include <memory>
#include "../../behavior/node.h"
#include "../../core/node/core_node.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Abstract Hedgehog Scheduler interface, define per graph how the threads are bound to the nodes
class AbstractScheduler {
 public:
  /// @brief Default constructor
  AbstractScheduler() = default;
  /// @brief Default destructor
  virtual ~AbstractScheduler() = default;
  /// @brief Virtual constructor interface
  /// @return A new instance of AbstractScheduler
  [[nodiscard]] virtual std::unique_ptr<AbstractScheduler> create() const = 0;

  /// @brief Spawn the threads for the specific graph
  virtual void spawnThreads(std::vector<std::shared_ptr<core::CoreNode>> &) = 0;

  /// @brief Join all the graph's thread
  virtual void joinAll() = 0;
};
}
#endif //HEDGEHOG_ABSTRACT_SCHEDULER_H
