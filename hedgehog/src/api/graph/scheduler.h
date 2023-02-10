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



#ifndef HEDGEHOG_SCHEDULER_H
#define HEDGEHOG_SCHEDULER_H

#include <memory>
#include <set>

#include "../../core/abstractions/base/node/node_abstraction.h"
/// @brief Hedgehog main namespace
namespace hh {

/// @brief Scheduler abstraction to manage graph's threads
class Scheduler {

 public:
  /// @brief Scheduler default constructor
  Scheduler() = default;

  /// @brief Scheduler default destructor
  virtual ~Scheduler() = default;

  /// @brief Definition of virtual constructor
  /// @return New instance of Scheduler
  [[nodiscard]] virtual std::unique_ptr<Scheduler> create() const = 0;

  /// @brief Spawn the threads of a graph
  /// @param cores Cores of every nodes in a graph
  /// @param waitForInitialization Wait for internal nodes to be initialized flags
  virtual void spawnThreads(std::set<core::abstraction::NodeAbstraction *> const &cores, bool waitForInitialization) = 0;

  /// @brief Method waiting for all graph's threads termination, called when Graph::waitForTermination() is called
  virtual void joinAll() = 0;
};

}
#endif //HEDGEHOG_SCHEDULER_H
