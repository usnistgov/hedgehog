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

#ifndef HEDGEHOG_CLEANABLE_ABSTRACTION_H_
#define HEDGEHOG_CLEANABLE_ABSTRACTION_H_

#include <stdexcept>
#include <unordered_set>

#include "../../../behavior/cleanable.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Abstraction for cleanable core
class CleanableAbstraction {
 private:
  hh::behavior::Cleanable *const cleanableNode_ = nullptr; ///< Link to cleanable node
 public:
  /// @brief Constructor used by the CoreGraph to have the handles to clean inner cleanable nodes
  CleanableAbstraction() = default;

  /// @brief Constructor used by cleanable nodes
  /// @param cleanableNode Node abstraction to clean
  /// @throw std::runtime_error if the node is not valid
  explicit CleanableAbstraction(behavior::Cleanable *const cleanableNode) : cleanableNode_(cleanableNode) {
    if(cleanableNode_ == nullptr){
      throw std::runtime_error("A cleanable abstraction should register a cleanable node.");
    }
  }

  /// @brief Default destructor
  virtual ~CleanableAbstraction() = default;

  /// @brief Gather cleanable node from the graph, and the state manager
  /// @param cleanableSet Mutable set to add inner nodes
  virtual void gatherCleanable(std::unordered_set<hh::behavior::Cleanable *> & cleanableSet){
    cleanableSet.insert(cleanableNode_);
  }

};
}
}
}
#endif //HEDGEHOG_CLEANABLE_ABSTRACTION_H_
