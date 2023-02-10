//  NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
//  software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
//  derivative works of the software or any portion of the software, and you may copy and distribute such modifications
//  or works. Modified works should carry a notice stating that you changed the software and should note the date and
//  nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
//  source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
//  EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
//  WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
//  CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
//  THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
//  are solely responsible for determining the appropriateness of using and distributing the software and you assume
//  all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
//  with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
//  operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
//  damage to property. The software developed by NIST employees is not subject to copyright protection within the
//  United States.



#ifndef HEDGEHOG_COPYABLE_H
#define HEDGEHOG_COPYABLE_H

#include <memory>
#include "node.h"
#include "../core/abstractions/base/any_groupable_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog behavior namespace
namespace behavior {

/// @brief Copy interface used to copy a node when either a group of nodes is created or a node is duplicated when an
/// execution pipeline is created
/// @tparam NodeType Type of node
template<class NodeType>
class Copyable {
 private:
  size_t const numberThreads_ = 0; ///< Number of threads
 public:

  /// @brief Copyable constructor, set the number of threads for a node
  /// @param numberThreads Number of threads attached to a node to form a group
  explicit Copyable(size_t const numberThreads) : numberThreads_(numberThreads) {}

  /// @brief Default destructor
  virtual ~Copyable() = default;

  /// @brief Number of threads accessor
  /// @return Number of threads attached to a node
  [[nodiscard]] size_t numberThreads() const { return numberThreads_; }

  /// @brief Get the group of nodes that hold the current nodes
  /// @return Group of nodes that hold the current nodes
  std::vector<NodeType const *> group() const {
    std::vector<NodeType const *> ret;
    std::shared_ptr<hh::core::abstraction::NodeAbstraction> core = dynamic_cast<Node const *>(this)->core();
    if (core) {
      auto groupableNode = std::dynamic_pointer_cast<core::abstraction::AnyGroupableAbstraction>(core);
      if (groupableNode) {
        auto group = groupableNode->groupAsNodes();
        for (auto &coresInGroup : *group) {
          if (coresInGroup){ ret.push_back(dynamic_cast<NodeType *>(coresInGroup->node())); }
          else { return {}; }
        }
      } else { // if (groupable)
        std::ostringstream oss;
        oss << "The core attached to the node " << this << " does not derives from hh::core::abstraction::AnyGroupableAbstraction.";
        throw(std::runtime_error(oss.str()));
      }
      return ret;
    } else { // if (core)
      std::ostringstream oss;
      oss << "The core attached to the node " << this << " is ill-formed.";
      throw std::runtime_error(oss.str());
    }
  }

  /// @brief Copy method called to either create a group of node or duplicate a node when an execution pipeline is
  /// created
  /// @return A new instance of the node
  virtual std::shared_ptr<NodeType> copy() { return nullptr; }
};
}
}
#endif //HEDGEHOG_COPYABLE_H
