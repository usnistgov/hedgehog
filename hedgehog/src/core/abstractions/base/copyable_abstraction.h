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

#ifndef HEDGEHOG_COPYABLE_ABSTRACTION_H
#define HEDGEHOG_COPYABLE_ABSTRACTION_H
#include <memory>
#include <sstream>
#include "../../../tools/concepts.h"
#include "node/node_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Core abstraction for copyable nodes
/// @tparam CopyableNode Type of node to be copied
template<tool::CopyableNode CopyableNode>
class CopyableAbstraction {
 private:
  CopyableNode *const copyableNode_ = nullptr; ///< Pointer to copyable node abstraction
  std::unique_ptr<std::set<std::shared_ptr<CopyableNode>>> nodeCopies_ = nullptr; ///< Set of copies of the node

 public:
  /// @brief Constructor using a node abstraction
  /// @param copyableNode Node to copy
  explicit CopyableAbstraction(CopyableNode *const copyableNode)
  : copyableNode_(copyableNode),
    nodeCopies_(std::make_unique<std::set<std::shared_ptr<CopyableNode>>>()){}

  /// @brief Default destructor
  virtual ~CopyableAbstraction() = default;

 protected:

  /// @brief Interface to call user-defined copy method
  /// @return Copy of the node
  /// @throw std::runtime_error A copy of the node is not valid
  std::shared_ptr<CopyableNode> callCopy() {
    auto copy = copyableNode_->copy();

    if (copy == nullptr) {
      std::ostringstream oss;
      if (auto node = dynamic_cast<NodeAbstraction *>(this))
        oss
            << "A copy for the node \"" << node->name()
            << "\" has been invoked but return nullptr. To fix this error, overload the "
            << tool::typeToStr<CopyableNode>()
            << "::copy function and return a valid object.";
      else {
        oss
            << "A copy for the node has been invoked but return nullptr. To fix this error, overload the "
            << tool::typeToStr<CopyableNode>()
            << "::copy function and return a valid object.";
      }
      throw (std::runtime_error(oss.str()));
    }

    if (nodeCopies_->find(copy) != nodeCopies_->cend()) {
      std::ostringstream oss;
      if (auto node = dynamic_cast<NodeAbstraction *>(this)) {
        oss << "A copy for the node \"" << node->name()
            << "\" has been invoked and return a copy already registered. Each copy of a node should be different.";
      }else {
        oss << "A copy for the node has been invoked and return a copy already registered. Each copy of a node "
               "should be different.";
      }
      throw (std::runtime_error(oss.str()));
    }
    nodeCopies_->insert(copy);
    return copy;
  }
};
}
}
}

#endif //HEDGEHOG_COPYABLE_ABSTRACTION_H
