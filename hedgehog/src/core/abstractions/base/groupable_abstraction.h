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



#ifndef HEDGEHOG_COPYABLE_ABSTRACTION_H
#define HEDGEHOG_COPYABLE_ABSTRACTION_H

#include <ostream>

#include "any_groupable_abstraction.h"
#include "copyable_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {
/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief Typed abstraction for groupable node
/// @tparam CopyableNode Type of the node to copy and group
/// @tparam CopyableCore Type of the core to copy and group
template<tool::CopyableNode CopyableNode, class CopyableCore>
class GroupableAbstraction :
    public CopyableAbstraction<CopyableNode>,
    public AnyGroupableAbstraction {
 private:
  int
      threadId_ = 0, ///< Thread id
      numberThreadsCreated_ = 1; ///< Number of thread created for the group

 public:
  /// @brief Constructor using the node abstraction to call the user-defined copy and the number of threads
  /// @param copyableNode Type of the node to copy and group
  /// @param numberThreads Number of threads (number of nodes) in the group
  GroupableAbstraction(CopyableNode *const copyableNode, size_t const &numberThreads)
      : CopyableAbstraction<CopyableNode>(copyableNode),
        AnyGroupableAbstraction(numberThreads) {}

  /// @brief Default destructor
  ~GroupableAbstraction() override = default;

  /// @brief Accessor to thread id
  /// @return Thread id
  [[nodiscard]] int threadId() const { return threadId_;}


  /// @brief Call the used-defined copy and register the copy in the group
  /// @return Copy of the node
  /// @throw std::runtime_error a copy is ill-formed
  std::shared_ptr<CopyableNode> callCopyAndRegisterInGroup() {
    auto copy = this->callCopy();

    auto copyableCore = dynamic_cast<AnyGroupableAbstraction *>(copy->core().get());

    if (copyableCore == nullptr) {
      std::ostringstream oss;
      if (auto node = dynamic_cast<NodeAbstraction *>(this))
        oss << "A copy for the node \"" << node->name()
            << "\" has a core that does not have the right structure (missing inheritance to GroupableAbstraction).";
      else {
        oss << "A copy for the node has a core that does not have the right structure (missing inheritance to "
               "GroupableAbstraction).";
      }
      throw (std::runtime_error(oss.str()));
    }

    copyableCore->groupRepresentative(this);
    this->group()->insert(copyableCore);
    copyableCore->group(this->group());

    auto copyAsCopyable = dynamic_cast<GroupableAbstraction<CopyableNode, CopyableCore>*>(copyableCore);
    if (copyAsCopyable != nullptr) {
      copyAsCopyable->threadId_ = numberThreadsCreated_++;
    }else {
      std::ostringstream oss;
      oss << "A copy for the node has a core that does not have the right structure (missing inheritance to GroupableAbstraction).";
      throw (std::runtime_error(oss.str()));
    }

    return copy;
  }

  /// @brief Copy the inner structure of the core
  /// @param copyableCore Core to copy the inner structure from
  virtual void copyInnerStructure(CopyableCore *copyableCore) = 0;

};
}
}
}
#endif //HEDGEHOG_COPYABLE_ABSTRACTION_H
