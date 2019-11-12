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


#ifndef HEDGEHOG_CORE_MULTI_RECEIVERS_H
#define HEDGEHOG_CORE_MULTI_RECEIVERS_H

#include "core_receiver.h"
#include "core_slot.h"
#include "../../../node/core_node.h"

/// @brief Hedgehog core namespace
namespace hh::core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of sender with a queue
/// @tparam Input Sender's input type
template<class Input>
class CoreQueueSender;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Multi receiver interface, gather multiple CoreReceiver
/// @tparam Inputs Multi receiver's inputs type
template<class ...Inputs>
class CoreMultiReceivers : public virtual CoreSlot, public virtual CoreReceiver<Inputs> ... {
 public:
  /// @brief CoreMultiReceivers constructor
  /// @param name Node's name
  /// @param type Node's type
  /// @param numberThreads Node's number of thread
  CoreMultiReceivers(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreSlot(name, type, numberThreads), CoreReceiver<Inputs>(name, type, numberThreads)... {
    HLOG_SELF(0, "Creating CoreMultiReceivers with type: " << (int) type << " and name: " << name)
  }

  /// @brief CoreMultiReceivers destructor
  ~CoreMultiReceivers() override {HLOG_SELF(0, "Destructing CoreMultiReceivers")}

  /// @brief Test if all receivers are empty
  /// @return True of all receivers are empty, else false
  virtual bool receiversEmpty() = 0;

  /// @brief Compute all receivers queue size
  /// @return All receivers queue size
  virtual size_t totalQueueSize() { return 0; }

  /// @brief Remove all coreNode's senders from this
  /// @param coreNode CoreNode representing the senders that will be removed from this
  void removeForAllSenders(CoreNode *coreNode) {
    (this->removeForAllSendersConditional<Inputs>(coreNode), ...);
  }

 private:
  /// @brief Remove all coreNode's senders from this for a specific Input type
  /// @tparam Input Sender Input's type
  /// @param coreNode CoreNode representing the senders that will be removed from this
  template<class Input>
  void removeForAllSendersConditional(CoreNode *coreNode) {
    // If coreNode *is* a CoreQueueSender for a specific Input type
    if (auto temp = dynamic_cast<CoreQueueSender<Input> *>(coreNode)) {
      // Remove the sender for this multi receiver
      static_cast<CoreReceiver<Input> *>(this)->removeSender(temp);
    }
  }

};

}
#endif //HEDGEHOG_CORE_MULTI_RECEIVERS_H
