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


#ifndef HEDGEHOG_CORE_RECEIVER_H
#define HEDGEHOG_CORE_RECEIVER_H

#include <queue>
#include <set>
#include <shared_mutex>
#include <algorithm>

#include "../../../node/core_node.h"
#include "core_slot.h"

/// @brief Hedgehog core namespace
namespace hh::core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward Declaration Core Sender
/// @tparam Input Sender Type
template<class Input>
class CoreSender;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Receiver Interface, receive one data type from CoreSender
/// @tparam Input Type of data received by the CoreReceiver
template<class Input>
class CoreReceiver : public virtual CoreNode {
 public:
  /// @brief Constructor with node name, node type and number of threads for the node
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Cluster number of threads
  CoreReceiver(std::string_view const &name, NodeType const type, size_t const numberThreads)
      : CoreNode(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreReceiver with type: " << (int) type << " and name: " << name)
  }

  /// @brief Default destructor
  ~CoreReceiver() override {HLOG_SELF(0, "Destructing CoreReceiver")}

  /// @brief Interface to add a CoreSender to the receiver
  /// @param sender CoreSender to add to this receiver
  virtual void addSender(CoreSender<Input> *sender) = 0;

  /// @brief Interface to remove a CoreSender from the receiver
  /// @param sender CoreSender to remove from this receiver
  virtual void removeSender(CoreSender<Input> *sender) = 0;

  /// @brief Interface to receive a data
  /// @param data Data received by this receiver
  virtual void receive(std::shared_ptr<Input> data) = 0;

  /// @brief Accessor to test emptiness on the receiver
  /// @return True if the receiver has no data, Else false
  virtual bool receiverEmpty() = 0;

  /// @brief Interface to get the number of element to be treated by this node for this type, by default return 0
  /// @return Get the number of element to be treated by this node for this type
  virtual size_t queueSize() { return 0; }

  /// @brief Accessor to  all receivers connected to this receiver
  /// @return All receivers connected to this receiver
  virtual std::set<CoreReceiver<Input> *> receivers() = 0;
};

}
#endif //HEDGEHOG_CORE_RECEIVER_H
