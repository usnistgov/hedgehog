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


#ifndef HEDGEHOG_CORE_SLOT_H
#define HEDGEHOG_CORE_SLOT_H

#include "../../../node/core_node.h"

/// @brief Hedgehog core namespace
namespace hh::core {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward declaration of CoreNotifier
class CoreNotifier;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Slot interface, receive notification from CoreNotifier
class CoreSlot : public virtual CoreNode {
 public:

  /// @brief Core slot constructor
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Node number of threads
  CoreSlot(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreSlot with type: " << (int) type << " and name: " << name)
  }

  /// @brief Core Slot destructor
  ~CoreSlot() override {HLOG_SELF(0, "Destructing CoreSlot")}

  /// @brief Interface to add a CoreNotifier to this slot
  /// @param notifier CoreNotifier to add to this slot
  virtual void addNotifier(CoreNotifier *notifier) = 0;

  /// @brief Interface to remove a CoreNotifier from this slot
  /// @param notifier CoreNotifier to remove from this notifier
  virtual void removeNotifier(CoreNotifier *notifier) = 0;

  /// @brief Test if notifiers are connected to this slot
  /// @return True if at least one notifier is connected to this slot, else False
  virtual bool hasNotifierConnected() = 0;

  /// @brief Return the number of notifiers connected to this slot
  /// @return The number of notifiers connected to this slot
  [[nodiscard]] virtual size_t numberInputNodes() const = 0;

  /// @brief Interface to define what the node do when it receive a signal
  virtual void wakeUp() = 0;

  /// @brief Interface to define how the node wait for a signal, and return if the node is terminated
  /// @return True if the node is terminated, else False
  virtual bool waitForNotification() = 0;
};

}
#endif //HEDGEHOG_CORE_SLOT_H
