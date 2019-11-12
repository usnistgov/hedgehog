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


#ifndef HEDGEHOG_CORE_NOTIFIER_H
#define HEDGEHOG_CORE_NOTIFIER_H

#include <memory>
#include <set>
#include <algorithm>
#include <ostream>

#include "../receiver/core_slot.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Core Notifier interface, emit notification to CoreSlot
class CoreNotifier : public virtual CoreNode {
 public:
  /// @brief Deleted default constructor
  CoreNotifier() = delete;

  /// @brief Notifier constructor
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Node number of threads
  CoreNotifier(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreNotifier with type: " << (int) type << " and name: " << name)
  }

  /// @brief Notifier destructor
  ~CoreNotifier() override {HLOG_SELF(0, "Destructing CoreNotifier")}

  /// @brief Interface to add a CoreSlot to this notifier
  /// @param slot CoreSlot to add to this notifier
  virtual void addSlot(CoreSlot *slot) = 0;

  /// @brief Interface to remove a CoreSlot from this notifier
  /// @param slot CoreSlot to remove from this notifier
  virtual void removeSlot(CoreSlot *slot) = 0;

  /// @brief Notify all slot that the node is terminated
  virtual void notifyAllTerminated() = 0;
};

}
#endif //HEDGEHOG_CORE_NOTIFIER_H
