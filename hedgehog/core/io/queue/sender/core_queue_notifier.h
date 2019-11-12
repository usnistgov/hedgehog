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


#ifndef HEDGEHOG_CORE_QUEUE_NOTIFIER_H
#define HEDGEHOG_CORE_QUEUE_NOTIFIER_H

#include "../../base/sender/core_notifier.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Notifier of CoreQueueSlot
class CoreQueueNotifier : public virtual CoreNotifier {
 private:
  std::shared_ptr<std::set<CoreSlot *>> slots_ = nullptr; ///< Set of connected slots

 public:
  /// @brief CoreQueueNotifier constructor
  /// @param name Node's name
  /// @param type Node's type
  /// @param numberThreads Node's number of thread
  CoreQueueNotifier(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNotifier(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreQueueNotifier with type: " << (int) type << " and name: " << name)
    slots_ = std::make_shared<std::set<CoreSlot *>>();
  }

  /// @brief CoreQueueNotifier destructor
  ~CoreQueueNotifier() override {HLOG_SELF(0, "Destructing CoreQueueNotifier")}

  /// @brief Connected slots accessor
  /// @return Connected slots
  [[nodiscard]] std::shared_ptr<std::set<CoreSlot *>> const &slots() const { return slots_; }

  /// @brief Add a slot to the set of connected slots
  /// @param slot Slot to connect
  void addSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Add Slot " << slot->name() << "(" << slot->id() << ")")
    this->slots()->insert(slot);
  }

  /// @brief Remove a slot from the set of connected slots
  /// @param slot Slot to remove
  void removeSlot(CoreSlot *slot) override {
    HLOG_SELF(0, "Remove Slot " << slot->name() << "(" << slot->id() << ")")
    this->slots_->erase(slot);
  }

  /// @brief Notify all slots that the node is terminated
  void notifyAllTerminated() override {
    HLOG_SELF(2, "Notify all terminated")
    std::for_each(this->slots()->begin(), this->slots()->end(), [this](CoreSlot *s) { s->removeNotifier(this); });
    std::for_each(this->slots()->begin(), this->slots()->end(), [](CoreSlot *s) { s->wakeUp(); });
  }

  /// @brief Copy the inner structure of the notifier (set of slots and connections)
  /// @param rhs CoreQueueNotifier to copy to this
  void copyInnerStructure(CoreQueueNotifier *rhs) {
    HLOG_SELF(0, "Copy Cluster CoreQueueNotifier information from " << rhs->name() << "(" << rhs->id() << ")")
    for (CoreSlot *slot : *(rhs->slots_)) { slot->addNotifier(this); }
    this->slots_ = rhs->slots_;
  }
};

}
#endif //HEDGEHOG_CORE_QUEUE_NOTIFIER_H
