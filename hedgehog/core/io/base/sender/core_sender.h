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


#ifndef HEDGEHOG_CORE_SENDER_H
#define HEDGEHOG_CORE_SENDER_H

#include <set>

#include "core_notifier.h"
#include "../../../node/core_node.h"
#include "../receiver/core_receiver.h"

/// @brief Hedgehog core namespace
namespace hh::core {

/// @brief Sender interface, send data to CoreReceiver
/// @tparam Output Data type sent to CoreReceiver
template<class Output>
class CoreSender : public virtual CoreNotifier {
 public:

  /// @brief CoreSender constructor
  /// @param name Node name
  /// @param type Node type
  /// @param numberThreads Node number of threads
  CoreSender(std::string_view const &name, NodeType const type, size_t const numberThreads) :
      CoreNode(name, type, numberThreads) {
    HLOG_SELF(0, "Creating CoreSender with type: " << (int) type << " and name: " << name)
  }

  /// @brief CoreSender destructor
  ~CoreSender() override {HLOG_SELF(0, "Destructing CoreSender")}

  /// @brief Interface to add a CoreReceiver to this CoreSender
  /// @param receiver CoreReceiver to add to this CoreSender
  virtual void addReceiver(CoreReceiver <Output> *receiver) = 0;

  /// @brief Interface to remove a CoreReceiver from this CoreSender
  /// @param receiver Receiver to CoreReceiver from this CoreSender
  virtual void removeReceiver(CoreReceiver <Output> *receiver) = 0;

  /// @brief Interface to send and notify a data to all connected CoreReceiver
  /// @param data data to send
  virtual void sendAndNotify(std::shared_ptr<Output> data) = 0;

  /// @brief Get inner CoreSender represented by this one in the case of outer graph for example
  /// @return Inner CoreSender represented by this one
  virtual std::set<CoreSender<Output> *> getSenders() = 0;

  /// @brief Duplicate all the edges from this to it's copy duplicateNode
  /// @param duplicateNode Node to connect
  /// @param correspondenceMap Correspondence  map from base node to copy
  void duplicateEdge(CoreNode *duplicateNode,
                     std::map<CoreNode *, std::shared_ptr<CoreNode>> &correspondenceMap) override {
    for (auto sender : this->getSenders()) {
      sender->duplicateEdge(duplicateNode, correspondenceMap);
    }
  }
};

}
#endif //HEDGEHOG_CORE_SENDER_H
