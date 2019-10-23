//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

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
