//
// Created by 775backup on 2019-04-16.
//

#ifndef HEDGEHOG_NODE_H
#define HEDGEHOG_NODE_H

#include <memory>

/// @brief Hedgehog main namespace
namespace hh {
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Hedgehog core namespace
namespace core {
/// @brief Forward declaration of CoreNode
class CoreNode;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/// @brief Hedgehog behavior namespace
namespace behavior {
/// @brief Node Behavior definition
/// @details Node has a core, can add information to print in the dot file, and can define the way they terminate
class Node {
 public:
  /// @brief Core Accessor
  /// @attention Should not be used by library user, only by the developer that wants to add node or functionality to
  /// the library
  /// @return Node Core
  virtual std::shared_ptr<core::CoreNode> core() = 0;

  /// @brief Adds node information to print in the dot file
  /// @return A string with extra information to show in the dot file
  [[nodiscard]] virtual std::string extraPrintingInformation() const { return ""; }

  /// @brief Determine if the node can terminate
  /// @details Return true if the node should terminate, False otherwise
  /// @attention Should override to break a cycle in a graph.
  /// @return True if the node should terminate, else False.
  virtual bool canTerminate() { return true; };
};
}
}
#endif //HEDGEHOG_NODE_H
