//
// Created by 775backup on 2019-04-16.
//

#ifndef HEDGEHOG_NODE_H
#define HEDGEHOG_NODE_H

#include <memory>
class CoreNode;

class Node {
 public:
  virtual ~Node() = default;
  virtual std::shared_ptr<CoreNode> core() = 0;
  virtual std::string extraPrintingInformation() const { return ""; }
  virtual bool canTerminate() { return true; };
};

#endif //HEDGEHOG_NODE_H
