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
  virtual CoreNode *core() = 0;
  virtual bool canTerminate() { return true; };
};

#endif //HEDGEHOG_NODE_H
