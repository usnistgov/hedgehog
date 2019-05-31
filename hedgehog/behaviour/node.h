//
// Created by 775backup on 2019-04-16.
//

#ifndef HEDGEHOG_NODE_H
#define HEDGEHOG_NODE_H

class CoreNode;

class Node {
 public:
  virtual ~Node() = default;
  virtual CoreNode *getCore() = 0;
};

#endif //HEDGEHOG_NODE_H
