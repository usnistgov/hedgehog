//
// Created by anb22 on 6/10/19.
//

#ifndef HEDGEHOG_CORE_EXECUTE_H
#define HEDGEHOG_CORE_EXECUTE_H
#include <memory>

template<class NodeInput>
class CoreExecute {
 public:
  virtual void callExecute(std::shared_ptr<NodeInput> data) = 0;
};

#endif //HEDGEHOG_CORE_EXECUTE_H
