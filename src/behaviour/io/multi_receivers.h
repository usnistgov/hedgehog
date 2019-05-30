//
// Created by anb22 on 5/2/19.
//


#ifndef HEDGEHOG_MULTI_RECEIVERS_H
#define HEDGEHOG_MULTI_RECEIVERS_H

#include "../node.h"

template<class ...Inputs>
class MultiReceivers : public virtual Node {
 public:
  using inputs_t = std::tuple<Inputs...>;
};

#endif //HEDGEHOG_MULTI_RECEIVERS_H
