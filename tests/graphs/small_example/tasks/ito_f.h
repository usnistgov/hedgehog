//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_I_TO_F_H
#define HEDGEHOG_I_TO_F_H

#include "../../../../hedgehog/hedgehog.h"

class IToF : public AbstractTask<float, int, double, char> {
 public:
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {}
  void execute([[maybe_unused]]std::shared_ptr<double> ptr) override {}
  void execute([[maybe_unused]]std::shared_ptr<char> ptr) override {}
};

#endif //HEDGEHOG_I_TO_F_H
