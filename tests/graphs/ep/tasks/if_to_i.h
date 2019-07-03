//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_IF_TO_I_H
#define HEDGEHOG_IF_TO_I_H

#include "../../../../hedgehog/hedgehog.h"

class IFToI : public AbstractTask<int, int, float> {
 public:
  IFToI(std::string_view const &name) : AbstractTask(name, 3) {}
  virtual ~IFToI() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    addResult(std::make_shared<int>(2));
  }
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
    addResult(std::make_shared<int>(2));
  }
  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
    return std::make_shared<IFToI>(this->name());
  }
};

#endif //HEDGEHOG_IF_TO_I_H
