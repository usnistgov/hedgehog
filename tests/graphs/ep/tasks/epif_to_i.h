//
// Created by anb22 on 7/2/19.
//

#ifndef HEDGEHOG_EPIF_TO_I_H
#define HEDGEHOG_EPIF_TO_I_H

#include "../../../../hedgehog/hedgehog.h"

class EPIFToI : public AbstractTask<int, int, float> {
 public:
  EPIFToI(std::string_view const &name) : AbstractTask(name, 3) {}
  virtual ~EPIFToI() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    addResult(std::make_shared<int>(2));
  }
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
    addResult(std::make_shared<int>(2));
  }
  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
    return std::make_shared<EPIFToI>(this->name());
  }

};

#endif //HEDGEHOG_EPIF_TO_I_H
