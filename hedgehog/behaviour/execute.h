//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_EXECUTE_H
#define HEDGEHOG_EXECUTE_H

#include <memory>

template<class Input>
class Execute {
 public:
  virtual ~Execute() = default;
  virtual void execute(std::shared_ptr<Input>) = 0;
};

#endif //HEDGEHOG_EXECUTE_H
