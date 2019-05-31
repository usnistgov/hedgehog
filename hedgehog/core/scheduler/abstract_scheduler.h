//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_ABSTRACT_SCHEDULER_H
#define HEDGEHOG_ABSTRACT_SCHEDULER_H

#include <map>
#include <memory>
#include "../../behaviour/node.h"
#include "../../core/node/core_node.h"

class AbstractScheduler {
 public:
  AbstractScheduler() = default;
  virtual ~AbstractScheduler() = default;
  virtual void spawnThreads(std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>> &) = 0;
  virtual void joinAll() = 0;
};

#endif //HEDGEHOG_ABSTRACT_SCHEDULER_H
