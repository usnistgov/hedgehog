//
// Created by Bardakoff, Alexandre (IntlAssoc) on 2019-04-03.
//

#ifndef HEDGEHOG_DEFAULT_SCHEDULER_H
#define HEDGEHOG_DEFAULT_SCHEDULER_H

#include <thread>
#include <vector>
#include <algorithm>
#include "../../core/scheduler/abstract_scheduler.h"

class DefaultScheduler : public AbstractScheduler {
 private:
  std::unique_ptr<std::vector<std::thread>> threads_ = nullptr;
  std::unique_ptr<std::vector<CoreNode *>> innerGraphs_ = nullptr;
 public:
  DefaultScheduler() {
    this->threads_ = std::make_unique<std::vector<std::thread>>();
    this->innerGraphs_ = std::make_unique<std::vector<CoreNode *>>();
  }

  ~DefaultScheduler() override = default;

  void spawnThreads(std::shared_ptr<std::multimap<std::string, std::shared_ptr<Node>>> &ptr) override {
    CoreNode *core = nullptr;
    for (auto &node : *(ptr.get())) {
      core = node.second->core();
      if (core->type() != NodeType::Graph) {
        threads_->emplace_back(&CoreNode::run, core);
      } else {
        innerGraphs_->push_back(core);
      }
    }
  }

  void joinAll() override {
    std::for_each(threads_->begin(), threads_->end(), [](std::thread &t) { t.join(); });
    for (CoreNode *innerGraph : *(this->innerGraphs_)) {
      innerGraph->joinThreads();
    }
  }

};

#endif //HEDGEHOG_DEFAULT_SCHEDULER_H
