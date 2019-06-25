//
// Created by anb22 on 6/21/19.
//

#ifndef HEDGEHOG_ABSTRACT_DYNAMIC_MEMORY_MANAGER_H
#define HEDGEHOG_ABSTRACT_DYNAMIC_MEMORY_MANAGER_H
#include "../../behaviour/memory_manager/abstract_memory_manager.h"

template<class MANAGEDDATA>
class AbstractDynamicMemoryManager : public AbstractMemoryManager<MANAGEDDATA> {
 public:
  AbstractDynamicMemoryManager() = delete;
  explicit AbstractDynamicMemoryManager(size_t const &poolSize) : AbstractMemoryManager<MANAGEDDATA>(poolSize) {}
  virtual void initializeDynamicMemoryManager() {};

 private:
  void initialize() final {
    if (!this->isInitialized()) {
      this->setInitialized();
      this->pool()->initialize(this->poolSize());
      std::for_each(
          this->pool()->begin(), this->pool()->end(),
          [this](std::shared_ptr<MANAGEDDATA> &emptyShared) { emptyShared->memoryManager(this); }
      );
      initializeDynamicMemoryManager();
    }
  }
};

#endif //HEDGEHOG_ABSTRACT_DYNAMIC_MEMORY_MANAGER_H
