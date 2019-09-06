//
// Created by anb22 on 6/21/19.
//

#ifndef HEDGEHOG_ABSTRACT_STATIC_MEMORY_MANAGER_H
#define HEDGEHOG_ABSTRACT_STATIC_MEMORY_MANAGER_H

#include "abstract_memory_manager.h"

template<class MANAGEDDATA>
class AbstractStaticMemoryManager : public AbstractMemoryManager<MANAGEDDATA> {
 public:
  AbstractStaticMemoryManager() = delete;
  explicit AbstractStaticMemoryManager(size_t const &poolSize) : AbstractMemoryManager<MANAGEDDATA>(poolSize) {}
  virtual void allocate(std::shared_ptr<MANAGEDDATA>) = 0;
  virtual void initializeStaticMemoryManager() {}

 private:
  void initialize() final {
    if (!this->isInitialized()) {
      this->setInitialized();
      this->pool()->initialize(this->poolSize());
      std::for_each(
          this->pool()->begin(), this->pool()->end(),
          [this](std::shared_ptr<MANAGEDDATA> &emptyShared) {
            emptyShared->memoryManager(this);
            this->allocate(emptyShared);
          }
      );
      initializeStaticMemoryManager();
    }
  }

};

#endif //HEDGEHOG_ABSTRACT_STATIC_MEMORY_MANAGER_H
