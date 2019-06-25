//
// Created by anb22 on 6/21/19.
//

#ifndef HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H
#define HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H

#include <memory>
#include <mutex>
#include <condition_variable>

#include "../../tools/traits.h"
#include "../../tools/logger.h"
#include "../../api/memory_manager/memory_data.h"
#include "pool.h"

template<class MANAGEDDATA>
class AbstractMemoryManager {
 private:
  size_t poolSize_ = 0;
  int deviceId_ = 0;
  std::unique_ptr<Pool<MANAGEDDATA>> pool_ = {};
  bool initialized_ = false;

 public:
  AbstractMemoryManager() = delete;
  explicit AbstractMemoryManager(size_t const &poolSize)
      : poolSize_(poolSize),
        deviceId_(0),
        pool_(std::make_unique<Pool<MANAGEDDATA>>()) {
    static_assert(
        HedgehogTraits::is_managed_memory_v<MANAGEDDATA>,
        "The type given to the memory manager should inherit \"MemoryData\", and be default constructible!");
  };

  virtual ~AbstractMemoryManager() = default;
  virtual std::shared_ptr<AbstractMemoryManager<MANAGEDDATA>> copy() = 0;

  void deviceId(int deviceId) { deviceId_ = deviceId; }

  std::shared_ptr<MANAGEDDATA> getData() {
    std::shared_ptr<MANAGEDDATA> managedMemory;
    HLOG(4, "StaticMemoryManager memory pool size = " << this->pool()->queue().size())
    managedMemory = this->pool()->pop_front();
    HLOG(4,
         "StaticMemoryManager After waiting: received: " << managedMemory << " pSize: " << (int) (this->pool()->size()))
    return managedMemory;
  };

  void releaseData(std::shared_ptr<MemoryData<MANAGEDDATA>> managedMemory) {
    managedMemory->lock();
    if (this->canRecycle(std::dynamic_pointer_cast<MANAGEDDATA>(managedMemory))) {
      managedMemory->recycle();
      this->pool_->push_back(std::dynamic_pointer_cast<MANAGEDDATA>(managedMemory));
    }
    managedMemory->unlock();
  };

  virtual bool canRecycle(std::shared_ptr<MANAGEDDATA> const &) = 0;
  virtual void initialize() {};

 protected:
  std::unique_ptr<Pool<MANAGEDDATA>> const &pool() const { return pool_; }
  size_t poolSize() const { return poolSize_; }
  int deviceId() const { return deviceId_; }
  bool isInitialized() const { return initialized_; }
  void setInitialized() { initialized_ = true; }
};
#endif //HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H
