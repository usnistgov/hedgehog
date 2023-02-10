

// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.

#ifndef HEDGEHOG_MEMORY_MANAGER_H_
#define HEDGEHOG_MEMORY_MANAGER_H_

#pragma once

#include "../managed_memory.h"
#include "abstract_memory_manager.h"
#include "../../../tools/concepts.h"

/// @brief Hedgehog main namespace
namespace hh {

/// Base memory manager
/// @brief Memory manager with default created managed object
/// @tparam T Type managed by the memory manager
template<tool::ManageableMemory T>
class MemoryManager : public AbstractMemoryManager {
 public:
  /// Create a memory manager with a certain capacity
  /// @param capacity Capacity of the memory manager
  explicit MemoryManager(size_t const &capacity) : AbstractMemoryManager(capacity) {}

  /// @brief Default destructor
  ~MemoryManager() override = default;

  /// Default copy method
  /// @attention Need to be overloaded by the end user if the memory manager is inherited
  /// @return Create another MemoryManager from this
  std::shared_ptr<AbstractMemoryManager> copy() override {
    return std::make_shared<MemoryManager>(this->capacity());
  }

  /// @brief User-definable initialization step for a memory manager
  virtual void initializeMemoryManager() {}

  /// Get managed memory from the pool
  /// @brief Get managed memory from the pool. If the pool is empty, the call will block until a new element get
  /// available.
  /// @return A managed memory
  std::shared_ptr<ManagedMemory> getManagedMemory() final   {
    std::shared_ptr<ManagedMemory> managedMemory = nullptr;
    managedMemory = this->pool()->pop_front();
    managedMemory->preProcess();
    return managedMemory;
  }

 private:
  /// Recycling mechanism for managed memory
  /// @brief Thread safe recycle that will in sequence:
  ///     - calls ManagedMemory::postProcess()
  ///     - if ManagedMemory::canBeRecycled() returns true
  ///     - calls ManagedMemory::clean()
  /// @param managedMemory Type of the managed memory
  void recycleMemory(std::shared_ptr<ManagedMemory> const &managedMemory) final {
    memoryManagerMutex_.lock();
    managedMemory->postProcess();
    if (managedMemory->canBeRecycled()) {
      this->profiler()->addReleaseMarker();
      managedMemory->clean();
      this->pool()->push_back(managedMemory);
    }
    memoryManagerMutex_.unlock();
  }

  /// Initialize the memory manager
  /// @brief Thread safe initialization, fill the pool with default constructed object
  void initialize() final {
    memoryManagerMutex_.lock();
    if (!this->isInitialized()) {
      this->initialized();
      std::for_each(
          this->pool()->begin(), this->pool()->end(),
          [this](std::shared_ptr<ManagedMemory> &emptyShared) {
            emptyShared = std::make_shared<T>();
            emptyShared->memoryManager(this);
          }
      );
      initializeMemoryManager();
    }
    memoryManagerMutex_.unlock();
  }

  /// Getter to real managed type as string
  /// @return String of the real managed type
  [[nodiscard]] std::string managedType() const final { return hh::tool::typeToStr<T>(); }
};
}
#endif //HEDGEHOG_MEMORY_MANAGER_H_
