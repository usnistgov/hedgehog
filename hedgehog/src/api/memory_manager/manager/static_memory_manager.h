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

#ifndef HEDGEHOG_STATIC_MEMORY_MANAGER_H
#define HEDGEHOG_STATIC_MEMORY_MANAGER_H

#pragma once

#include "../managed_memory.h"
#include "abstract_memory_manager.h"

/// @brief Hedgehog main namespace
namespace hh {

/// Static memory manager
/// @brief Static Memory manager with custom created managed object. The type T managed has to present a constructor
/// with a signature that matches exactly the template argument list Args...
/// @tparam T Type managed
/// @tparam Args List of types that should match the constructor of T parameters
template<class T, class ...Args>
class StaticMemoryManager : public AbstractMemoryManager {
  static_assert(std::is_base_of_v<ManagedMemory, T>, "The type managed by the StaticMemoryManager should derive from hh::ManagedMemory.");
  static_assert(std::is_constructible_v<T, Args...>, "The type managed by the StaticMemoryManager should be constructible with Args type(s).");
 private:
  std::tuple<Args...> args_ = {}; ///< Values to pass to the constructor
 public:
  /// Constructor to a static memory manager.
  ///  @brief Construct a static memory manager from its capacity, and the arguments used to construct all elements
  /// in the pool.
  /// @param capacity Memory manager capacity
  /// @param args Arguments used to construct every elements in the pool
  explicit StaticMemoryManager(size_t const &capacity, Args ... args)
      : AbstractMemoryManager(capacity), args_(std::forward<Args>(args)...) {}

  /// @brief Copy constructor
  /// @param rhs StaticMemoryManager to construct from
  StaticMemoryManager(StaticMemoryManager<T, Args...> const & rhs)
      : AbstractMemoryManager(rhs.capacity()), args_(rhs.args_) {}

  /// @brief Default destructor
  ~StaticMemoryManager() override = default;

  /// Get a managed memory from the pool
  /// @brief Get a managed memory from the pool. If the pool is empty, the call will block until a new element get
  /// available.
  /// @return A managed memory
  std::shared_ptr<ManagedMemory> getManagedMemory() final {
    std::shared_ptr<ManagedMemory> managedMemory = nullptr;
    managedMemory = this->pool()->pop_front();
    managedMemory->preProcess();
    return managedMemory;
  }

  /// Default copy method
  /// @attention Need to be overloaded by the end user if the memory manager is inherited
  /// @return Create another MemoryManager from this
  std::shared_ptr<AbstractMemoryManager> copy() override {
    return std::make_shared<StaticMemoryManager<T, Args...>>(*this);
  }

  /// @brief User-definable initialization step for a memory manager
  virtual void initializeMemoryManager() {}

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
      managedMemory->clean();
      this->pool()->push_back(managedMemory);
    }
    memoryManagerMutex_.unlock();
  }

  /// Initialize the memory manager
  /// @brief Thread safe initialization, fill the pool with custom constructed object
  void initialize() final {
    memoryManagerMutex_.lock();
    if (!this->isInitialized()) {
      this->initialized();
      initialize(std::make_index_sequence<sizeof...(Args)>());
      this->initializeMemoryManager();
    }
    memoryManagerMutex_.unlock();
  }

  /// @brief Initialize implementation using the tuple of arguments stored
  /// @tparam Is Index of the arguments tuple
  template<size_t... Is>
  void initialize(std::index_sequence<Is...>) {
    std::for_each(
        this->pool()->begin(), this->pool()->end(),
        [this](std::shared_ptr<ManagedMemory> &emptyShared) {
          emptyShared = std::make_shared<T>(std::get<Is>(args_)...);
          emptyShared->memoryManager(this);
        }
    );
  }

  /// Getter to real managed type as string
  /// @return String of the real managed type
  [[nodiscard]] std::string managedType() const final { return hh::tool::typeToStr<T>(); }

};
}

#endif //HEDGEHOG_STATIC_MEMORY_MANAGER_H
