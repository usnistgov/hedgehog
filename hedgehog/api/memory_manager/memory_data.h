//
// Created by anb22 on 6/21/19.
//

#ifndef HEDGEHOG_MEMORY_DATA_H
#define HEDGEHOG_MEMORY_DATA_H

#include <memory>
#include "abstract_memory_manager.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Memory data interface to use a data type in a Memory manager (AbstractMemoryManager or StaticMemoryManager)
/// @details To declare a data A  or B using the interface MemoryData, it can be written as:
/// @code
/// class A : public MemoryData<A>{};
/// //Or
/// template <class T>
/// class B : public MemoryData<B<T>>{};
/// @endcode
/// The data will be served from the pool to the task through the AbstractMemoryManager, and can be return with
/// MemoryData::returnToMemoryManager().
/// When returned, the MemoryData::used() and MemoryData::canBeRecycled() are called. If MemoryData::canBeRecycled()
/// returns true, MemoryData::recycle() is called and it is returned to the pool and made available by its
/// AbstractMemoryManager
///
/// @par Virtual Functions
/// MemoryData::used
/// MemoryData::canBeRecycled
/// MemoryData::recycle
///
/// @tparam ManagedMemory type of data to managed
template<class ManagedMemory>
class MemoryData : public std::enable_shared_from_this<MemoryData<ManagedMemory>> {
 private:
  AbstractMemoryManager <ManagedMemory> *memoryManager_ = nullptr; ///< Link to the Memory Manager
 public:
  /// @brief Default constructor
  MemoryData() = default;

  /// @brief Default destructor
  virtual ~MemoryData() = default;

  /// @brief Memory manager accessor
  /// @return Memory manager
  [[nodiscard]] AbstractMemoryManager <ManagedMemory> *memoryManager() const { return memoryManager_; }

  /// @brief Memory manager setter
  /// @param memoryManager Memory manager to set
  void memoryManager(AbstractMemoryManager <ManagedMemory> *memoryManager) { memoryManager_ = memoryManager; }

  /// @brief Return the data to the memory manager
  void returnToMemoryManager() { this->memoryManager_->recycleMemory(this->shared_from_this()); }

  /// @brief Mechanism to update the state of the data
  virtual void used() {};

  /// @brief Accessor to test if the data can be recycle and sent bask to the Pool, true by default
  /// @return True if the data can be sent to the Pool, else False
  virtual bool canBeRecycled() { return true; }

  /// @brief Mechanism to recycle data.
  /// @details If the ManagedMemory type uses user-defined allocations, then recycle is an appropriate place to
  /// deallocate the user-allocated data. It will be called only once before it is sent back to the pool.
  /// @attention If a StaticMemoryManager is used, then deallocation should be done within the destructor to match
  /// any allocations done within the constructor. (see StaticMemoryManager for more details)
  virtual void recycle() {};
};
}
#endif //HEDGEHOG_MEMORY_DATA_H
