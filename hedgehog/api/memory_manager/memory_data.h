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


#ifndef HEDGEHOG_MEMORY_DATA_H
#define HEDGEHOG_MEMORY_DATA_H

#include <memory>
#include "memory_manager.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Memory data interface to use a data type in a Memory manager (MemoryManager or StaticMemoryManager)
/// @details To declare a data A  or B using the interface MemoryData, it can be written as:
/// @code
/// class A : public MemoryData<A>{};
/// //Or
/// template <class T>
/// class B : public MemoryData<B<T>>{};
/// @endcode
/// The data will be served from the pool to the task through the MemoryManager, and can be return with
/// MemoryData::returnToMemoryManager().
/// When returned, the MemoryData::used() and MemoryData::canBeRecycled() are called. If MemoryData::canBeRecycled()
/// returns true, MemoryData::recycle() is called and it is returned to the pool and made available by its
/// MemoryManager
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
  MemoryManager <ManagedMemory> *memoryManager_ = nullptr; ///< Link to the Memory Manager
 public:
  /// @brief Default constructor
  MemoryData() = default;

  /// @brief Default destructor
  virtual ~MemoryData() = default;

  /// @brief Test is a memory manager has been connected to the managed memory
  /// @return True if a memory manager has been connected, else False
  bool isMemoryManagerConnected(){ return memoryManager_ != nullptr; }

  /// @brief Memory manager accessor
  /// @return Memory manager
  [[nodiscard]] MemoryManager <ManagedMemory> *memoryManager() const { return memoryManager_; }

  /// @brief Memory manager setter
  /// @param memoryManager Memory manager to set
  void memoryManager(MemoryManager <ManagedMemory> *memoryManager) { memoryManager_ = memoryManager; }

  /// @brief Return the data to the memory manager
  void returnToMemoryManager() {
    if(memoryManager_){
      this->memoryManager_->recycleMemory(this->shared_from_this());
    }else{
      throw(
          std::runtime_error("The data you are trying to return to a memory manager is not linked to a memory manager."));
    }
  }

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

  /// @brief Mechanism to reuse the data.
  /// @details If the ManagedMemory type uses user-defined allocations such as unified memory, then reuse is an
  /// appropriate place to apply synchronization on any asynchronous operations that were applied in the recycle
  /// function. It will be called only once before it is returned from getManagedMemory.
  virtual void reuse() {};
};
}
#endif //HEDGEHOG_MEMORY_DATA_H
