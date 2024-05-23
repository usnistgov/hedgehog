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

#ifndef HEDGEHOG_MANAGED_MEMORY_H
#define HEDGEHOG_MANAGED_MEMORY_H

#pragma once

#include <memory>

#include "manager/abstract_memory_manager.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Abstraction used to manage an user type with a memory manager
class ManagedMemory : public std::enable_shared_from_this<ManagedMemory> {
 private:
  AbstractMemoryManager *memoryManager_ = nullptr; ///< Link to the Memory Manager
 public:
/// @brief Default constructor
  ManagedMemory() = default;

/// @brief Default destructor
  virtual ~ManagedMemory() = default;

/// @brief Test is a memory manager has been connected to the managed memory
/// @return True if a memory manager has been connected, else False
  bool isMemoryManagerConnected() { return memoryManager_ != nullptr; }

/// @brief Memory manager accessor
/// @return Memory manager
  [[nodiscard]] AbstractMemoryManager *memoryManager() const { return memoryManager_; }

/// @brief Memory manager setter
/// @param memoryManager Memory manager to set
  void memoryManager(AbstractMemoryManager *memoryManager) { memoryManager_ = memoryManager; }

  /// @brief Return the data to the memory manager
  /// @throw std::runtime_error if the data is not linked to a memory manager
  void returnToMemoryManager() {
    if (memoryManager_) {
      memoryManager_->recycleMemory(this->shared_from_this());
    } else {
      throw (std::runtime_error("The data you are trying to return is not linked to a memory manager."));
    }
  }

  /// @brief Mechanism called by Hedgehog when the node returns the memory before it is tested for being recycled (call to canBeRecycled)
  virtual void postProcess() {};

  /// @brief Accessor to test if the data can be cleaned and sent back to the Pool, true by default
  /// @return True if the data can be sent to the Pool, else False
  virtual bool canBeRecycled() { return true; }

  /// @brief Mechanism to clean data.
  /// @details If the ManagedMemory type uses user-defined allocations, then clean is an appropriate place to
  /// deallocate the user-allocated data. It will be called only once before it is sent back to the pool.
  /// @attention If a StaticMemoryManager is used, then deallocation should be done within the destructor to match
  /// any allocations done within the constructor. (see StaticMemoryManager for more details)
  virtual void clean() {};

  /// @brief Mechanism to pre process the data.
  /// @details If the ManagedMemory type uses user-defined allocations such as unified memory, then preProcess is an
  /// appropriate place to apply synchronization on any asynchronous operations that were applied in the clean
  /// function. It will be called only once before it is returned from getManagedMemory.
  virtual void preProcess() {};

};
}

#endif //HEDGEHOG_MANAGED_MEMORY_H
