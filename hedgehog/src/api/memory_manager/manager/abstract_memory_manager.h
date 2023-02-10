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

#ifndef HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H_
#define HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H_

#pragma once

#include <memory>
#include <mutex>

#include "../pool.h"
#include "../../../tools/nvtx_profiler.h"

/// @brief Hedgehog main namespace
namespace hh {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/// Forward declaration of Managed Memory
class ManagedMemory;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// Abstract Memory manager
/// @brief Present a thread safe pool of Managed memory
class AbstractMemoryManager {
 private:
  int deviceId_ = 0; ///< Device Id of linked task
  bool initialized_ = false; ///< Flag to determine if AbstractMemoryManager has been initialized
  std::unique_ptr<tool::Pool<ManagedMemory>> pool_ = {}; ///< Inside pool to store the data
  std::shared_ptr<NvtxProfiler> profiler_ = nullptr; ///< NVTX profiler instance to follow memory manager state
 protected:
  std::mutex memoryManagerMutex_ = {}; ///< Mutex for user interface

 public:
  /// @brief Only used constructor
  /// @param capacity Memory Manager capacity, number of elements available, set to 1 if 0
  explicit AbstractMemoryManager(size_t const &capacity) :
      deviceId_(0),
      pool_(std::make_unique<tool::Pool<ManagedMemory>>(capacity > 0 ? capacity : 1)) {}

  /// @brief Default destructor
  virtual ~AbstractMemoryManager() = default;

  /// @brief Device Id accessor
  /// @return Device id
  [[nodiscard]] int deviceId() const { return deviceId_; }

  /// @brief Return the current size of the inside pool
  /// @details Lock the api user mutex before getting the current size of the inside pool
  /// @return The current number of available data
  [[nodiscard]] size_t currentSize() {
    memoryManagerMutex_.lock();
    auto s = this->pool()->size();
    memoryManagerMutex_.unlock();
    return s;
  }

  /// @brief Capacity accessor
  /// @return Pool's capacity
  [[nodiscard]] size_t capacity() const {
    return this->pool()->capacity();
  };

  /// @brief Device id setter
  /// @param deviceId Task's device id to set
  void deviceId(int deviceId) { deviceId_ = deviceId; }

  /// @brief NVTX profiler setter
  /// @param profiler NVTX profiler to set
  void profiler(const std::shared_ptr<NvtxProfiler> &profiler) { this->profiler_ = profiler; }

  /// @brief Get an available managed memory, block if none are available
  /// @return An available managed memory
  virtual std::shared_ptr<ManagedMemory> getManagedMemory() = 0;

  /// @brief Recycle memory
  /// @details Lock the user api mutex before call used(), canBeRecycled() and if true, clean() and push it back into
  /// the pool
  /// @param managedMemory Data to clean
  virtual void recycleMemory(std::shared_ptr<ManagedMemory> const &managedMemory) = 0;

  /// @brief Virtual copy method used for task duplication and execution pipeline
  /// @return Return a copy of this specialised AbstractMemoryManager
  virtual std::shared_ptr<AbstractMemoryManager> copy() = 0;

  /// @brief Initialize the memory manager
  /// @details Lock the user api mutex, fill the pool with default constructed data, and call initializeMemoryManager()
  virtual void initialize() = 0;

  /// @brief Return the real managed type under the form of a string
  /// @return Real managed type under the form of a string
  [[nodiscard]] virtual std::string managedType() const = 0;

 protected:
  /// @brief Accessor to NVTX profiler
  /// @return Attached NVTX profiler
  [[nodiscard]] std::shared_ptr<NvtxProfiler> const &profiler() const {
    return profiler_;
  }

  /// @brief Inside pool accessor
  /// @return Inside pool
  [[nodiscard]] std::unique_ptr<tool::Pool<ManagedMemory>> const &pool() const { return pool_; }

  /// @brief Initialized flag accessor
  /// @return True is the memory manager has been initialized, else False
  [[nodiscard]] bool isInitialized() const { return initialized_; }

  /// @brief User api mutex accessor
  /// @return User api mutex
  [[nodiscard]] std::mutex &memoryManagerMutex() { return memoryManagerMutex_; }

  /// @brief Flag the memory manager has initialized
  void initialized() { initialized_ = true; }
};
}

#endif //HEDGEHOG_ABSTRACT_MEMORY_MANAGER_H_
