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


#ifndef HEDGEHOG_MEMORY_MANAGER_H
#define HEDGEHOG_MEMORY_MANAGER_H

#include <memory>
#include <mutex>
#include <condition_variable>

#include "../../tools/traits.h"
#include "../../tools/logger.h"
#include "../../tools/nvtx_profiler.h"
#include "../../behavior/memory_manager/pool.h"

/// @brief Hedgehog main namespace
namespace hh {
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @brief Forward Declaration of Memory Data
/// @tparam ManagedMemory Managed memory data type
template<class ManagedMemory>
class MemoryData;
#endif //DOXYGEN_SHOULD_SKIP_THIS

/// @brief Abstract interface for Hedgehog's Memory manager
/// @details The memory managers for Hedgehog have two main goals:
/// -# Throttle the amount of memory that lives inside the graph,
/// -# Provide a mechanism for data reuse (with recycling to clean to-be-reused pieces of memory).
///
/// @attention Data that is managed by an MemoryManager, requires:
/// - Default constructible: can be constructed without arguments or initialization values, either cv-qualified or
/// not. This includes scalar types, default constructible classes and arrays of such types
/// (ref: http://www.cplusplus.com/reference/type_traits/is_default_constructible/),
/// - Derive from MemoryData.
///
/// The amount of data available is set at construction: MemoryManager(size_t const &capacity). At construction,
/// these data are default constructed and stored into a Pool.
/// When constructed, a memory manager can be attached to a task with AbstractTask::connectMemoryManager().
///
/// @attention The Type handled by the memory manager must be the same type as the AbstractTask output type.
///
/// Data can be retrieved from the memory manager from within the AbstractTask::execute method with
/// MemoryManager::getManagedMemory(). If no data is available (i.e. the Pool is empty), the task will
/// block on the call and wait for data to be recycled and available. To return data to the memory manager from
/// another task, or, outside the graph, the method MemoryData::returnToMemoryManager() should be called.
///
/// @attention When MemoryData::returnToMemoryManager() is invoked, the recycling mechanism is used which consists of
/// calling the following virtual methods, in the following order:
/// -# MemoryData::used(): Method used to update the "state" of the MemoryData (for example in the case the MemoryData
/// is returned to the MemoryManager multiple times before being recycled and sent back to the Pool.
/// -# MemoryData::canBeRecycled(): Boolean method to determine if the MemoryData can be recycled, and send back to the
/// Pool.
/// -# MemoryData::recycle(): Recycle the MemoryData. The data given by the MemoryManager is default constructed
/// the first time. If specialised class attributes are allocated, they should be deallocated in this method, to avoid
/// memory leaks, to return the ManagedMemory to the same state as default construction.
///
/// The only pure virtual method is the copy method to duplicate a derived MemoryManager to different
/// graphs in an ExecutionPipeline.
///
/// If an initialize step is needed to be added, MemoryManager::initializeMemoryManager() can be overload.
///
/// @code
/// // Derived Memory data
/// template<class T>
/// class DynamicMemoryManageData : public MemoryData<DynamicMemoryManageData<T>> {
///  private:
///   T *data_ = nullptr;
///   size_t ttl_ = 1; // Time to live for data reuse
///
///  public:
///   DynamicMemoryManageData() = default;
///   virtual ~DynamicMemoryManageData() = default;
///   void ttl(size_t ttl) { ttl_ = ttl; }
///   void data(T *data) { data_ = data; }
///   // Delete data allocated after default cstr by IntToDynamicMemoryManagedInt
///   void recycle() override { delete[] data_; }
///   void used() override { ttl_--; }
///   bool canBeRecycled() override { return ttl_ == 0; }
/// };
///
/// // Derived MemoryManager
/// template<class T>
/// class DynamicMemoryManager : public MemoryManager<DynamicMemoryManageData<T>> {
///  public:
///   explicit DynamicMemoryManager(size_t const &capacity)
///     : MemoryManager<DynamicMemoryManageData<T>>(capacity) {}
///
///   std::shared_ptr<MemoryManager<DynamicMemoryManageData<T>>> copy() override {
///     return std::make_shared<DynamicMemoryManager<T>>(this->capacity());
///   }
/// };
///
/// // Task returning DynamicMemoryManagedData and getting data from a memory manager
/// class IntToDynamicMemoryManagedInt : public AbstractTask<DynamicMemoryManageData<int>, int> {
///  public:
///   explicit IntToDynamicMemoryManagedInt(size_t numberThreads) : AbstractTask("Dynamic Task", numberThreads) {}
///   virtual ~IntToDynamicMemoryManagedInt() = default;
///
///   void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
///     auto mem = this->getManagedMemory(); // Get the memory from the memory manager
///     mem->data(new int[30]()); // Allocate more memory
///     mem->ttl(1); // Sets the time to live (number of times reused)
///     addResult(mem);
///   }
///
///   std::shared_ptr<AbstractTask<DynamicMemoryManageData<int>, int>> copy() override {
///     return std::make_shared<IntToDynamicMemoryManagedInt>(this->numberThreads());
///   }
/// };
///
/// // Task accepting DynamicMemoryManagedData and return the data to the memory manager
/// class DynamicMemoryManagedIntToVoid :
///     public AbstractTask<void, DynamicMemoryManageData<int>> {
///  public:
///   DynamicMemoryManagedIntToVoid() : AbstractTask("Output") {}
///   virtual ~DynamicMemoryManagedIntToStaticMemoryManagedInt() = default;
///
///   void execute(std::shared_ptr<DynamicMemoryManageData<int>> ptr) override {
///     ptr->returnToMemoryManager(); // Return the data to the memory manager
///   }
///
///   std::shared_ptr<
///       AbstractTask<void, DynamicMemoryManageData<int>>> copy() override {
///     return std::make_shared<DynamicMemoryManagedIntToVoid>();
///   }
///
/// };
///
/// // In main
/// auto dynamicTask = std::make_shared<IntToDynamicMemoryManagedInt>(2); // Create the task using the memory manager
/// // Create the task returning the data to the memory manager
/// auto outTask = std::make_shared<DynamicMemoryManagedIntToVoid>();
/// auto dynamicMM = std::make_shared<DynamicMemoryManager<int>>(2); // Create the memory manager
/// dynamicTask->connectMemoryManager(dynamicMM); // Associate the memory manager to the task
/// @endcode
///
/// @par Virtual Functions
/// MemoryManager::copy <br>
/// MemoryManager::initialize (advanced, should be used carefully) <br>
/// MemoryManager::initializeMemoryManager (user-defined customization)
///
/// @tparam ManagedMemory Type of data that will be managed by an MemoryManager
template<class ManagedMemory, class = void>
class MemoryManager {
 private:
  int deviceId_ = 0; ///< Device Id of linked task
  bool initialized_ = false; ///< Flag to determine if MemoryManager has been initialized
  std::unique_ptr<behavior::Pool<ManagedMemory>> pool_ = {}; ///< Inside pool to store the data

  std::shared_ptr<NvtxProfiler> profiler_ = nullptr; ///< NVTX profiler instance to follow memory manager state

 protected:
  std::mutex memoryManagerMutex_ = {}; ///< Mutex for user interface

 public:
  /// @brief Deleted Default constructor
  MemoryManager() = delete;

  /// @brief Only used constructor
  /// @param capacity Memory Manager capacity, number of elements available, set to 1 if 0
  explicit MemoryManager(size_t const &capacity)
      : deviceId_(0) {
    auto tempCapacity = capacity > 0 ? capacity : 1; // Cannot have an empty memory manager
    pool_ = std::make_unique<behavior::Pool<ManagedMemory>>(tempCapacity);
  }

  /// @brief Default destructor
  virtual ~MemoryManager() = default;

  /// @brief Virtual copy method used for task duplication and execution pipeline
  /// @return Return a copy of this specialised MemoryManager
  virtual std::shared_ptr<MemoryManager<ManagedMemory>> copy()  {
    return std::make_shared<MemoryManager<ManagedMemory>>(this->capacity());
  }

  /// @brief Device id setter
  /// @param deviceId Task's device id to set
  void deviceId(int deviceId) { deviceId_ = deviceId; }

  /// @brief NVTX profiler setter
  /// @param profiler NVTX profiler to set
  void profiler(const std::shared_ptr<NvtxProfiler> &profiler) { this->profiler_ = profiler; }

  /// @brief Return the current size of the inside pool
  /// @details Lock the api user mutex before getting the current size of the inside pool
  /// @return The current number of available data
  [[nodiscard]] size_t currentSize() {
    std::lock_guard<std::mutex> lk(memoryManagerMutex_);
    return this->pool()->size();
  }

  /// @brief Get an available managed memory, block if none are available
  /// @return An available managed memory
  std::shared_ptr<ManagedMemory> getManagedMemory() {
    std::shared_ptr<ManagedMemory> managedMemory;
    HLOG(4, "StaticMemoryManager memory pool size = " << this->currentSize())
    managedMemory = this->pool()->pop_front();
    HLOG(4,
         "StaticMemoryManager After waiting: received: " << managedMemory << " pSize: " << (int) (this->pool()->size()))
    return managedMemory;
  };

  /// @brief Recycle memory
  /// @details Lock the user api mutex before call used(), canBeRecycled() and if true, recycle() and push it back into
  /// the pool
  /// @param managedMemory Data to recycle
  void recycleMemory(std::shared_ptr<MemoryData<ManagedMemory>> managedMemory) {
    std::lock_guard<std::mutex> lk(memoryManagerMutex_);
    managedMemory->used();
    if (managedMemory->canBeRecycled()) {
      this->profiler_->addReleaseMarker();
      managedMemory->recycle();
      this->pool_->push_back(std::dynamic_pointer_cast<ManagedMemory>(managedMemory));
    }
  };

  /// @brief Initialize the memory manager
  /// @details Lock the user api mutex, fill the pool with default constructed data, and call initializeMemoryManager()
  virtual void initialize() {
    std::lock_guard<std::mutex> lk(memoryManagerMutex_);
    if (!this->isInitialized()) {
      this->initialized();
      std::for_each(
          this->pool()->begin(), this->pool()->end(),
          [this](std::shared_ptr<ManagedMemory> &emptyShared) {
            emptyShared = std::make_shared<ManagedMemory>();
            emptyShared->memoryManager(this);
          }
      );
      initializeMemoryManager();
    }
  };

  /// @brief Device Id accessor
  /// @return Device id
  [[nodiscard]] int deviceId() const { return deviceId_; }

  /// @brief User-definable initialization step for a memory manager
  virtual void initializeMemoryManager() {}
 protected:
  /// @brief Inside pool accessor
  /// @return Inside pool
  [[nodiscard]] std::unique_ptr<behavior::Pool<ManagedMemory>> const &pool() const { return pool_; }

  /// @brief Capacity accessor
  /// @return Pool's capacity
  [[nodiscard]] size_t capacity() const { return this->pool()->capacity(); };

  /// @brief Initialized flag accessor
  /// @return True is the memory manager has been initialized, else False
  [[nodiscard]] bool isInitialized() const { return initialized_; }

  /// @brief User api mutex accessor
  /// @return User api mutex
  [[nodiscard]] std::mutex &memoryManagerMutex() { return memoryManagerMutex_; }

  /// @brief Flag the memory manager has initialized
  void initialized() { initialized_ = true; }

};

/// @brief Definition of MemoryManager, for data that do not derive from MemoryData and/ or are not default
/// constructible
/// @details The class defines all methods used by Hedgehog internally and will only throw errors. Should not be used,
/// just here to define the methods.
/// @attention The existence of a memory manager into a Task should be checked before usage.
/// @tparam ManagedMemory Type of data that do not derive from MemoryData and/ or are not default
/// constructible
template<class ManagedMemory>
class MemoryManager<ManagedMemory,
                    typename std::enable_if_t<!traits::is_managed_memory_v<ManagedMemory>>> {
 public:
  /// @brief Initializer
  /// @exception std::runtime_error A Memory Manager without MemoryData
  virtual void initialize() {
    std::ostringstream oss;
    oss << "Call Memory manager method: " << __FUNCTION__ << " without managed memory data.";
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Device id setter
  /// @exception std::runtime_error A Memory Manager without MemoryData
  void deviceId(int) {
    std::ostringstream oss;
    oss << "Call Memory manager method: " << __FUNCTION__ << " without managed memory data.";
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Device id getter
  /// @return The device Id
  /// @exception std::runtime_error A Memory Manager without MemoryData
  int deviceId() const {
    std::ostringstream oss;
    oss << "Call Memory manager method: " << __FUNCTION__ << " without managed memory data.";
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Profiler setter
  /// @exception std::runtime_error A Memory Manager without MemoryData
  void profiler(const std::shared_ptr<NvtxProfiler> &) {
    std::ostringstream oss;
    oss << "Call Memory manager method: " << __FUNCTION__ << " without managed memory data.";
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  }

  /// @brief Copy definition
  /// @exception std::runtime_error A Memory Manager without MemoryData
  /// @return Memory manager copy
  virtual std::shared_ptr<MemoryManager<ManagedMemory>> copy() {
    std::ostringstream oss;
    oss << "Call Memory manager method: " << __FUNCTION__ << " without managed memory data.";
    HLOG_SELF(0, oss.str())
    throw (std::runtime_error(oss.str()));
  };
};


}

#endif //HEDGEHOG_MEMORY_MANAGER_H
