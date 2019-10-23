//
// Created by anb22 on 6/21/19.
//

#ifndef HEDGEHOG_STATIC_MEMORY_MANAGER_H
#define HEDGEHOG_STATIC_MEMORY_MANAGER_H

#include "abstract_memory_manager.h"
/// @brief Hedgehog main namespace
namespace hh {
/// @brief Derived class from AbstractMemoryManager for statically allocated MemoryData, used for example for GPU
/// computation to avoid synchronisation during dynamic allocation.
/// @details The main difference between the AbstractMemoryManager and the StaticMemoryManager, is the allocation
/// process. Instead of filling the pool with default constructed data, the data will be constructed with a specific
/// constructor.
/// In order to select the constructor, the parameter's type(s) of the constructor that will be used as part of the
/// template list of the StaticMemoryManager.
/// For example, we can build a class A as:
/// @code
/// class A : public MemoryData<A>{
///     A(){/*[...]*/}; // Default constructor
///     A(int, float){/*[...]*/}; // Int, float constructor
///     A(int, float, double){/*[...]*/}; // Int, float, double constructor
///    ~A() { ... } // Free/delete memory allocated in constructors
/// };
/// @endcode
/// So the memory manager
/// @code
/// StaticMemoryManager<A> smm(10);
/// @endcode
/// will fill a pool with 10 A, calling the constructor:
/// @code
/// A(){/*[...]*/}; // Default constructor
/// @endcode
/// The memory manager
/// @code
/// StaticMemoryManager<A, int, float> smm(12, 42, 6.5);
/// @endcode
/// will fill a pool with 12 A, calling the constructor:
/// @code
/// A(int, float){/*[...]*/}; // Int, float constructor
/// @endcode
/// with 42 (int) and 6.5 (float) as values.
/// So the memory manager
/// @code
/// StaticMemoryManager<A, int, float, double> smm(0, 42, 6.5, 3.14159);
/// @endcode
/// will fill a pool with 1 A (a pool can not be empty), calling the constructor:
/// @code
/// A(int, float, double){/*[...]*/}; // Int, float, double constructor
/// @endcode
/// with 42 (int), 6.5 (float), 3.14159 (double) as values.
///
/// @attention Memory that is allocated from a user-defined MemoryData's constructor should be deallocated within its
/// destructor. The destructor will be called when all references to its std::shared_ptr have been lost. The
/// MemoryData::recycle should be used only if user-defined allocation occurs outside of the constructor.
///
/// Because the static memory manager is made to be used "as is", a copy method has been implemented using the copy
/// constructor. So if the StaticMemoryManager needs to be derived, use the copy constructor of the StaticMemoryManager
/// to transfer mandatory attributes to the copy.
///
/// @par Virtual Functions
/// AbstractMemoryManager::initializeMemoryManager
///
/// @tparam ManagedMemory Type to be managed by the memory manager
/// @tparam Args List of types corresponding to the constructor list of types
template<class ManagedMemory, class... Args>
class StaticMemoryManager : public AbstractMemoryManager<ManagedMemory> {
 private:

  /// @brief SFINAE construct to test if ManagedMemory has a constructor with Args... as parameter
  /// @details https://en.cppreference.com/w/cpp/language/sfinae
  class HasConstructor {
    /// @brief True test, testing if a constructor with the right parameters exist and can be called.
    /// @tparam TestManagedMemory Type of data to test
    /// @tparam TestArgs Type of constructor's parameters
    /// @return std::true_type
    template<class TestManagedMemory, class... TestArgs>
    static std::true_type test(
        decltype(new TestManagedMemory((std::declval<TestArgs>())...)));

    /// @brief False test, default test in case the searched constructor does not exist, without instantiating anything
    /// @tparam TestManagedMemory Type of data to test
    /// @tparam TestArgs Type of constructor's parameters
    /// @return std::false_type
    template<class TestManagedMemory, class ...TestArgs>
    static std::false_type test(...);
   public:
    /// @brief Enum to get statically the result of the test
    enum { value = std::is_same_v<decltype(test<ManagedMemory, Args...>(0)), std::true_type> };
  };

  static_assert(HasConstructor::value,
                "The Memory that you are trying to manage does not have the right constructor definition.");

  std::tuple<Args...> args_ = {}; ///< Values to pass to the constructor

 public:
  /// @brief Deleted default constructor
  StaticMemoryManager() = delete;

  /// @brief Constructor to use defining the pool capacity and the arguments to give the type constructor
  /// @param capacity Pool capacity
  /// @param args List of arguments to give the type constructor
  explicit StaticMemoryManager(size_t const &capacity, Args ... args) :
      AbstractMemoryManager<ManagedMemory>(capacity), args_(std::forward<Args>(args)...) {
  }

  /// @brief Copy constructor used by the copy method
  /// @param rhs StaticMemoryManager to copy
  StaticMemoryManager(StaticMemoryManager<ManagedMemory, Args...> &rhs)
      : AbstractMemoryManager<ManagedMemory>(rhs.capacity()), args_(rhs.args_) {}

  /// @brief Initialize method, calling the private initialize method with the pack arguments, and the user definable
  /// initializeMemoryManager
  void initialize() final {
    std::lock_guard<std::mutex> lk(this->memoryManagerMutex());
    if (!this->isInitialized()) {
      this->initialized();
      initialize(std::make_index_sequence<sizeof...(Args)>());
      this->initializeMemoryManager();
    }
  }

  /// @brief Copy method used for task duplication and execution pipeline
  /// @return The copy of the current (this) StaticMemoryManager
  std::shared_ptr<AbstractMemoryManager < ManagedMemory>> copy() override {
    return std::make_shared<StaticMemoryManager<ManagedMemory, Args...>>(*this);
  }

 private:
  /// @brief Private initialize method to call a specific constructor for the type
  /// @tparam Is Index sequence to iterate over the args
  template<size_t... Is>
  void initialize(std::index_sequence<Is...>) {
    std::for_each(
        this->pool()->begin(), this->pool()->end(),
        [this](std::shared_ptr<ManagedMemory> &emptyShared) {
          emptyShared = std::make_shared<ManagedMemory>(std::get<Is>(args_)...);
          emptyShared->memoryManager(this);
        }
    );
  }
};
}

#endif //HEDGEHOG_STATIC_MEMORY_MANAGER_H
