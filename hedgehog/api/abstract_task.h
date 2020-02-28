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


#ifndef HEDGEHOG_TASK_H
#define HEDGEHOG_TASK_H

#include "../behavior/io/multi_receivers.h"
#include "../behavior/io/sender.h"
#include "../behavior/execute.h"
#include "memory_manager/abstract_memory_manager.h"
#include "../core/node/core_node.h"
#include "../core/defaults/core_default_task.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Base node for computation
/// @details Hedgehog Graph's node made for holding a computation kernel into their execute overload method from
/// Execute::execute.
///
/// An AbstractTask can be bound to multiple threads, forming a cluster of tasks. The base AbstractTask will be copied n
/// times and each of them will be bound to a thread. The AbstractTask::copy method must be overloaded to use this
/// functionality. Also, if the AbstractTask is part of Graph that will be duplicated with an AbstractExecutionPipeline,
/// AbstractTask::copy method needs to be overloaded.
///
/// An AbstractMemoryManager could be linked to a task through a graph (from tutorial 4, example shows Cuda memory usage):
/// @code
/// auto productTask = std::make_shared<CudaProductTask<MatrixType>>(p, numberThreadProduct);
/// auto cudaMemoryManagerA = std::make_shared<CudaMemoryManager<MatrixType, 'a'>>(nBlocks + 4, blockSize);
/// productTask->connectMemoryManager(cudaMemoryManagerProduct);
/// @endcode
///
/// Linking an AbstractMemoryManager requires that the AbstractTask's TaskOutput match the template type of the AbstractMemoryManager
///
/// The default order of execution is:
///     -# The cluster is created, for each task:
///     -# The thread is spawned,
///     -# The AbstractTask::initialize is called,
///     -# The memory manager, if it exists, is bound to the task (the device Id is shared, and is initialized),
///     -# Execute is called, while AbstractTask::canTerminate is True:
///         - If the task is set to start automatically Execute::execute is called with nullptr,
///         - If not, the task will wait for a data to come and Execute::execute is called with the received data,
///     -# AbstractTask::shutdown is called, and is signal to the linked nodes.
///
/// Only Execute::execute method need to be overloaded for every AbstractTask input types.
///
/// \par Virtual functions
/// Execute::execute (one for each of TaskInputs) <br>
/// AbstractTask::initialize <br>
/// AbstractTask::shutdown <br>
/// AbstractTask::copy (mandatory if usage of clusters or AbstractExecutionPipeline) <br>
/// Node::canTerminate (mandatory if cycle in the graph) <br>
/// Node::extraPrintingInformation
///
/// @attention In case of cycle in the graph AbstractTask::canTerminate needs to be overloaded or the graph will
/// deadlock. By default, AbstractTask::canTerminate will be true if there is no "input node" connected AND no data
/// available in the task input queue (CF tutorial 3).
/// @tparam TaskOutput Output task data type.
/// @tparam TaskInputs Inputs task data type.
template<class TaskOutput, class ...TaskInputs>
class AbstractTask :
    public behavior::MultiReceivers<TaskInputs...>,
    public behavior::Sender<TaskOutput>,
    public virtual behavior::Node,
    public behavior::Execute<TaskInputs> ... {

  static_assert(traits::isUnique<TaskInputs...>, "A Task can't accept multiple inputs with the same type.");
  static_assert(sizeof... (TaskInputs) >= 1, "A node need to have one output type and at least one output type.");

  std::shared_ptr<core::CoreTask<TaskOutput, TaskInputs...>> taskCore_ = nullptr; ///< Task Core
  std::shared_ptr<AbstractMemoryManager<TaskOutput>> mm_ = nullptr; ///< Task's memory manager

 public:
  /// @brief AbstractTask default constructor
  /// @details Create a task with the name "Task", one task in the cluster (no copy overload needed), and no automatic
  /// start.
  AbstractTask(){
    taskCore_ = std::make_shared<core::CoreDefaultTask<TaskOutput, TaskInputs...>>("Task",
                                                                                   1,
                                                                                   core::NodeType::Task,
                                                                                   this,
                                                                                   false);
  }

  /// @brief Create a task with a custom name
  /// @details Create a task with one task in the cluster (no copy overload needed), and no automatic start.
  /// @param name Task name
  explicit AbstractTask(std::string_view const &name) {
    taskCore_ =
        std::make_shared<core::CoreDefaultTask<TaskOutput, TaskInputs...>>(name, 1, core::NodeType::Task, this, false);
  }

  /// @brief Create a task with a custom name and the task's cluster size
  /// @param name Task name
  /// @param numberThreads Task's cluster size
  explicit AbstractTask(std::string_view const &name, size_t numberThreads) {
    taskCore_ =
        std::make_shared<core::CoreDefaultTask<TaskOutput, TaskInputs...>>(name,
                                                                           numberThreads,
                                                                           core::NodeType::Task,
                                                                           this,
                                                                           false);
  }

  /// @brief Create a task with a custom name, the task's cluster size, and the automatic start.
  /// @param name Task name
  /// @param numberThreads Task's cluster size
  /// @param automaticStart Task's automatic start
  explicit AbstractTask(std::string_view const &name, size_t numberThreads, bool automaticStart) {
    taskCore_ =
        std::make_shared<core::CoreDefaultTask<TaskOutput, TaskInputs...>>
            (name, numberThreads, core::NodeType::Task, this, automaticStart);
  }

  /// @brief Internal constructor needed to create nodes that derive from AbstractTask.
  /// @attention Do not use
  /// @param name Task name
  /// @param numberThreads Task's cluster size
  /// @param nodeType Task node type
  /// @param automaticStart Task's automatic start
  AbstractTask(std::string_view const name, size_t numberThreads, core::NodeType nodeType, bool automaticStart) {
    taskCore_ = std::make_shared<core::CoreDefaultTask<TaskOutput, TaskInputs...>>
        (name, numberThreads, nodeType, this, automaticStart);
  }

  /// @brief Copy constructor
  /// @param rhs Task to copy
  explicit AbstractTask(AbstractTask<TaskOutput, TaskInputs ...> *rhs) {
    taskCore_ = std::make_shared<core::CoreDefaultTask<TaskOutput, TaskInputs...>>
        (rhs->name(),
         rhs->numberThreads(),
         rhs->nodeType(),
         this,
         rhs->automaticStart());
  }

  /// @brief Default destructor
  virtual ~AbstractTask() = default;

  /// @brief Add an output data
  /// @param output Data to send to the output nodes
  void addResult(std::shared_ptr<TaskOutput> output) { this->taskCore_->sendAndNotify(output); }

  /// @brief Task's name accessor
  /// @return Task name
  std::string_view name() { return this->taskCore_->name(); }

  /// @brief Task's number of threads accessor
  /// @return Task number of threads
  size_t numberThreads() { return this->taskCore_->numberThreads(); }

  /// @brief Task's automatic start accessor
  /// @return Task automatic start property
  bool automaticStart() { return this->taskCore_->automaticStart(); }

  /// @brief Task's node type accessor
  /// @return Task's node type property
  core::NodeType nodeType() { return this->taskCore_->type(); }

  /// @brief Task's device ID accessor
  /// @return Task's device ID
  int deviceId() { return this->taskCore_->deviceId(); }

  /// @brief Task's graph ID accessor
  /// @return Task's graph ID
  int graphId() { return this->taskCore_->graphId(); }

  /// @brief Task's core accessor
  /// @return Task's core
  std::shared_ptr<core::CoreNode> core() final { return taskCore_; }

  /// @brief Task's memory manager accessor
  /// @return Task's memory manager
  std::shared_ptr<AbstractMemoryManager<TaskOutput>> const & memoryManager() const { return mm_; }

  /// @brief Default copy overload, fail if cluster need to be copied
  /// @return A copy of the current AbstractTask
  virtual std::shared_ptr<AbstractTask<TaskOutput, TaskInputs...>> copy() { return nullptr; }

  /// @brief Initialize method called before AbstractTask::Execute loop
  virtual void initialize() {}

  /// @brief Shutdown method called after AbstractTask::Execute loop, when AbstractTask::canTerminate evaluates to true
  virtual void shutdown() {}

  /// @brief Connect a memory manager to the task
  /// @param mm Memory manager to connect
  void connectMemoryManager(std::shared_ptr<AbstractMemoryManager<TaskOutput>> mm) {
    static_assert(
        traits::is_managed_memory_v<TaskOutput>,
        "The type given to the memory manager should inherit \"MemoryData\", and be default constructible!");
    mm_ = mm;
  }

  /// @brief Memory manager accessor
  /// @return Connected memory manager
  std::shared_ptr<TaskOutput> getManagedMemory() {
    if (mm_ == nullptr) {
      std::cerr
          << "For the task:\"" << this->name()
          << "\"To get managed memory, you need first to connect a memory manager to the task via "
             "\"connectMemoryManager()\""
          << std::endl;
      exit(42);
    }

    auto start = std::chrono::high_resolution_clock::now();
    this->taskCore_->nvtxProfiler()->startRangeWaitingForMemory();
    auto data = mm_->getManagedMemory();
    this->taskCore_->nvtxProfiler()->endRangeWaitingForMem();
    auto finish = std::chrono::high_resolution_clock::now();

    this->core()->incrementWaitForMemoryDuration(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));

    return data;
  }

  /// @brief Method called to test the task;s termination, by default, the test is: no input nodes connected and, no
  /// data waiting to be treated
  /// @return True if the task can be terminated, else False
  bool canTerminate() override { return !this->taskCore_->hasNotifierConnected() && this->taskCore_->receiversEmpty(); }
};
}
#endif //HEDGEHOG_TASK_H
