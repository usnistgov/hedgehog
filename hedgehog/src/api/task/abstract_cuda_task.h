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

#ifndef HEDGEHOG_ABSTRACT_CUDA_TASK_H
#define HEDGEHOG_ABSTRACT_CUDA_TASK_H
#ifdef HH_USE_CUDA

#include <cuda_runtime.h>
#include <unordered_set>

#include "abstract_task.h"
#include "../../tools/cuda_debugging.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Abstract Task specialized for CUDA computation.
/// @details At initialization, the device is set to the task (cudaSetDevice), and a stream is created and bound to the
/// task (cudaStreamCreate). During shutdown, the stream is destroyed (cudaStreamDestroy).
/// @par Virtual functions
/// Execute::execute (one for each of TaskInputs) <br>
/// AbstractCUDATask::copy (only used if number of threads is greater than 1 or used in an ExecutionPipeline) <br>
/// AbstractCUDATask::initializeCuda <br>
/// AbstractCUDATask::shutdownCuda <br>
/// Node::canTerminate <br>
/// Node::extraPrintingInformation <br>
/// @tparam Separator Separator position between input types and output types
/// @tparam AllTypes List of input and output types
template<size_t Separator, class ...AllTypes>
class AbstractCUDATask : public AbstractTask<Separator, AllTypes...> {
 private:
  bool enablePeerAccess_ = false;               ///< Enable CUDA Peer Access through all CUDA devices available
  std::unordered_set<int> peerDeviceIds_ = {};  ///< Sparse matrix of linked CUDA devices
  cudaStream_t stream_ = {};                    ///< CUDA stream linked to the task

 public:
  /// @brief AbstractCUDATask full constructor
  /// @param name Task name
  /// @param numberThreads Number of threads for the task
  /// @param enablePeerAccess Enable peer access for NVIDIA GPUs
  /// @param automaticStart Flag for automatic start (Cf. AbstractTask)
  AbstractCUDATask(std::string const &name, size_t numberThreads, bool enablePeerAccess, bool automaticStart = false)
      : AbstractTask<Separator, AllTypes...>(name, numberThreads, automaticStart),
        enablePeerAccess_(enablePeerAccess) {
    this->coreTask()->printOptions().background({0x76, 0xb9, 0x00, 0xff});
    this->coreTask()->printOptions().font({0xff, 0xff, 0xff, 0xff});
  }

  /// @brief Main constructor for a AbstractCUDATask
  /// @param name Name of the AbstractCUDATask
  /// @param numberThreads Number of thread for this task (default 1)
  explicit AbstractCUDATask(std::string const name = "CudaTask", size_t numberThreads = 1) :
      AbstractCUDATask<Separator, AllTypes...>(name, numberThreads, false, false) {};

  /// @brief Custom core task constructor
  /// @param coreTask Custom core to use
  /// @param enablePeerAccess Enable per access for NVIDIA GPUs
  AbstractCUDATask(std::shared_ptr<hh::core::CoreTask<Separator, AllTypes...>> coreTask, bool enablePeerAccess)
      : AbstractTask<Separator, AllTypes...>(std::shared_ptr<hh::core::CoreTask<Separator, AllTypes...>>(coreTask)),
        enablePeerAccess_(enablePeerAccess) {
    this->coreTask()->printOptions().background({0x76, 0xb9, 0x00, 0xff});
    this->coreTask()->printOptions().font({0xff, 0xff, 0xff, 0xff});
  }

  /// @brief Default destructor
  ~AbstractCUDATask() override {
    if (this->memoryManager() != nullptr) {
      checkCudaErrors(cudaSetDevice(this->memoryManager()->deviceId()));
    }
  }

  /// @brief Initialize an AbstractCUDATask to bind it to a CUDA device, and do the peer access if enabled.
  /// At the end will call AbstractCUDATask::initializeCuda.
  void initialize() final {
    int numGpus = 0;
    int canAccess = 0;
    checkCudaErrors(cudaGetDeviceCount(&numGpus));
    assert(this->deviceId() < numGpus);
    checkCudaErrors(cudaSetDevice(this->deviceId()));
    checkCudaErrors(cudaStreamCreate(&stream_));

    if (enablePeerAccess_) {
      for (int i = 0; i < numGpus; ++i) {
        if (i != this->deviceId()) {
          checkCudaErrors(cudaDeviceCanAccessPeer(&canAccess, this->deviceId(), i));

          if (canAccess) {
            auto ret = cudaDeviceEnablePeerAccess(i, 0);
            if (ret != cudaErrorPeerAccessAlreadyEnabled) {
              checkCudaErrors(ret);
            }
            peerDeviceIds_.insert(i);
          }
        }
      }
    }
    auto ret = cudaGetLastError();
    if (ret != cudaErrorPeerAccessAlreadyEnabled) {
      checkCudaErrors(ret);
    }
    this->initializeCuda();
  }

  /// @brief Shutdown an AbstractCUDATask to destroy the task's CUDA stream created during
  /// AbstractCUDATask::initialize.
  /// First calls AbstractCUDATask::shutdownCuda.
  void shutdown() final {
    this->shutdownCuda();
    checkCudaErrors(cudaStreamDestroy(stream_));
  }

  /// @brief Virtual initialization step, where user defined data structure can be initialized.
  virtual void initializeCuda() {}

  /// @brief Virtual shutdown step, where user defined data structure can be destroyed.
  virtual void shutdownCuda() {}

  /// @brief Accessor for peer access choice
  /// @return True if peer access is enabled, else False
  bool enablePeerAccess() const { return enablePeerAccess_; }

  /// @brief Getter for CUDA task's stream
  /// @return CUDA stream
  cudaStream_t stream() const { return stream_; }

  /// @brief Accessor for peer access enabled for a specific device id
  /// @param peerDeviceId Device id to test
  /// @return True if peer access enable for device id peerDeviceId, else False
  bool hasPeerAccess(int peerDeviceId) { return peerDeviceIds_.find(peerDeviceId) != peerDeviceIds_.end(); }

};

}

#endif //HH_USE_CUDA
#endif //HEDGEHOG_ABSTRACT_CUDA_TASK_H
