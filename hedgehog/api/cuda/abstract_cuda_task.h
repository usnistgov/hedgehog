//
// Created by hobbsubuntu on 6/9/19.
//

#ifndef HEDGEHOG_ABSTRACT_CUDA_TASK_H
#define HEDGEHOG_ABSTRACT_CUDA_TASK_H
#ifdef HH_USE_CUDA
#include <unordered_set>
#include <cuda_runtime.h>
#include "../task/abstract_task.h"


#ifndef checkCudaErrors
#define checkCudaErrors(err) if (!HLOG_ENABLED) {} \
        else __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err) {
        HLOG(0, "checkCudaErrors() API error = "
        << err
        << "\"" << cudaGetErrorString(err) << " \" from "
        << file << ":" << line);
        cudaDeviceReset();
        exit(42);
    }
}
#endif

template <class TaskOutput, class ... TaskInputs>
class AbstractCudaTask : public AbstractTask<TaskOutput, TaskInputs...> {
 private:
  int cudaDeviceId_;
  bool enablePeerAccess_;
  std::unordered_set<int> peerDeviceIds_;
  cudaStream_t stream_;

public:
    AbstractCudaTask(int cudaDeviceId, bool automaticStart = false, bool enablePeerAccess = true)
        : AbstractTask<TaskOutput, TaskInputs...>("CudaTask", 1, automaticStart),
                cudaDeviceId_(cudaDeviceId), enablePeerAccess_(enablePeerAccess) { }

    AbstractCudaTask(std::string_view const &name, int cudaDeviceId, bool automaticStart = false, bool enablePeerAccess = true)
            : AbstractTask<TaskOutput, TaskInputs...>(name, 1, automaticStart),
              cudaDeviceId_(cudaDeviceId), enablePeerAccess_(enablePeerAccess) { }

    void initialize() final {
        int numGpus;

        checkCudaErrors(cudaGetDeviceCount(&numGpus));

        assert(cudaDeviceId_ < numGpus);

        checkCudaErrors(cudaSetDevice(cudaDeviceId_));
        checkCudaErrors(cudaStreamCreate(&stream_));

        if (enablePeerAccess_)
        {
            for (int i = 0; i < numGpus; ++i) {
                if (i != this->cudaDeviceId_)
                {
                    int canAccess;
                    checkCudaErrors(cudaDeviceCanAccessPeer(&canAccess, this->cudaDeviceId_, i));
                    if (canAccess) {
                        cudaDeviceEnablePeerAccess(i, 0);
                        peerDeviceIds_.insert(i);
                    }
                }
            }
        }

        this->initializeCuda();
    }

    void shutdown() final {
        this->shutdownCuda();
        checkCudaErrors(cudaStreamDestroy(stream_));
    }

    virtual void initializeCuda() {}
    virtual void shutdownCuda() {}

    int cudaDeviceId() const {
        return cudaDeviceId_;
    }

    bool enablePeerAccess() const {
        return enablePeerAccess_;
    }

    cudaStream_t stream() const {
        return stream_;
    }

    bool hasPeerAccess(int peerDeviceId)
    {
        return peerDeviceIds_.find(peerDeviceId) != peerDeviceIds_.end();
    }


};

#endif //HH_USE_CUDA

#endif //HEDGEHOG_ABSTRACT_CUDA_TASK_H
