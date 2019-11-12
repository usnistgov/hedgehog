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


#ifndef HEDGEHOG_NVTX_PROFILER_H
#define HEDGEHOG_NVTX_PROFILER_H

#ifdef HH_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include <string_view>
#include <cstring>
#include <iostream>

/// @brief Hedgehog main namespace
namespace hh {
#define NVTX_COLOR_INITIALIZING    0xFF123456
#define NVTX_COLOR_EXECUTING       0xFF72ff68
#define NVTX_COLOR_WAITING         0xFFff7f83
#define NVTX_COLOR_WAITING_FOR_MEM 0xFFffc86a
#define NVTX_COLOR_RELEASE_MEM     0xFF7fbdff
#define NVTX_COLOR_SHUTTING_DOWN   0xFF654321

/// @brief A class to wrap calls to the NVTX library for tracking events that occur within an Hedgehog task graph
/// @details
/// Hedgehog uses the NVTX API and NVIDIA Nsight Systems to visualize the execution of a graph of tasks.
///
/// The current profiling mode is to have one NVTX domain per task.
///
/// In this mode, a task uses the single domain to visualize when the task is initializing, executing, waiting for memory,
/// waiting for data, releasing memory, and shutting down. This has the effect of visualizing all tasks and their
/// threads to identify precisely what that task is doing at any moment in time. This is useful for
/// visualizing the interaction between tasks and identify bottlenecks.
///
/// Depending on the version of the Nsight Systems tool, there may be limitations to the number of NVTX domains. If
/// your graph is extremely large, it is recommended to get the latest version of the NVIDIA Nsight Systems tool.
///
/// @note To enable NVTX profiling you must add the USE_NVTX directive.
/// @note Add 'FindNVTX.cmake' to your project to assist in finding the necessary includes and libraries for use with
/// NVTX
class NvtxProfiler {
 private:
#ifdef HH_USE_NVTX
  std::string_view initializeName_{}; ///< Name for the initialization attribute
  std::string_view executeName_{}; ///< Name for the execute attribute
  std::string_view waitName_{}; ///< Name for the wait attribute
  std::string_view waitForMemName_{}; ///< Name for the wait for memory attribute
  std::string_view releaseMemName_{}; ///< Name for the release memory attribute
  std::string_view shutdownName_{}; ///< Name for the shutdown attribute

  nvtxDomainHandle_t taskDomain_; ///< The domain for the task

  nvtxStringHandle_t initializeString_{}; ///< Cache'd string used within the initialize attribute
  nvtxStringHandle_t executeString_{}; ///< Cache'd string used within the execute attribute
  nvtxStringHandle_t waitString_{}; ///< Cache'd string used within the wait attribute
  nvtxStringHandle_t waitForMemString_{}; ///< Cache'd string used within the wait for memory attribute
  nvtxStringHandle_t releaseMemString_{}; ///< Cache'd string used within the release memory attribute
  nvtxStringHandle_t shutdownString_{}; ///< Cache'd string used within the shutdown attribute

  nvtxEventAttributes_t *initializeAttrib_; ///< The initialize attribute
  nvtxEventAttributes_t *executeAttrib_; ///< The execute attribute
  nvtxEventAttributes_t *waitAttrib_; ///< The wait attribute
  nvtxEventAttributes_t *waitForMemAttrib_; ///< The wait for memory attribute
  nvtxEventAttributes_t *releaseMemAttrib_; ///< The release memory attribute
  nvtxEventAttributes_t *shutdownAttrib_; ///< The shutdown attribute

  nvtxRangeId_t initializeRangeId_ = 0; ///< Range identifier for initialize
  nvtxRangeId_t executeRangeId_ = 0; ///< Range identifier for execute
  nvtxRangeId_t waitRangeId_ = 0; ///< Range identifier for wait (for data)
  nvtxRangeId_t waitForMemRangeId_ = 0; ///< Range identifier for wait for memory
  nvtxRangeId_t shutdownRangeId_ = 0; ///< Range identifier for shutdown


  /// @brief Creates an event attribute with a specified color
  /// @param color the color shown in the timeline view
  nvtxEventAttributes_t *createEventAttribute(uint32_t color) {
    nvtxEventAttributes_t *event = new nvtxEventAttributes_t;
    bzero(event, NVTX_EVENT_ATTRIB_STRUCT_SIZE);
    event->version = NVTX_VERSION;
    event->size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event->colorType = NVTX_COLOR_ARGB;
    event->color = color;
    return event;
  }
#endif

 public:

  /// @brief Deleted default constructor
  NvtxProfiler() = delete;


#ifdef HH_USE_NVTX
  /// @brief Constructs the NvtxProfiler with the name of the task
  /// @details Each NvtxProfiler will hold profiling information for each task. It will profile all stages of the task's
  /// life cycle: initialize, execution, waiting for data, waiting for memory, releasing memory, and shutting down.
  /// @param taskName the name of the task
  explicit NvtxProfiler(std::string_view const & taskName) {
    taskDomain_ = nvtxDomainCreateA(taskName.data());

    initializeAttrib_ = createEventAttribute(NVTX_COLOR_INITIALIZING);
    executeAttrib_ = createEventAttribute(NVTX_COLOR_EXECUTING);

    waitAttrib_ = createEventAttribute(NVTX_COLOR_WAITING);
    waitAttrib_->payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;

    waitAttrib_->payload.ullValue = 0;

    waitForMemAttrib_ = createEventAttribute(NVTX_COLOR_WAITING_FOR_MEM);
    releaseMemAttrib_ = createEventAttribute(NVTX_COLOR_RELEASE_MEM);
    shutdownAttrib_ = createEventAttribute(NVTX_COLOR_SHUTTING_DOWN);
  }
#else //HH_USE_NVTX
  /// @brief Constructs the NvtxProfiler with the name of the task
  /// @details Each NvtxProfiler will hold profiling information for each task. It will profile all stages of the task's
  /// life cycle: initialize, execution, waiting for data, waiting for memory, releasing memory, and shutting down.
  explicit NvtxProfiler(std::string_view const &) {}
#endif //HH_USE_NVTX

#ifdef HH_USE_NVTX
  /// @brief Destructor, deletes all attributes allocated
  ~NvtxProfiler() {
    delete initializeAttrib_;
    delete executeAttrib_;
    delete waitAttrib_;
    delete waitForMemAttrib_;
    delete releaseMemAttrib_;
    delete shutdownAttrib_;
    nvtxDomainDestroy(taskDomain_);
  }
#else //HH_USE_NVTX
  /// @brief Destructor, deletes all attributes allocated
  ~NvtxProfiler() = default;
#endif //HH_USE_NVTX

  /// Initializes the NvtxProfiler, and adds the threadId that is associated with the task.
  /// @details Initialization of the NvtxProfiler creates and cache's the string names of the various event attributes
  /// @param threadId the thread identifier
  void initialize([[maybe_unused]]int threadId) {
#ifdef HH_USE_NVTX
    std::string prefixName(std::to_string(threadId));
    initializeName_ = prefixName + ":Initializing";
    executeName_ = prefixName + ":Executing";
    waitName_ = prefixName + ":Waiting";
    waitForMemName_ = prefixName + ":MemWait";
    releaseMemName_ = prefixName + ":Release";
    shutdownName_ = prefixName + ":Shutdown";

    initializeString_ = nvtxDomainRegisterStringA(taskDomain_, initializeName_.data());
    executeString_ = nvtxDomainRegisterStringA(taskDomain_, executeName_.data());
    waitString_ = nvtxDomainRegisterStringA(taskDomain_, waitName_.data());
    waitForMemString_ = nvtxDomainRegisterStringA(taskDomain_, waitForMemName_.data());
    releaseMemString_ = nvtxDomainRegisterStringA(taskDomain_, releaseMemName_.data());
    shutdownString_ = nvtxDomainRegisterStringA(taskDomain_, shutdownName_.data());

    initializeAttrib_->messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    initializeAttrib_->message.registered = initializeString_;

    executeAttrib_->messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    executeAttrib_->message.registered = executeString_;

    waitAttrib_->messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    waitAttrib_->message.registered = waitString_;

    waitForMemAttrib_->messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    waitForMemAttrib_->message.registered = waitForMemString_;

    releaseMemAttrib_->messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    releaseMemAttrib_->message.registered = releaseMemString_;

    shutdownAttrib_->messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    shutdownAttrib_->message.registered = shutdownString_;
#endif
  }

  /// @brief Adds a release marker into the timeline to show when the task released memory.
  void addReleaseMarker() {
#ifdef HH_USE_NVTX
    nvtxDomainMarkEx(taskDomain_, releaseMemAttrib_);
#endif
  }

   /// @brief Starts tracking intialization in the timeline to show when the task has started its initialization phase.
  void startRangeInitializing() {
#ifdef HH_USE_NVTX
    initializeRangeId_ = nvtxDomainRangeStartEx(taskDomain_, initializeAttrib_);
#endif
  }

  /// @brief Starts tracking execution in the timeline to show when the task has started executing on data.
  void startRangeExecuting() {
#ifdef HH_USE_NVTX
    executeRangeId_ = nvtxDomainRangeStartEx(taskDomain_, executeAttrib_);
#endif
  }

  /// @brief Starts tracking execution in the timeline to show when the task has started waiting for data.
  /// @details This event shows the current queue size in the payload within the attribute.
  /// @param queueSize the queue size
  void startRangeWaiting([[maybe_unused]]uint64_t const &queueSize) {
#ifdef HH_USE_NVTX
    waitAttrib_->payload.ullValue = queueSize;
    waitRangeId_ = nvtxDomainRangeStartEx(taskDomain_, waitAttrib_);
#endif
  }

  /// @brief Starts tracking waiting for memory in the timeline to show when the task has started waiting for memory
  /// from a memory manager.
  void startRangeWaitingForMemory() {
#ifdef HH_USE_NVTX
    waitForMemRangeId_ = nvtxDomainRangeStartEx(taskDomain_, waitForMemAttrib_);
#endif
  }

  /// @brief Starts tracking shutdown in the timeline to show when the task has started its shutdown phase.
  void startRangeShuttingDown() {
#ifdef HH_USE_NVTX
    shutdownRangeId_ = nvtxDomainRangeStartEx(taskDomain_, shutdownAttrib_);
#endif
  }

  /// @brief Ends tracking the initialization phase for a task
  void endRangeInitializing() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, initializeRangeId_);
#endif
  }

  /// @brief Ends tracking the execute for a task
  void endRangeExecuting() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, executeRangeId_);
#endif
  }

  /// @brief Ends tracking the waiting for data for a task
  void endRangeWaiting() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, waitRangeId_);
#endif
  }

  /// @brief Ends tracking the waiting for memory from a memory edge.
  void endRangeWaitingForMem() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, waitForMemRangeId_);
#endif
  }

  /// @brief Ends tracking the shutdown phase for a task
  void endRangeShuttingDown() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, shutdownRangeId_);
#endif
  }
#ifdef HH_USE_NVTX


  /// Gets the task domain.
  /// @return the task domain
  nvtxDomainHandle_t taskDomain() const {
    return taskDomain_;
  }
#endif

};
}
#endif //HEDGEHOG_NVTX_PROFILER_H
