//
// Created by tjb3 on 7/31/19.
//

#ifndef HEDGEHOG_NVTX_PROFILER_H
#define HEDGEHOG_NVTX_PROFILER_H

#ifdef HH_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include <string_view>
#include <string.h>
#include <iostream>

#define NVTX_COLOR_INITIALIZING    0xFF123456
#define NVTX_COLOR_EXECUTING       0xFF72ff68
#define NVTX_COLOR_WAITING         0xFFff7f83
#define NVTX_COLOR_WAITING_FOR_MEM 0xFFffc86a
#define NVTX_COLOR_RELEASE_MEM     0xFF7fbdff
#define NVTX_COLOR_SHUTTING_DOWN   0xFF654321

class NvtxProfiler {
 private:
#ifdef HH_USE_NVTX
  std::string_view initializeName_{};
  std::string_view executeName_{};
  std::string_view waitName_{};
  std::string_view waitForMemName_{};
  std::string_view releaseMemName_{};
  std::string_view shutdownName_{};

  nvtxDomainHandle_t taskDomain_;

  nvtxStringHandle_t initializeString_{};
  nvtxStringHandle_t executeString_{};
  nvtxStringHandle_t waitString_{};
  nvtxStringHandle_t waitForMemString_{};
  nvtxStringHandle_t releaseMemString_{};
  nvtxStringHandle_t shutdownString_{};

  nvtxEventAttributes_t *initializeAttrib_;
  nvtxEventAttributes_t *executeAttrib_;
  nvtxEventAttributes_t *waitAttrib_;
  nvtxEventAttributes_t *waitForMemAttrib_;
  nvtxEventAttributes_t *releaseMemAttrib_;
  nvtxEventAttributes_t *shutdownAttrib_;

  nvtxRangeId_t initializeRangeId_ = 0;
  nvtxRangeId_t executeRangeId_ = 0;
  nvtxRangeId_t waitRangeId_ = 0;
  nvtxRangeId_t waitForMemRangeId_ = 0;
  nvtxRangeId_t shutdownRangeId_ = 0;


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

  NvtxProfiler() = delete;

  NvtxProfiler(std::string_view const &taskName [[maybe_unused]])
  {
#ifdef HH_USE_NVTX
    taskDomain_ = nvtxDomainCreateA(taskName.data());

    initializeAttrib_ = createEventAttribute(NVTX_COLOR_INITIALIZING);
    executeAttrib_ = createEventAttribute(NVTX_COLOR_EXECUTING);

    waitAttrib_ = createEventAttribute(NVTX_COLOR_WAITING);
    waitAttrib_->payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;

    waitAttrib_->payload.ullValue = 0;

    waitForMemAttrib_ = createEventAttribute(NVTX_COLOR_WAITING_FOR_MEM);
    releaseMemAttrib_ = createEventAttribute(NVTX_COLOR_RELEASE_MEM);
    shutdownAttrib_ = createEventAttribute(NVTX_COLOR_SHUTTING_DOWN);
#endif
  }

  ~NvtxProfiler() {
#ifdef HH_USE_NVTX
    delete initializeAttrib_;
    delete executeAttrib_;
    delete waitAttrib_;
    delete waitForMemAttrib_;
    delete releaseMemAttrib_;
    delete shutdownAttrib_;

    nvtxDomainDestroy(taskDomain_);
#endif
  }

  void initialize([[maybe_unused]]int threadId)
  {
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

  /**
   * Adds a release marker into the timeline to show when the task released memory.
   */
  void addReleaseMarker() {
#ifdef HH_USE_NVTX
    nvtxDomainMarkEx(taskDomain_, releaseMemAttrib_);
#endif
  }

  /**
   * Starts tracking intialization in the timeline to show when the task has started its initialization phase.
   */
  void startRangeInitializing() {
#ifdef HH_USE_NVTX
    initializeRangeId_ = nvtxDomainRangeStartEx(taskDomain_, initializeAttrib_);
#endif
  }

  /**
   * Starts tracking execution in the timeline to show when the task has started executing on data.
   */
  void startRangeExecuting() {
#ifdef HH_USE_NVTX
    executeRangeId_ = nvtxDomainRangeStartEx(taskDomain_, executeAttrib_);
#endif
  }

  /**
   * Starts tracking execution in the timeline to show when the task has started waiting for data.
   * This event shows the current queue size in the payload within the attribute.
   * @param queueSize the queue size
   */
  void startRangeWaiting([[maybe_unused]]uint64_t const &queueSize) {
#ifdef HH_USE_NVTX
    waitAttrib_->payload.ullValue = queueSize;
    waitRangeId_ = nvtxDomainRangeStartEx(taskDomain_, waitAttrib_);
#endif
  }

  /**
   * Starts tracking waiting for memory in the timeline to show when the task has started waiting for memory from a memory manager.
   */
  void startRangeWaitingForMemory() {
#ifdef HH_USE_NVTX
    waitForMemRangeId_ = nvtxDomainRangeStartEx(taskDomain_, waitForMemAttrib_);
#endif
  }

  /**
   * Starts tracking shutdown in the timeline to show when the task has started its shutdown phase.
   */
  void startRangeShuttingDown() {
#ifdef HH_USE_NVTX
    shutdownRangeId_ = nvtxDomainRangeStartEx(taskDomain_, shutdownAttrib_);
#endif
  }

  /**
   * Ends tracking the initialization phase for a task
   */
  void endRangeInitializing() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, initializeRangeId_);
#endif
  }

  /**
   * Ends tracking the execute for a task
   */
  void endRangeExecuting() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, executeRangeId_);
#endif
  }

  /**
   * Ends tracking the waiting for data for a task
   */
  void endRangeWaiting() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, waitRangeId_);
#endif
  }

  /**
   * Ends tracking the waiting for memory from a memory edge.
   */
  void endRangeWaitingForMem() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, waitForMemRangeId_);
#endif
  }

  /**
   * Ends tracking the shutdown phase for a task
   */
  void endRangeShuttingDown() {
#ifdef HH_USE_NVTX
    nvtxDomainRangeEnd(taskDomain_, shutdownRangeId_);
#endif
  }
#ifdef HH_USE_NVTX
  /**
   * Gets the task domain.
   * @return the task domain
   */
  nvtxDomainHandle_t taskDomain() const {
    return taskDomain_;
  }
#endif


};


#endif //HEDGEHOG_NVTX_PROFILER_H
