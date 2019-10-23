//
// Created by 775backup on 2019-04-17.
//

#ifndef HEDGEHOG_HEDGEHOG_H
#define HEDGEHOG_HEDGEHOG_H

/// @brief Main include file, including all files that are used when using Hedgehog

#include "api/graph.h"
#include "api/abstract_task.h"
#ifdef HH_USE_CUDA
#include "api/abstract_cuda_task.h"
#endif // HH_USE_CUDA
#include "api/memory_manager/memory_data.h"
#include "api/memory_manager/static_memory_manager.h"
#include "api/memory_manager/abstract_memory_manager.h"
#include "api/state_manager/state_manager.h"
#include "api/abstract_execution_pipeline.h"

#endif //HEDGEHOG_HEDGEHOG_H
