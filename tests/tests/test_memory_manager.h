//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_TEST_MEMORY_MANAGER_H
#define HEDGEHOG_TESTS_TEST_MEMORY_MANAGER_H

#include "../../hedgehog/hedgehog.h"

#include "../data_structures/cuda_tasks/cuda_int_to_memory_managed_int.h"

#include "../data_structures/tasks/int_to_static_memory_managed_int.h"
#include "../data_structures/tasks/int_to_dynamic_memory_managed_int.h"
#include "../data_structures/tasks/static_memory_managed_int_dynamic_memory_managed_int_to_static_memory_managed_int.h"

#include "../data_structures/memory_managers/dynamic_memory_manager.h"

#include "../data_structures/execution_pipelines/execution_pipeline_int_to_static_memory_managed_int.h"

void testMemoryManagers() {
#ifdef HH_USE_CUDA
  int count = 0;
  std::vector<int> deviceId = {0, 0, 0, 0, 0};

  auto innerGraph = std::make_shared<hh::Graph<StaticMemoryManageData<int>, int>>();

  auto staticTask = std::make_shared<IntToStaticMemoryManagedInt>(2);
  auto dynamicTask = std::make_shared<IntToDynamicMemoryManagedInt>(2);
  auto CUDAStaticTask = std::make_shared<CudaIntToStaticMemoryManagedInt>();

  auto outTask = std::make_shared<StaticMemoryManagedIntDynamicMemoryManagedIntToStaticMemoryManagedInt>();
  auto staticMM = std::make_shared<hh::StaticMemoryManager<StaticMemoryManageData<int>>>(2);
  auto CUDAMM = std::make_shared<hh::StaticMemoryManager<StaticMemoryManageData<int>>>(2);
  auto dynamicMM = std::make_shared<DynamicMemoryManager<int>>(2);

  staticTask->connectMemoryManager(staticMM);
  dynamicTask->connectMemoryManager(dynamicMM);
  CUDAStaticTask->connectMemoryManager(CUDAMM);

  innerGraph->input(staticTask);
  innerGraph->input(dynamicTask);
  innerGraph->input(CUDAStaticTask);
  innerGraph->addEdge(staticTask, outTask);
  innerGraph->addEdge(dynamicTask, outTask);
  innerGraph->addEdge(CUDAStaticTask, outTask);
  innerGraph->output(outTask);

  auto intToStaticMemoryManagedIntExecutionPipeline =
      std::make_shared<ExecutionPipelineIntToStaticMemoryManagedInt>(innerGraph, deviceId.size(), deviceId);

  auto intToStaticMemoryManagedIntGraph = std::make_shared<hh::Graph<StaticMemoryManageData<int>, int>>();

  intToStaticMemoryManagedIntGraph->input(intToStaticMemoryManagedIntExecutionPipeline);
  intToStaticMemoryManagedIntGraph->output(intToStaticMemoryManagedIntExecutionPipeline);

  intToStaticMemoryManagedIntGraph->executeGraph();

  for (int i = 0; i < 100; ++i) { intToStaticMemoryManagedIntGraph->pushData(std::make_shared<int>(i)); }

  intToStaticMemoryManagedIntGraph->finishPushingData();

  while (auto result = intToStaticMemoryManagedIntGraph->getBlockingResult()) {
    ++count;
    result->returnToMemoryManager();
  }

  intToStaticMemoryManagedIntGraph->waitForTermination();
  ASSERT_EQ(count, 1000);
#endif
}

#endif //HEDGEHOG_TESTS_TEST_MEMORY_MANAGER_H
