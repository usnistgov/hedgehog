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


#ifndef HEDGEHOG_TESTS_TEST_MEMORY_MANAGER_H
#define HEDGEHOG_TESTS_TEST_MEMORY_MANAGER_H

#include "../../hedgehog/hedgehog.h"
#ifdef HH_USE_CUDA
#include "../data_structures/cuda_tasks/cuda_int_to_memory_managed_int.h"
#endif
#include "../data_structures/tasks/int_to_static_memory_managed_int.h"
#include "../data_structures/tasks/int_to_dynamic_memory_managed_int.h"
#include "../data_structures/tasks/static_memory_managed_int_dynamic_memory_managed_int_to_static_memory_managed_int.h"

#include "../data_structures/memory_managers/dynamic_memory_manager.h"

#include "../data_structures/execution_pipelines/execution_pipeline_int_to_static_memory_managed_int.h"

void testMemoryManagers() {
  int count = 0;
  std::vector<int> deviceId = {0, 0, 0, 0, 0};

  auto innerGraph = std::make_shared<hh::Graph<StaticMemoryManageData<int>, int>>();

  auto staticTask = std::make_shared<IntToStaticMemoryManagedInt>(2);
  auto dynamicTask = std::make_shared<IntToDynamicMemoryManagedInt>(2);
#ifdef HH_USE_CUDA
  auto CUDAStaticTask = std::make_shared<CudaIntToStaticMemoryManagedInt>();
#endif
  auto outTask = std::make_shared<StaticMemoryManagedIntDynamicMemoryManagedIntToStaticMemoryManagedInt>();
  auto staticMM = std::make_shared<hh::StaticMemoryManager<StaticMemoryManageData<int>>>(2);
#ifdef HH_USE_CUDA
  auto CUDAMM = std::make_shared<hh::StaticMemoryManager<StaticMemoryManageData<int>>>(2);
#endif
  auto dynamicMM = std::make_shared<DynamicMemoryManager<int>>(2);

  staticTask->connectMemoryManager(staticMM);
  dynamicTask->connectMemoryManager(dynamicMM);
#ifdef HH_USE_CUDA
  CUDAStaticTask->connectMemoryManager(CUDAMM);
#endif
  innerGraph->input(staticTask);
  innerGraph->input(dynamicTask);
#ifdef HH_USE_CUDA
  innerGraph->input(CUDAStaticTask);
#endif
  innerGraph->addEdge(staticTask, outTask);
  innerGraph->addEdge(dynamicTask, outTask);
#ifdef HH_USE_CUDA
  innerGraph->addEdge(CUDAStaticTask, outTask);
#endif
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
#ifdef HH_USE_CUDA
  ASSERT_EQ(count, 1000);
#else
  ASSERT_EQ(count, 500);
#endif
}

#endif //HEDGEHOG_TESTS_TEST_MEMORY_MANAGER_H
