#include "data/matrix_data.h"
#include "data/dynamic_matrix_data.h"
#include "tests/graphs/memory_manager/ep/iiep_mm.h"
#include "memory_manager/cuda_mm.h"
#include "memory_manager/dynamic_mm.h"
#include "memory_manager/static_mm.h"
#include "tests/graphs/memory_manager/task/output_md_task.h"
#include "task/my_static_task.h"
#include "task/my_cuda_task.h"
#include "task/my_dynamic_task.h"

void testMemoryManagers() {
#ifdef HH_USE_CUDA
  std::vector<int> vDevices = {0, 0, 0, 0, 0};
  std::shared_ptr<IIEPMM> iiep = nullptr;

  auto
      insideGraph = std::make_shared<Graph<MatrixData<int>, int>>("MMGraph"),
      outerGraph = std::make_shared<Graph<MatrixData<int>, int>>("MMGraph");

  std::shared_ptr<MyStaticTask> staticTask = nullptr;
  std::shared_ptr<StaticMM<int>> staticMM = nullptr;
  std::shared_ptr<MyCUDATask> cudaTask = nullptr;
  std::shared_ptr<CudaMM<int>> cudaMM = nullptr;
  std::shared_ptr<MyDynamicTask> dynamicTask = nullptr;
  std::shared_ptr<DynamicMM<int>> dynMM = nullptr;
  std::shared_ptr<OutputMDTask> outputTask = nullptr;
  std::shared_ptr<MatrixData<int>> result = nullptr;
  size_t count = 0;

  outputTask = std::make_shared<OutputMDTask>();
  insideGraph->output(outputTask);

  staticTask = std::make_shared<MyStaticTask>();
  staticMM = std::make_shared<StaticMM<int>>(2);
  staticTask->connectMemoryManager(std::static_pointer_cast<AbstractMemoryManager<MatrixData<int>>>(staticMM));
  insideGraph->input(staticTask);
  insideGraph->addEdge(staticTask, outputTask);

  cudaTask = std::make_shared<MyCUDATask>();
  cudaMM = std::make_shared<CudaMM<int>>(2);
  cudaTask->connectMemoryManager(std::static_pointer_cast<AbstractMemoryManager<MatrixData<int>>>(cudaMM));
  insideGraph->input(cudaTask);
  insideGraph->addEdge(cudaTask, outputTask);

  dynamicTask = std::make_shared<MyDynamicTask>();
  dynMM = std::make_shared<DynamicMM<int>>(2);
  dynamicTask->connectMemoryManager(std::static_pointer_cast<AbstractMemoryManager<DynamicMatrixData<int>>>(dynMM));
  insideGraph->input(dynamicTask);
  insideGraph->addEdge(dynamicTask, outputTask);

  iiep = std::make_shared<IIEPMM>(insideGraph, 5, vDevices, false);

  outerGraph->input(iiep);
  outerGraph->output(iiep);

  outerGraph->executeGraph();

  for (int i = 0; i < 100; ++i) { outerGraph->pushData(std::make_shared<int>(i)); }

  outerGraph->finishPushingData();

  while ((result = outerGraph->getBlockingResult())) {
    ++count;
    result->returnToMemoryManager();
  }

  outerGraph->waitForTermination();
#endif //HH_USE_CUDA
}