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


#include <gtest/gtest.h>
#include "../../hedgehog/hedgehog.h"
#include "../data_structures/tasks/static_size_t_to_managed_memory.h"
#include "../data_structures/tasks/dynamic_size_t_to_managed_memory.h"

#ifdef HH_USE_CUDA
#include "../data_structures/tasks/static_size_t_to_managed_memory_cuda.h"
#endif //HH_USE_CUDA


void testMemoryManagers() {
  int count = 0, countStatic = 0, countDynamic = 0;

  auto graph = std::make_shared<hh::Graph<1, size_t, StaticManagedMemory, DynamicManagedMemory>>();

  auto staticTask = std::make_shared<StaticSizeTToManagedMemory>(2);
  auto dynamicTask = std::make_shared<DynamicSizeTToManagedMemory>(2);

  auto staticMM = std::make_shared<hh::StaticMemoryManager<StaticManagedMemory, size_t>>(2, 2);
  auto dynamicMM = std::make_shared<hh::MemoryManager<DynamicManagedMemory>>(2);

  staticTask->connectMemoryManager(staticMM);
  dynamicTask->connectMemoryManager(dynamicMM);

  graph->inputs(staticTask);
  graph->inputs(dynamicTask);

  graph->outputs(staticTask);
  graph->outputs(dynamicTask);

#ifdef HH_USE_CUDA
  auto CUDAStaticTask = std::make_shared<StaticSizeTToManagedACUDA>(2);
  auto CUDAMM = std::make_shared<hh::StaticMemoryManager<StaticManagedMemory, size_t>>(2, 2);
  CUDAStaticTask->connectMemoryManager(CUDAMM);
  graph->inputs(CUDAStaticTask);
  graph->outputs(CUDAStaticTask);
#endif

  graph->executeGraph();
  for (int i = 0; i < 100; ++i) { graph->pushData(std::make_shared<size_t>(i)); }
  graph->finishPushingData();

  while (auto result = graph->getBlockingResult()) {
    ++count;
    std::visit(hh::ResultVisitor{
        [&countStatic](std::shared_ptr<StaticManagedMemory> &val) {
          ++countStatic;
          val->returnToMemoryManager();
          },
        [&countDynamic](std::shared_ptr<DynamicManagedMemory> &val) {
          ++countDynamic;
          val->dealloc();
          val->returnToMemoryManager();
          },
    }, *result);
  }

  graph->waitForTermination();

#ifdef HH_USE_CUDA
  ASSERT_EQ(count, 300);
#else
  ASSERT_EQ(count, 200);
#endif
}

void testPool(){
  std::array<std::shared_ptr<int>, 3> data{};
  hh::tool::Pool<int> pool(3);

  ASSERT_EQ(pool.size(), 3);
  ASSERT_EQ(pool.capacity(), 3);
  ASSERT_EQ(pool.empty(), false);

  data.at(0) = pool.pop_front();
  ASSERT_EQ(pool.size(), 2);
  ASSERT_EQ(pool.capacity(), 3);
  ASSERT_EQ(pool.empty(), false);
  data.at(1) = pool.pop_front();
  ASSERT_EQ(pool.size(), 1);
  ASSERT_EQ(pool.capacity(), 3);
  ASSERT_EQ(pool.empty(), false);
  data.at(2) = pool.pop_front();
  ASSERT_EQ(pool.size(), 0);
  ASSERT_EQ(pool.capacity(), 3);
  ASSERT_EQ(pool.empty(), true);


  pool.push_back(data.at(2));
  ASSERT_EQ(pool.size(), 1);
  ASSERT_EQ(pool.capacity(), 3);
  ASSERT_EQ(pool.empty(), false);
  pool.push_back(data.at(1));
  ASSERT_EQ(pool.size(), 2);
  ASSERT_EQ(pool.capacity(), 3);
  ASSERT_EQ(pool.empty(), false);
  pool.push_back(data.at(0));
  ASSERT_EQ(pool.size(), 3);
  ASSERT_EQ(pool.capacity(), 3);
  ASSERT_EQ(pool.empty(), false);
}

