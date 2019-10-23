//
// Created by tjb3 on 10/18/19.
//

#include <hedgehog/api/graph.h>

#include "test_link.h"

#include <hedgehog/api/tools/graph_signal_handler.h>

#include <gtest/gtest.h>
#include "../data_structures/cuda_tasks/cuda_link_example.h"
#include "tests/data_structures/cuda_tasks/cuda_link2_example.h"



void testLink2() {
#ifdef HH_USE_CUDA
  hh::Graph<int, int> g("SimpleLinkGraph");
  auto t = std::make_shared<CudaLinkExample>();
  auto t2 = std::make_shared<CudaLink2Example>();
  size_t count = 0;

  g.input(t);
  g.addEdge(t, t2);
  g.output(t2);

  hh::GraphSignalHandler<int, int>::registerGraph(&g);
  hh::GraphSignalHandler<int, int>::registerSignal();

  g.executeGraph();

  for (int i = 0; i < 100; ++i) { g.pushData(std::make_shared<int>(i)); }

  g.finishPushingData();

  while((g.getBlockingResult())) {
    ++count;
  }

  g.waitForTermination();

  ASSERT_EQ(count, 100);
#endif
}
