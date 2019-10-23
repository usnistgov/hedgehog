//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_TEST_EXECUTION_PIPELINE_COMPOSITION_H
#define HEDGEHOG_TESTS_TEST_EXECUTION_PIPELINE_COMPOSITION_H

#include "../data_structures/execution_pipelines/execution_pipeline_int_to_int.h"
#include "../data_structures/tasks/int_to_int.h"

std::shared_ptr<IntToInt> createTask() { return std::make_shared<IntToInt>(); }

std::shared_ptr<hh::Graph<int, int>> innerGraph() {
  auto g = std::make_shared<hh::Graph<int, int>>();
  auto t = createTask();
  g->input(t);
  g->output(t);
  return g;
}

std::shared_ptr<hh::Graph<int, int>> wrapperGraph(const std::shared_ptr<hh::Graph<int, int>> &innerGraph) {
  std::vector<int> deviceIds(4, 0);
  auto g = std::make_shared<hh::Graph<int, int>>();
  auto ep = std::make_shared<ExecutionPipelineIntToInt>(innerGraph, deviceIds.size(), deviceIds);
  auto t1 = createTask(), t2 = createTask();
  g->input(t1);
  g->addEdge(t1, ep);
  g->addEdge(ep, t2);
  g->output(t2);
  return g;
}

void testEPComposition() {
  std::shared_ptr<hh::Graph<int, int>>
      graph = nullptr,
      tempGraph = innerGraph();

  size_t count = 0;
  for (int i = 0; i < 3; ++i) { tempGraph = wrapperGraph(tempGraph); }
  graph = wrapperGraph(tempGraph);
  graph->executeGraph();

  for (int i = 0; i < 100; ++i) { graph->pushData(std::make_shared<int>(i)); }
  graph->finishPushingData();
  while (graph->getBlockingResult()) { count++; }
  graph->waitForTermination();

  ASSERT_EQ(count, 25600);
}

#endif //HEDGEHOG_TESTS_TEST_EXECUTION_PIPELINE_COMPOSITION_H
