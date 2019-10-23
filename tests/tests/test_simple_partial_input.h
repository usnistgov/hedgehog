//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_TEST_SIMPLE_PARTIAL_INPUT_H
#define HEDGEHOG_TESTS_TEST_SIMPLE_PARTIAL_INPUT_H

#include "../../hedgehog/hedgehog.h"
#include "../data_structures/tasks/int_to_int.h"
#include "../data_structures/execution_pipelines/execution_pipeline_int_double_to_int.h"

void testPartialInputEP() {
  size_t duplication = 10;
  std::vector<int> deviceIds(duplication, 0);
  size_t count = 0;
  auto og = std::make_shared<hh::Graph<int, int, float, double>>();
  auto ig = std::make_shared<hh::Graph<int, int, double>>();
  auto t = std::make_shared<IntToInt>();
  auto t2 = std::make_shared<IntToInt>();
  ig->input(t);
  ig->addEdge(t, t2);
  ig->output(t2);
  auto ep = std::make_shared<ExecutionPipelineIntDoubleToInt>(ig, duplication, deviceIds);
  auto it = std::make_shared<IntToInt>();
  og->input(ep);
  og->input(it);
  auto ot = std::make_shared<IntToInt>();
  og->output(ot);
  og->output(it);
  og->addEdge(ep, ot);
  og->addEdge(it, ot);
  og->executeGraph();
  for (int i = 0; i < 100; ++i) {
    og->pushData(std::make_shared<int>(i));
    og->pushData(std::make_shared<float>(i));
    og->pushData(std::make_shared<double>(i));
  }

  og->finishPushingData();
  while (og->getBlockingResult()) { ++count; }
  ASSERT_EQ(count, 1200);
  og->waitForTermination();
}

void testSimplePartialInput() {
  for (int r = 0; r < 100; ++r) {
    size_t count = 0;

    auto g = std::make_shared<hh::Graph<int, int, float>>();
    auto t = std::make_shared<IntToInt>();
    auto t2 = std::make_shared<IntToInt>();

    g->input(t);
    g->addEdge(t, t2);
    g->output(t2);

    g->executeGraph();

    for (int i = 0; i < 100; ++i) {
      g->pushData(std::make_shared<int>(i));
      g->pushData(std::make_shared<float>(i));
    }
    g->finishPushingData();
    while (g->getBlockingResult()) { ++count; }
    g->waitForTermination();

    ASSERT_EQ(count, 100);
  }
}

#endif //HEDGEHOG_TESTS_TEST_SIMPLE_PARTIAL_INPUT_H
