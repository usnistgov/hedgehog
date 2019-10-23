//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_TEST_EXECUTION_PIPELINE_H
#define HEDGEHOG_TESTS_TEST_EXECUTION_PIPELINE_H

void testEP() {
  int count = 0;
  std::vector<int> deviceIds = {0, 0, 0};

  auto insideGraph = std::make_shared<hh::Graph<int, int, double>>();
  auto insideInput = std::make_shared<IntDoubleFloat>();
  auto insideOutput = std::make_shared<IntFloatToInt>();
  insideGraph->input(insideInput);
  insideGraph->addEdge(insideInput, insideOutput);
  insideGraph->output(insideOutput);
  auto ep = std::make_shared<ExecutionPipelineIntDoubleToInt>(insideGraph, deviceIds.size(), deviceIds);

  auto outsideGraph = std::make_shared<hh::Graph<int, int>>();
  auto outsideInput = std::make_shared<IntToInt>();
  auto outsideOutput = std::make_shared<IntToInt>();
  outsideGraph->input(outsideInput);
  outsideGraph->addEdge(outsideInput, ep);
  outsideGraph->addEdge(ep, outsideOutput);
  outsideGraph->output(outsideOutput);

  outsideGraph->executeGraph();

  for (int i = 0; i < 100; ++i) { outsideGraph->pushData(std::make_shared<int>(i)); }
  outsideGraph->finishPushingData();

  while (outsideGraph->getBlockingResult()) { ++count; }
  outsideGraph->waitForTermination();

  ASSERT_EQ(count, 300);
}

#endif //HEDGEHOG_TESTS_TEST_EXECUTION_PIPELINE_H
