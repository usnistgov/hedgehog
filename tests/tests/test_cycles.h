//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_TEST_CYCLES_H
#define HEDGEHOG_TESTS_TEST_CYCLES_H

#include "../../hedgehog/hedgehog.h"
#include "../data_structures/tasks/float_to_int_cycle.h"
#include "../data_structures/tasks/int_double_char_to_float.h"

void testCycles() {
  int count = 0;
  auto graph = std::make_shared<hh::Graph<int, int, double, char>>();
  auto input = std::make_shared<IntDoubleCharToFloat>(5);
  auto cycleTask = std::make_shared<FloatToIntCycle>(3);
  auto state = std::make_shared<StateIntFloatToInt>();
  auto stateManager = std::make_shared<hh::StateManager<int, int, float>>(state);

  graph->input(input);
  graph->addEdge(input, cycleTask);
  graph->addEdge(cycleTask, input);
  graph->addEdge(input, stateManager);
  graph->output(stateManager);

  graph->executeGraph();

  for (int i = 0; i < 100; i++) {
    graph->pushData(std::make_shared<int>(i));
    graph->pushData(std::make_shared<double>(i));
    graph->pushData(std::make_shared<char>(i));
  }

  graph->finishPushingData();

  while (graph->getBlockingResult()) { count++; }
  graph->waitForTermination();

  ASSERT_EQ(count, 1200);
}

#endif //HEDGEHOG_TESTS_TEST_CYCLES_H
