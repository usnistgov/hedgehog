//
// Created by anb22 on 9/24/19.
//

#ifndef HEDGEHOG_TESTS_TEST_SMALL_GRAPH_H
#define HEDGEHOG_TESTS_TEST_SMALL_GRAPH_H

#include "../../hedgehog/hedgehog.h"
#include "../data_structures/tasks/int_double_char_to_float.h"

void testSmallGraph() {
  hh::Graph<float, int, double, char> g("GraphOutput");
  auto t = std::make_shared<IntDoubleCharToFloat>();
  size_t count = 0;

  g.input(t);
  g.output(t);

  g.executeGraph();

  for (uint64_t i = 0; i < 100; ++i) { g.pushData(std::make_shared<int>(i)); }

  g.finishPushingData();

  while ((g.getBlockingResult())) { ++count; }

  g.waitForTermination();

  ASSERT_EQ(count, 100);
}

#endif //HEDGEHOG_TESTS_TEST_SMALL_GRAPH_H
