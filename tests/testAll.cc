//
// Created by anb22 on 2/25/19.
//

#ifndef HEDGEHOG_TESTGTEST_H
#define HEDGEHOG_TESTGTEST_H

#include <gtest/gtest.h>
#include "graphs/big_example/test_big_example.h"
#include "graphs/cuda/test_cuda.h"
#include "graphs/cycle/test_cycle.h"
#include "graphs/ep/test_ep.h"
#include "graphs/ep_composition/test_ep_composition.h"
#include "graphs/memory_manager/test_mm_example.h"
#include "graphs/small_example/test_small_example.h"
#include "graphs/partial_input/test_partial_input.h"

TEST(TEST_GRAPH, TEST_GLOBAL_GRAPH) {
  ASSERT_NO_FATAL_FAILURE(testSmallGraph());
  ASSERT_NO_FATAL_FAILURE(testBigExample());
}

TEST(TEST_GRAPH, TEST_GLOBAL_GRAPH_CUDA) {
#ifdef HH_USE_CUDA
  ASSERT_NO_FATAL_FAILURE(testCUDA());
  ASSERT_NO_FATAL_FAILURE(testMemoryManagers());
#endif //HH_USE_CUDA
}

TEST(TEST_GRAPH, TEST_GLOBAL_GRAPH_EP) {
  ASSERT_NO_FATAL_FAILURE(testEP());
}

TEST(TEST_GRAPH, TEST_GLOBAL_GRAPH_EP_COMPO) {
  ASSERT_NO_FATAL_FAILURE(testEPComposition());
}

TEST(TEST_GRAPH, TEST_GLOBAL_GRAPH_CYCLES) {
  ASSERT_NO_FATAL_FAILURE(testCycles());
}

TEST(TEST_GRAPH, TEST_GLOBAL_PARTIAL_INPUT) {
  ASSERT_NO_FATAL_FAILURE(testSimplePartialInput());
  ASSERT_NO_FATAL_FAILURE(testPartialInputEP());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}

#endif //HEDGEHOG_TESTGTEST_H