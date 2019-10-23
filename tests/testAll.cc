//
// Created by anb22 on 2/25/19.
//

#ifndef HEDGEHOG_TESTS_TESTGTEST_H
#define HEDGEHOG_TESTS_TESTGTEST_H

#include <gtest/gtest.h>

#include "tests/test_memory_manager.h"
#include "tests/test_small_graph.h"
#include "tests/test_complex_graph.h"
#include "tests/test_execution_pipeline.h"
#include "tests/test_execution_pipeline_composition.h"
#include "tests/test_cycles.h"
#include "tests/test_simple_partial_input.h"

#ifdef HH_USE_CUDA
#include "tests/test_cuda.h"
#include "tests/test_link.h"
#include "tests/test_link2.h"
#endif

TEST(TEST_GRAPH, TEST_GLOBAL_GRAPH) {
  ASSERT_NO_FATAL_FAILURE(testSmallGraph());
  ASSERT_NO_FATAL_FAILURE(testComplexGraph());
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

TEST(TEST_GRAPH, TEST_LINK) {
#ifdef HH_USE_CUDA
  ASSERT_NO_FATAL_FAILURE(testLink());
  ASSERT_NO_FATAL_FAILURE(testLink2());
#endif //HH_USE_CUDA
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}

#endif //HEDGEHOG_TESTS_TESTGTEST_H