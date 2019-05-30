//
// Created by anb22 on 2/25/19.
//

#ifndef HEDGEHOG_TESTGTEST_H
#define HEDGEHOG_TESTGTEST_H

#include <gtest/gtest.h>
#include "testGraph.h"

TEST(TEST_FEATURE_COLLECTION, TEST_GLOBAL_FC) {
  ASSERT_NO_FATAL_FAILURE(creationGraph());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  int ret = RUN_ALL_TESTS();

  return ret;
}

#endif //HEDGEHOG_TESTGTEST_H