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
#include "tests_impl/test_tools.h"
#include "tests_impl/test_simple_graph.h"
#include "tests_impl/test_complex_graph.h"
#include "tests_impl/test_state_manager.h"
#include "tests_impl/test_memory_manager.h"

#include "tests_impl/compile_time_tests/test_basic.h"
#include "tests_impl/compile_time_tests/test_critical_path.h"
#include "tests_impl/compile_time_tests/test_cycle_improved.h"
#include "tests_impl/compile_time_tests/test_error_message.h"
#include "tests_impl/compile_time_tests/test_matrix_multiplication.h"
#include "tests_impl/compile_time_tests/test_race_condition.h"

TEST(HedgehogGraph, simpleGraph) {
  ASSERT_NO_THROW(graphSimpleExecution());
  ASSERT_NO_THROW(graphSimpleDataTransfer());
  ASSERT_NO_THROW(graphSimpleDataTransferTaskGroup());
  ASSERT_NO_THROW(graphSimpleMultiGroups());
}

TEST(HedgehogGraph, complexGraph) {
  ASSERT_NO_THROW(complexGraphTestEdges());
  ASSERT_NO_THROW(complexGraphComposition());
  ASSERT_NO_THROW(testMemoryManagers());
  ASSERT_NO_THROW(testSharedState());
  ASSERT_NO_THROW(testSimpleRecursiveEP());
  ASSERT_NO_THROW(testEPComposition());
  ASSERT_NO_THROW(testComplexEPComposition());
  ASSERT_NO_THROW(testCustomizedNodes());
  ASSERT_NO_THROW(testAtomicTask());
}

TEST(HedgehogTools, metafunctions) {
  ASSERT_NO_THROW(testTupleIntersection());
}

TEST(HedgehogTools, pool) {
  ASSERT_NO_THROW(testPool());
}

#ifdef HH_ENABLE_HH_CX

TEST(TEST_STATIC_ANALYSIS, TEST_CRITICAL_PATH) {
  ASSERT_NO_FATAL_FAILURE(testCriticalPath());
}

TEST(TEST_STATIC_ANALYSIS, TEST_DATA_RACES) {
  ASSERT_NO_FATAL_FAILURE(testDataRacesWithAllEdgesTreatedAsRW());
  ASSERT_NO_FATAL_FAILURE(testDataRacesWithAllConstEdges());
  ASSERT_NO_FATAL_FAILURE(testDataRacesWithSomeEdgesTreatedAsRO());
}

TEST(TEST_STATIC_ANALYSIS, TEST_CYCLE_DETECTION) {
  ASSERT_NO_FATAL_FAILURE(testTarjanSimpleCycle());
  ASSERT_NO_FATAL_FAILURE(testTarjanSameNodeCycle());
  ASSERT_NO_FATAL_FAILURE(testTarjanComplexCycles());
  ASSERT_NO_FATAL_FAILURE(testTarjanComplexCyclesConst());
  ASSERT_NO_FATAL_FAILURE(testCycleMultiInputs());
  ASSERT_NO_FATAL_FAILURE(testTarjanSimpleCycle3Nodes());
  ASSERT_NO_FATAL_FAILURE(testCyclesWithoutCanTerminate());
  ASSERT_NO_FATAL_FAILURE(testCyclesWithCanTerminate());
}

TEST(TEST_STATIC_ANALYSIS, TEST_COMPOSITION){
  ASSERT_NO_FATAL_FAILURE(testCompositionAsStaticNode());
  ASSERT_NO_FATAL_FAILURE(testCompositionAsStaticGraph());
}


TEST(TEST_STATIC_ANALYSIS, TEST_BASIC){
  ASSERT_NO_FATAL_FAILURE(testTarjanNoCycle());
  ASSERT_NO_FATAL_FAILURE(testSameNodeType());
}

TEST(TEST_STATIC_ANALYSIS, TEST_ERROR_MESSAGE){
  ASSERT_NO_FATAL_FAILURE(testErrorMessage());
}

TEST(TEST_STATIC_ANALYSIS, MATRIX_MULTIPLICATION){
  ASSERT_NO_FATAL_FAILURE(testMatrixMultiplication());
}

#endif //HH_ENABLE_HH_CX