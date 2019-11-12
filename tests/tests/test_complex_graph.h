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


#ifndef HEDGEHOG_TESTS_TEST_COMPLEX_GRAPH_H
#define HEDGEHOG_TESTS_TEST_COMPLEX_GRAPH_H

#include "../data_structures/execution_pipelines/execution_pipeline_int_double_to_int.h"

#include "../data_structures/tasks/int_float_to_double.h"
#include "../data_structures/tasks/int_float_to_int.h"
#include "../data_structures/tasks/int_double_float.h"
#include "../data_structures/tasks/int_to_int.h"
#include "../data_structures/states/state_int_float_to_int.h"

void testComplexGraph() {
  int count = 0;
  std::vector<int> deviceIds = {0, 1, 2, 3, 4, 5};

  auto innerGraph = std::make_shared<hh::Graph<int, int, double>>("Inner Graph");
  auto innerInput1 = std::make_shared<IntDoubleFloat>();
  auto innerInput2 = std::make_shared<IntDoubleFloat>();
  auto innerOut1 = std::make_shared<IntFloatToInt>();
  auto innerOut2 = std::make_shared<IntToInt>();
  auto state = std::make_shared<StateIntFloatToInt>();

  auto stateManager = std::make_shared<hh::StateManager<int, int, float>>(state);

  innerGraph->input(innerInput1);
  innerGraph->input(innerInput2);
  innerGraph->addEdge(innerInput1, innerOut1);
  innerGraph->addEdge(innerInput2, innerOut1);
  innerGraph->addEdge(innerInput1, stateManager);
  innerGraph->addEdge(innerInput2, stateManager);
  innerGraph->addEdge(stateManager, innerOut2);
  innerGraph->addEdge(stateManager, innerOut2);
  innerGraph->output(innerOut1);
  innerGraph->output(innerOut2);
  auto innerEP = std::make_shared<ExecutionPipelineIntDoubleToInt>(innerGraph, deviceIds.size(), deviceIds);

  hh::Graph<int, int, float> graph("Outer Graph");
  auto outerInput1 = std::make_shared<IntFloatToDouble>();
  auto outerInput2 = std::make_shared<IntFloatToInt>();
  auto outerOut1 = std::make_shared<IntToInt>();
  auto outerOut2 = std::make_shared<IntToInt>();

  graph.input(outerInput1);
  graph.input(outerInput2);
  graph.addEdge(outerInput1, innerEP);
  graph.addEdge(outerInput2, innerEP);
  graph.addEdge(innerEP, outerOut1);
  graph.addEdge(innerEP, outerOut2);
  graph.output(outerOut1);
  graph.output(outerOut2);

  graph.executeGraph();

  for (int i = 0; i < 100; ++i) {
    graph.pushData(std::make_shared<int>());
    graph.pushData(std::make_shared<float>());
  }

  graph.finishPushingData();

  while (auto result = graph.getBlockingResult()) { count++; }

  graph.waitForTermination();

  ASSERT_EQ(count, 19200);
}

#endif //HEDGEHOG_TESTS_TEST_COMPLEX_GRAPH_H
