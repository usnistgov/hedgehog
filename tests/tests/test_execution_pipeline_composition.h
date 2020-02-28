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

  ASSERT_EQ(count, (size_t)25600);
}

#endif //HEDGEHOG_TESTS_TEST_EXECUTION_PIPELINE_COMPOSITION_H
