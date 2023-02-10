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

#ifndef HEDGEHOG_TEST_COMPLEX_GRAPH_H_
#define HEDGEHOG_TEST_COMPLEX_GRAPH_H_

#include "../../hedgehog/hedgehog.h"
#include "../data_structures/states/int_state.h"
#include "../data_structures/states/priority_queue_state_manager.h"

#include "../data_structures/tasks/int_float_double_task.h"
#include "../data_structures/tasks/int_int_priority_queue_task.h"
#include "../data_structures/graph/int_float_double_graph.h"
#include "../data_structures/execution_pipelines/int_float_double_execution_pipeline.h"
#include "../data_structures/execution_pipelines/int_execution_pipeline.h"

void complexGraphTestEdges() {
  size_t nbResults = 0, nbResultsInt = 0, nbResultsFloat = 0, nbResultsDouble = 0;
  hh::Graph<3, int, float, double, int, float, double> graph;
  auto inputIFDTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto inputITask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto inputFTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto inputDTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto inputIFTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto inputIDTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto inputFDTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto mid = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto outputIFDTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto outputITask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto outputFTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto outputDTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto outputIFTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto outputIDTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);
  auto outputFDTask = std::make_shared<IntFloatDoubleTask>(rand() % 3 + 1);

  graph.input<int>(inputIFDTask);
  graph.input<float>(inputIFDTask);
  graph.input<double>(inputIFDTask);
  graph.input<int>(inputITask);
  graph.input<float>(inputFTask);
  graph.input<double>(inputDTask);
  graph.input<int>(inputIFTask);
  graph.input<float>(inputIFTask);
  graph.input<int>(inputIDTask);
  graph.input<double>(inputIDTask);
  graph.input<float>(inputFDTask);
  graph.input<double>(inputFDTask);

  graph.edge<int>(inputIFDTask, mid);
  graph.edge<float>(inputIFDTask, mid);
  graph.edge<double>(inputIFDTask, mid);
  graph.edge<int>(inputITask, mid);
  graph.edge<float>(inputFTask, mid);
  graph.edge<double>(inputDTask, mid);
  graph.edge<int>(inputIFTask, mid);
  graph.edge<float>(inputIFTask, mid);
  graph.edge<int>(inputIDTask, mid);
  graph.edge<double>(inputIDTask, mid);
  graph.edge<float>(inputFDTask, mid);
  graph.edge<double>(inputFDTask, mid);

  graph.edge<int>(mid, outputIFDTask);
  graph.edge<float>(mid, outputIFDTask);
  graph.edge<double>(mid, outputIFDTask);
  graph.edge<int>(mid, outputITask);
  graph.edge<float>(mid, outputFTask);
  graph.edge<double>(mid, outputDTask);
  graph.edge<int>(mid, outputIFTask);
  graph.edge<float>(mid, outputIFTask);
  graph.edge<int>(mid, outputIDTask);
  graph.edge<double>(mid, outputIDTask);
  graph.edge<float>(mid, outputFDTask);
  graph.edge<double>(mid, outputFDTask);

  graph.output<int>(outputIFDTask);
  graph.output<float>(outputIFDTask);
  graph.output<double>(outputIFDTask);
  graph.output<int>(outputITask);
  graph.output<float>(outputFTask);
  graph.output<double>(outputDTask);
  graph.output<int>(outputIFTask);
  graph.output<float>(outputIFTask);
  graph.output<int>(outputIDTask);
  graph.output<double>(outputIDTask);
  graph.output<float>(outputFDTask);
  graph.output<double>(outputFDTask);

  graph.executeGraph();

  for (int i = 0; i < 100; ++i) {
    graph.pushData(std::make_shared<int>(i));
    graph.pushData(std::make_shared<float>(i));
    graph.pushData(std::make_shared<double>(i));
  }
  graph.finishPushingData();

  while (auto tpl = graph.getBlockingResult()) {
    ++nbResults;
    std::visit(hh::ResultVisitor{
        [&nbResultsInt]([[maybe_unused]]std::shared_ptr<int> &val) { ++nbResultsInt; },
        [&nbResultsFloat]([[maybe_unused]]std::shared_ptr<float> &val) { ++nbResultsFloat; },
        [&nbResultsDouble]([[maybe_unused]]std::shared_ptr<double> &val) { ++nbResultsDouble; }
    }, *tpl);
  }

  ASSERT_EQ(nbResults, 4800);
  ASSERT_EQ(nbResultsInt, 1600);
  ASSERT_EQ(nbResultsFloat, 1600);
  ASSERT_EQ(nbResultsDouble, 1600);

  graph.waitForTermination();
}

std::shared_ptr<hh::Graph<1, int, int>> innerGraph() {
  auto g = std::make_shared<hh::Graph<1, int, int>>();
  auto t = std::make_shared<IntTask>();
  g->inputs(t);
  g->outputs(t);
  return g;
}

std::shared_ptr<hh::Graph<1, int, int>> wrapperGraph(const std::shared_ptr<hh::Graph<1, int, int>> &innerGraph) {
  auto g = std::make_shared<hh::Graph<1, int, int>>();
  auto t1 = std::make_shared<IntTask>(), t2 = std::make_shared<IntTask>();
  g->inputs(t1);
  g->edges(t1, innerGraph);
  g->edges(innerGraph, t2);
  g->outputs(t2);
  return g;
}

void complexGraphComposition() {
  std::shared_ptr<hh::Graph<1, int, int>>
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

  ASSERT_EQ(count, 100);
}

void testSimpleRecursiveEP() {
  size_t nbResults = 0;
  hh::Graph<3, int, float, double, int, float, double> g;

  auto insideGraph = std::make_shared<hh::Graph<3, int, float, double, int, float, double>>();
  auto innerInputInt = std::make_shared<IntFloatDoubleTask>();
  auto innerTaskFloat = std::make_shared<IntFloatDoubleTask>();
  auto innerOutput = std::make_shared<IntFloatDoubleTask>();
  auto innerSM = std::make_shared<hh::StateManager<1, int, int>>(std::make_shared<IntState>());
  auto innerGraph = std::make_shared<IntFloatDoubleGraph>();

  insideGraph->input<int>(innerInputInt);
  insideGraph->input<float>(innerTaskFloat);
  insideGraph->inputs(innerSM);
  insideGraph->inputs(innerGraph);

  insideGraph->edges(innerInputInt, innerOutput);
  insideGraph->edges(innerSM, innerOutput);
  insideGraph->edges(innerTaskFloat, innerOutput);
  insideGraph->outputs(innerOutput);
  insideGraph->outputs(innerGraph);

  auto ep = std::make_shared<IntFloatDoubleExecutionPipeline>(insideGraph, 5);

  auto outerTask1 = std::make_shared<IntFloatDoubleTask>();
  auto outerTask2 = std::make_shared<IntFloatDoubleTask>();

  g.input<int>(outerTask1);
  g.input<float>(outerTask1);
  g.edges(outerTask1, ep);
  g.edges(ep, outerTask2);
  g.outputs(outerTask2);

  g.executeGraph();

  for (int i = 0; i < 2000; ++i) {
    g.pushData(std::make_shared<int>(i));
    g.pushData(std::make_shared<float>(i));
    g.pushData(std::make_shared<double>(i));
  }

  g.finishPushingData();
  while (auto variant = g.getBlockingResult()) {
    ++nbResults;
  }

  g.waitForTermination();

  ASSERT_EQ(70000, nbResults);
}

std::shared_ptr<hh::Graph<1, int, int>> wrapperEPGraph(const std::shared_ptr<hh::Graph<1, int, int>> &innerGraph) {
  auto g = std::make_shared<hh::Graph<1, int, int>>();
  auto ep = std::make_shared<IntExecutionPipeline>(innerGraph, 4);
  auto t1 = std::make_shared<IntTask>(), t2 = std::make_shared<IntTask>();
  g->inputs(t1);
  g->edges(t1, ep);
  g->edges(ep, t2);
  g->outputs(t2);
  return g;
}

void testEPComposition() {
  std::shared_ptr<hh::Graph<1, int, int>>
      graph = nullptr,
      tempGraph = innerGraph();

  size_t count = 0;
  for (int i = 0; i < 3; ++i) { tempGraph = wrapperEPGraph(tempGraph); }
  graph = wrapperEPGraph(tempGraph);
  graph->executeGraph();

  for (int i = 0; i < 100; ++i) { graph->pushData(std::make_shared<int>(i)); }
  graph->finishPushingData();
  while (graph->getBlockingResult()) { count++; }
  graph->waitForTermination();

  ASSERT_EQ(count, (size_t) 25600);
}

std::shared_ptr<hh::Graph<3, int, float, double, int, float, double>> innerIntFloatDoubleGraph() {
  auto insideGraph = std::make_shared<hh::Graph<3, int, float, double, int, float, double>>();
  auto innerInputInt = std::make_shared<IntFloatDoubleTask>();
  auto innerTaskFloat = std::make_shared<IntFloatDoubleTask>();
  auto innerOutput = std::make_shared<IntFloatDoubleTask>();
  auto innerSM = std::make_shared<hh::StateManager<1, int, int>>(std::make_shared<IntState>());
  auto innerGraph = std::make_shared<IntFloatDoubleGraph>();

  insideGraph->input<int>(innerInputInt);
  insideGraph->input<float>(innerTaskFloat);
  insideGraph->inputs(innerSM);
  insideGraph->inputs(innerGraph);

  insideGraph->edges(innerInputInt, innerOutput);
  insideGraph->edges(innerSM, innerOutput);
  insideGraph->edges(innerTaskFloat, innerOutput);
  insideGraph->outputs(innerOutput);
  insideGraph->outputs(innerGraph);
  return insideGraph;
}

std::shared_ptr<hh::Graph<3, int, float, double, int, float, double>> wrapperIntFloatDoubleGraph(
    const std::shared_ptr<hh::Graph<3, int, float, double, int, float, double>> &innerGraph) {
  auto g = std::make_shared<hh::Graph<3, int, float, double, int, float, double>>();
  auto ep = std::make_shared<IntFloatDoubleExecutionPipeline>(innerGraph, 4);
  auto t1 = std::make_shared<IntFloatDoubleTask>(), t2 = std::make_shared<IntFloatDoubleTask>();
  g->inputs(t1);
  g->edges(t1, ep);
  g->edges(ep, t2);
  g->outputs(t2);
  return g;
}

void testComplexEPComposition() {
  std::shared_ptr<hh::Graph<3, int, float, double, int, float, double>>
      graph = nullptr,
      tempGraph = innerIntFloatDoubleGraph();

  size_t nbResults = 0;
  for (int i = 0; i < 3; ++i) { tempGraph = wrapperIntFloatDoubleGraph(tempGraph); }
  graph = wrapperIntFloatDoubleGraph(tempGraph);

  graph->executeGraph();

  for (int i = 0; i < 100; ++i) {
    graph->pushData(std::make_shared<int>(i));
    graph->pushData(std::make_shared<float>(i));
    graph->pushData(std::make_shared<double>(i));
  }
  graph->finishPushingData();
  while (graph->getBlockingResult()) { nbResults++; }
  graph->waitForTermination();
  ASSERT_EQ(nbResults, 179200);
}

void testCustomizedNodes() {
  std::vector<int> inputs(100, 0);
  std::iota(inputs.rbegin(), inputs.rend(), 0);

  {
    hh::Graph<1, int, int> g;
    auto task = std::make_shared<IntIntPriorityQueueTask>();
    std::vector<int> resVector;

    g.inputs(task);
    g.outputs(task);
    g.executeGraph();

    for (auto &i : inputs) { g.pushData(std::make_shared<int>(i)); }
    g.finishPushingData();

    while (auto res = g.getBlockingResult()) {
      resVector.push_back(*(std::get<std::shared_ptr<int >>(*res)));
    }

    g.waitForTermination();

    ASSERT_TRUE(inputs != resVector);
  }
  {
    hh::Graph<1, int, int> g;
    auto state = std::make_shared<IntState>();
    std::vector<int> resVector;

    auto sm = std::make_shared<PriorityQueueStateManager<1, int, int >>(state, "Priority State Manager", false);

    g.inputs(sm);
    g.outputs(sm);
    g.executeGraph();

    for (auto &i : inputs) { g.pushData(std::make_shared<int>(i)); }
    g.finishPushingData();

    while (auto res = g.getBlockingResult()) {
      resVector.push_back(*(std::get<std::shared_ptr<int >>(*res)));
    }
    g.waitForTermination();

    ASSERT_TRUE(inputs != resVector);
  }
}

#endif //HEDGEHOG_TEST_COMPLEX_GRAPH_H_
