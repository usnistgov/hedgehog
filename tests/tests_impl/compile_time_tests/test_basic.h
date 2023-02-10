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

#ifdef HH_ENABLE_HH_CX
#include "../../data_structures/compile_time_analysis/graphs/graph_int_int.h"
#include "../../data_structures/compile_time_analysis/tasks/task_int_int.h"
#include "../../data_structures/compile_time_analysis/graphs/graph_const_int_const_int.h"
#include "../../data_structures/compile_time_analysis/tasks/task_const_int_const_int.h"
#include "../../data_structures/compile_time_analysis/graphs/inside_graph.h"

constexpr auto constructGraphTarjanNoCycle() {
  hh_cx::Node<TaskIntInt> node("Task1");
  hh_cx::Node<TaskIntInt> node2("Task2");
  hh_cx::Graph<GraphIntInt> g("Graph without cycle");
  g.inputs(node);
  g.edges(node, node2);
  g.outputs(node2);
  auto cycleTest = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&cycleTest);
  g.addTest(&constTest);
  return g;
}

void testTarjanNoCycle() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphTarjanNoCycle>();
  ASSERT_TRUE(defroster.isValid());
}

constexpr auto constructGraphTarjanSimpleCycle() {
  hh_cx::Node<TaskIntInt> node("Task1");
  hh_cx::Node<TaskIntInt> node2("Task2");
  hh_cx::Graph<GraphIntInt> g("Graph with hedgehog simple cycle between input and output");

  g.inputs(node);
  g.edges(node, node2);
  g.edges(node2, node);
  g.outputs(node2);

  auto cycle = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&cycle);
  g.addTest(&constTest);

  return g;
}

void testTarjanSimpleCycle() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphTarjanSimpleCycle>();
  ASSERT_FALSE(defroster.isValid());
  ASSERT_TRUE(defroster.report().find("Task1 -> Task2 -> Task1") != std::string::npos);
}

constexpr auto constructGraphTarjanCycle3Nodes() {
  hh_cx::Node<TaskIntInt> node("Task1");
  hh_cx::Node<TaskIntInt> node2("Task2");
  hh_cx::Node<TaskIntInt> node3("Task3");
  hh_cx::Graph<GraphIntInt> g("Graph with hedgehog cycle in the middle");

  g.inputs(node);
  g.edges(node, node2);
  g.edges(node2, node3);
  g.edges(node3, node2);
  g.outputs(node3);

  auto cycle = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&cycle);
  g.addTest(&constTest);

  return g;
}

void testTarjanSimpleCycle3Nodes() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphTarjanCycle3Nodes>();
  ASSERT_FALSE(defroster.isValid());
  ASSERT_TRUE(defroster.report().find("Task2 -> Task3 -> Task2") != std::string::npos);
}

constexpr auto constructGraphTarjanComplexCycle() {
  hh_cx::Node<TaskIntInt> node1("Task1");
  hh_cx::Node<TaskIntInt> node2("Task2");
  hh_cx::Node<TaskIntInt> node3("Task3");
  hh_cx::Node<TaskIntInt> node4("Task4");
  hh_cx::Node<TaskIntInt> node5("Task5");
  hh_cx::Node<TaskIntInt> node6("Task6");
  hh_cx::Node<TaskIntInt> node7("Task7");

  hh_cx::Graph<GraphIntInt> g("Graph with multiple cycles, some inside");

  g.inputs(node1);
  g.edges(node1, node2);
  g.edges(node2, node3);
  g.edges(node3, node4);
  g.edges(node4, node7);
  g.edges(node4, node1);
  g.edges(node3, node5);
  g.edges(node5, node6);
  g.edges(node6, node2);
  g.edges(node2, node5);
  g.outputs(node7);

  auto cycle = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&cycle);
  g.addTest(&constTest);

  return g;
}

void testTarjanComplexCycles() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphTarjanComplexCycle>();

  ASSERT_FALSE(defroster.isValid());
  ASSERT_TRUE(defroster.report().find("Task1 -> Task2 -> Task3 -> Task4 -> Task1") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task2 -> Task3 -> Task5 -> Task6 -> Task2") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task2 -> Task5 -> Task6 -> Task2") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task2 -> Task3 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task2 -> Task5 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task3 -> Task4 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task3 -> Task5 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task4 -> Task1 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task4 -> Task7 (int)") != std::string::npos);
}

constexpr auto constructGraphTarjanComplexCycleConst() {
  hh_cx::Node<TaskConstIntConstInt> node1("Task1");
  hh_cx::Node<TaskConstIntConstInt> node2("Task2");
  hh_cx::Node<TaskConstIntConstInt> node3("Task3");
  hh_cx::Node<TaskConstIntConstInt> node4("Task4");
  hh_cx::Node<TaskConstIntConstInt> node5("Task5");
  hh_cx::Node<TaskConstIntConstInt> node6("Task6");
  hh_cx::Node<TaskConstIntConstInt> node7("Task7");

  hh_cx::Graph<GraphConstIntConstInt>
      g("Graph with const inputs, with multiple cycles, some inside");

  g.inputs(node1);
  g.edges(node1, node2);
  g.edges(node2, node3);
  g.edges(node3, node4);
  g.edges(node4, node7);
  g.edges(node4, node1);
  g.edges(node3, node5);
  g.edges(node5, node6);
  g.edges(node6, node2);
  g.edges(node2, node5);
  g.outputs(node7);

  auto cycle = hh_cx::CycleTest < GraphConstIntConstInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphConstIntConstInt > {};
  g.addTest(&cycle);
  g.addTest(&constTest);
  return g;
}

void testTarjanComplexCyclesConst() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphTarjanComplexCycleConst>();
  ASSERT_FALSE(defroster.isValid());
  ASSERT_TRUE(defroster.report().find("Task1 -> Task2 -> Task3 -> Task4 -> Task1") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task2 -> Task3 -> Task5 -> Task6 -> Task2") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task2 -> Task5 -> Task6 -> Task2") != std::string::npos);
}

constexpr auto constructGraphTarjanSameNodeCycle() {
  hh_cx::Node<TaskIntInt> node("Task1");
  hh_cx::Graph<GraphIntInt> g("Graph with hedgehog cycle in hedgehog single node");
  g.inputs(node);
  g.edges(node, node);
  g.outputs(node);
  auto cycle = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&cycle);
  g.addTest(&constTest);
  return g;
}

void testTarjanSameNodeCycle() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphTarjanSameNodeCycle>();
  ASSERT_FALSE(defroster.isValid());
  ASSERT_TRUE(defroster.report().find("Task1 -> Task1") != std::string::npos);
}

constexpr auto constructGraphCycleMultiInputs() {
  hh_cx::Node<TaskIntInt> node0("Task0");
  hh_cx::Node<TaskIntInt> node1("Task1");
  hh_cx::Node<TaskIntInt> node2("Task2");
  hh_cx::Node<TaskIntInt> node3("Task3");

  hh_cx::Graph<GraphIntInt> g("Graph with multiple inputs and cycles");

  g.inputs(node0);
  g.inputs(node1);
  g.edges(node0, node2);
  g.edges(node1, node2);
  g.edges(node2, node3);
  g.edges(node3, node2);
  g.edges(node3, node1);
  g.outputs(node3);

  auto cycle = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&cycle);
  g.addTest(&constTest);

  return g;
}

void testCycleMultiInputs() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphCycleMultiInputs>();
  ASSERT_FALSE(defroster.isValid());
  ASSERT_TRUE(defroster.report().find("Task1 -> Task2 -> Task3 -> Task1") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task2 -> Task3 -> Task2") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task3 -> Task1 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task3 -> Task2 (int)") != std::string::npos);
}

constexpr auto constructGraphCompositionAsGraph() {
  auto insideGraph = hh_cx::Graph<InsideGraph>("Inside Graph");
  auto insideNode = hh_cx::Node<TaskIntInt>("Output Node");

  auto graph = hh_cx::Graph<GraphIntInt>("Outside Grpah");

  graph.inputs(insideGraph);
  graph.edges(insideGraph, insideNode);
  graph.outputs(insideNode);

  auto cycle = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  graph.addTest(&cycle);
  graph.addTest(&constTest);

  return graph;
}

void testCompositionAsStaticGraph() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphCompositionAsGraph>();
  ASSERT_TRUE(defroster.isValid());
  if constexpr (defroster.isValid()) {
    auto dynGraph = std::make_shared<InsideGraph>("DynInsideGRaph");
    auto dynNode = std::make_shared<TaskIntInt>("DynInsideNode");

    auto graph = defroster.map(
        "Inside Graph", dynGraph,
        "Output Node", dynNode
    );

    graph->executeGraph();
    for (int i = 0; i < 10; ++i) { graph->pushData(std::make_shared<int>(i)); }
    graph->finishPushingData();

    uint8_t numberReceived = 0;
    while (graph->getBlockingResult()) { numberReceived++; }
    ASSERT_EQ((int) numberReceived, 10);

    graph->waitForTermination();
  }
}

constexpr auto constructGraphCompositionAsNode() {
  auto insideGraph = hh_cx::Node<InsideGraph>("Inside Graph");
  auto insideNode = hh_cx::Node<TaskIntInt>("Output Node");

  auto graph = hh_cx::Graph<GraphIntInt>("Outside Grpah");

  graph.inputs(insideGraph);
  graph.edges(insideGraph, insideNode);
  graph.outputs(insideNode);

  auto cycle = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  graph.addTest(&cycle);
  graph.addTest(&constTest);

  return graph;
}

void testCompositionAsStaticNode() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphCompositionAsNode>();

  if constexpr (defroster.isValid()) {
    auto dynGraph = std::make_shared<InsideGraph>("DynInsideGRaph");
    auto dynNode = std::make_shared<TaskIntInt>("DynInsideNode");

    auto graph = defroster.map(
        "Inside Graph", dynGraph,
        "Output Node", dynNode
    );

    graph->executeGraph();

    for (int i = 0; i < 10; ++i) { graph->pushData(std::make_shared<int>(i)); }
    graph->finishPushingData();

    int numberReceived = 0;
    while (auto val = graph->getBlockingResult()) { numberReceived++; }
    ASSERT_EQ((int) numberReceived, 10);

    graph->waitForTermination();
  }
}

constexpr auto constructGraphSameNodeType() {
  hh_cx::Node<TaskIntInt> node("StaticTask1");
  hh_cx::Node<TaskIntInt> node2("StaticTask2");

  hh_cx::Graph<GraphIntInt> g("Graph without cycle");

  g.inputs(node);
  g.edges(node, node2);
  g.outputs(node2);

  auto cycle = hh_cx::CycleTest < GraphIntInt > {};
  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&cycle);
  g.addTest(&constTest);
  return g;
}

void testSameNodeType() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphSameNodeType>();

  ASSERT_TRUE(defroster.isValid());

  if constexpr (defroster.isValid()) {
    auto dynNode1 = std::make_shared<TaskIntInt>("t1");
    auto dynNode2 = std::make_shared<TaskIntInt>("t2");

    auto graph = defroster.map(
        "StaticTask1", dynNode1,
        "StaticTask2", dynNode2
    );

    graph->executeGraph();

    for (int i = 0; i < 10; ++i) { graph->pushData(std::make_shared<int>(i)); }
    graph->finishPushingData();
    uint8_t numberReceived = 0;
    while (graph->getBlockingResult()) { numberReceived++; }
    ASSERT_EQ((int) numberReceived, 10);
    graph->waitForTermination();
  }
}

#endif //HH_ENABLE_HH_CX
