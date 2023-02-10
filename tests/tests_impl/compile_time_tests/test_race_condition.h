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

#include "../../data_structures/compile_time_analysis/tasks/task_int_int.h"
#include "../../data_structures/compile_time_analysis/tasks/task_const_int_const_int.h"
#include "../../data_structures/compile_time_analysis/graphs/graph_int_int.h"
#include "../../data_structures/compile_time_analysis/graphs/graph_const_int_const_int.h"

constexpr auto constructGraphDataRacesWithAllEdgesTreatedAsRW() {
  hh_cx::Node<TaskIntInt> node1("Task1");
  hh_cx::Node<TaskIntInt> node2("Task2");
  hh_cx::Node<TaskIntInt> node3("Task3");
  hh_cx::Node<TaskIntInt> node4("Task4");
  hh_cx::Node<TaskIntInt> node5("Task5");
  hh_cx::Node<TaskIntInt> node6("Task6");
  hh_cx::Node<TaskIntInt> node7("Task7");
  hh_cx::Graph<GraphIntInt> g("Graph with all edges treated as RW");

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

  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&constTest);

  return g;
}

void testDataRacesWithAllEdgesTreatedAsRW() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphDataRacesWithAllEdgesTreatedAsRW>();

  ASSERT_FALSE(defroster.isValid());
  ASSERT_TRUE(defroster.report().find("Task2 -> Task3 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task2 -> Task5 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task3 -> Task4 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task3 -> Task5 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task4 -> Task1 (int)") != std::string::npos);
  ASSERT_TRUE(defroster.report().find("Task4 -> Task7 (int)") != std::string::npos);
}

constexpr auto constructGraphDataRacesWithAllConstEdges() {
  hh_cx::Node<TaskConstIntConstInt> node1("Task1");
  hh_cx::Node<TaskConstIntConstInt> node2("Task2");
  hh_cx::Node<TaskConstIntConstInt> node3("Task3");
  hh_cx::Node<TaskConstIntConstInt> node4("Task4");
  hh_cx::Node<TaskConstIntConstInt> node5("Task5");
  hh_cx::Node<TaskConstIntConstInt> node6("Task6");
  hh_cx::Node<TaskConstIntConstInt> node7("Task7");
  hh_cx::Graph<GraphConstIntConstInt> g("Graph with all edges treated as RW");

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

  auto constTest = hh_cx::DataRaceTest < GraphConstIntConstInt > {};
  g.addTest(&constTest);

  return g;
}

void testDataRacesWithAllConstEdges() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphDataRacesWithAllConstEdges>();
  ASSERT_TRUE(defroster.isValid());
}

constexpr auto constructGraphDataRacesWithSomeEdgesTreatedAsRO() {
  hh_cx::Node<TaskIntInt, int> node1("Task1");
  hh_cx::Node<TaskIntInt> node2("Task2");
  hh_cx::Node<TaskIntInt, int> node3("Task3");
  hh_cx::Node<TaskIntInt, int> node4("Task4");
  hh_cx::Node<TaskIntInt, int> node5("Task5");
  hh_cx::Node<TaskIntInt> node6("Task6");
  hh_cx::Node<TaskIntInt, int> node7("Task7");

  hh_cx::Graph<GraphIntInt> g("Graph with some edges treated as RO");

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

  auto constTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&constTest);

  return g;
}

void testDataRacesWithSomeEdgesTreatedAsRO() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphDataRacesWithSomeEdgesTreatedAsRO>();
  ASSERT_TRUE(defroster.isValid());
}

#endif //HH_ENABLE_HH_CX