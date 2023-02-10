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
#include "../../data_structures/compile_time_analysis/graphs/graph_int_int.h"

constexpr auto constructGraphErrorMessage() {
  hh_cx::Node<TaskIntInt> nodeInit("TaskInit");
  hh_cx::Node<TaskIntInt> node3("Task3");
  hh_cx::Node<TaskIntInt> node4("Task4");
  hh_cx::Node<TaskIntInt> node5("Task5");
  hh_cx::Node<TaskIntInt> node6("Task6");
  hh_cx::Node<TaskIntInt> node7("Task7");
  hh_cx::Node<TaskIntInt> node8("Task8");
  hh_cx::Node<TaskIntInt> node9("Task9");
  hh_cx::Node<TaskIntInt> node10("Task10");
  hh_cx::Node<TaskIntInt> node11("Task11");
  hh_cx::Node<TaskIntInt> node12("Task12");
  hh_cx::Node<TaskIntInt> node13("Task13");
  hh_cx::Node<TaskIntInt> node14("Task14");
  hh_cx::Node<TaskIntInt> node15("Task15");
  hh_cx::Node<TaskIntInt> nodeFinal("TaskFinal");

  hh_cx::Graph<GraphIntInt> g("Graph");

  g.inputs(nodeInit);
  g.outputs(nodeFinal);
  g.edges(nodeFinal, nodeInit);

  g.edges(nodeInit, node3);
  g.edges(node3, node4);
  g.edges(node4, node5);
  g.edges(node5, node6);
  g.edges(node6, node7);
  g.edges(node7, node8);
  g.edges(node8, node9);
  g.edges(node9, node10);
  g.edges(node10, node11);
  g.edges(node11, node12);
  g.edges(node12, node13);
  g.edges(node13, node14);
  g.edges(node14, node15);
  g.edges(node15, nodeFinal);

  auto cycleTest = hh_cx::CycleTest < GraphIntInt > {};
  auto dataRaceTest = hh_cx::DataRaceTest < GraphIntInt > {};
  g.addTest(&cycleTest);
  g.addTest(&dataRaceTest);

  return g;
}

void testErrorMessage() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphErrorMessage>();
  ASSERT_FALSE(defroster.isValid());
}

#endif //HH_ENABLE_HH_CX