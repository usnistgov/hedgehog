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
#include "../../data_structures/compile_time_analysis/static_tests/test_critical_path.h"

constexpr auto constructGraphCriticalPath() {
  hh_cx::Node<TaskIntInt> node0("Task0");
  hh_cx::Node<TaskIntInt> node1("Task1");
  hh_cx::Node<TaskIntInt> node2("Task2");
  hh_cx::Node<TaskIntInt> node3("Task3");
  hh_cx::Node<TaskIntInt> node4("Task4");
  hh_cx::Node<TaskIntInt> node5("Task5");
  hh_cx::Node<TaskIntInt> node6("Task6");
  hh_cx::Node<TaskIntInt> node7("Task7");

  hh_cx::Graph<GraphIntInt> g("Graph of matrix multiplication");

  g.inputs(node0);
  g.inputs(node1);
  g.inputs(node2);
  g.edges(node0, node3);
  g.edges(node1, node3);
  g.edges(node2, node4);
  g.edges(node3, node5);
  g.edges(node5, node4);
  g.edges(node4, node6);
  g.edges(node6, node4);
  g.edges(node6, node7);
  g.outputs(node7);

  hh_cx::PropertyMap<double> propertyMap;

  propertyMap.insert("Task0", 1);
  propertyMap.insert("Task1", 1);
  propertyMap.insert("Task2", 1);
  propertyMap.insert("Task3", 1);
  propertyMap.insert("Task4", 1);
  propertyMap.insert("Task5", 1);
  propertyMap.insert("Task6", 1);
  propertyMap.insert("Task7", 20);

  auto criticalPath = TestCriticalPath<GraphIntInt>(propertyMap);
  g.addTest(&criticalPath);

  return g;
}

void testCriticalPath() {
  constexpr auto defroster = hh_cx::createDefroster<&constructGraphCriticalPath>();

  ASSERT_FALSE(defroster.isValid());
  ASSERT_TRUE(
      defroster.report().find("Task0 -> Task3 -> Task5 -> Task4 -> Task6 -> Task7") != std::string::npos
  );
}

#endif //HH_ENABLE_HH_CX