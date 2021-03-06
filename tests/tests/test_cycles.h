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


#ifndef HEDGEHOG_TESTS_TEST_CYCLES_H
#define HEDGEHOG_TESTS_TEST_CYCLES_H

#include "../../hedgehog/hedgehog.h"
#include "../data_structures/tasks/float_to_int_cycle.h"
#include "../data_structures/tasks/int_double_char_to_float.h"
#include "../data_structures/states/state_int_float_to_int.h"

void testCycles() {
  int count = 0;
  auto graph = std::make_shared<hh::Graph<int, int, double, char>>();
  auto input = std::make_shared<IntDoubleCharToFloat>(5);
  auto cycleTask = std::make_shared<FloatToIntCycle>(3);
  auto state = std::make_shared<StateIntFloatToInt>();
  auto stateManager = std::make_shared<hh::StateManager<int, int, float>>(state);

  graph->input(input);
  graph->addEdge(input, cycleTask);
  graph->addEdge(cycleTask, input);
  graph->addEdge(input, stateManager);
  graph->output(stateManager);

  graph->executeGraph();

  for (int i = 0; i < 100; i++) {
    graph->pushData(std::make_shared<int>(i));
    graph->pushData(std::make_shared<double>(i));
    graph->pushData(std::make_shared<char>(i));
  }

  graph->finishPushingData();

  while (graph->getBlockingResult()) { count++; }
  graph->waitForTermination();

  ASSERT_EQ(count, 1200);
}

#endif //HEDGEHOG_TESTS_TEST_CYCLES_H
