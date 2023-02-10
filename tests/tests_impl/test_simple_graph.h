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

#ifndef HEDGEHOG_TEST_SIMPLE_GRAPH_H_
#define HEDGEHOG_TEST_SIMPLE_GRAPH_H_

#include <gtest/gtest.h>
#include "../data_structures/tasks/int_task.h"

void graphSimpleExecution(){
  hh::Graph<1, int, int> g;
  auto intTask = std::make_shared<IntTask>();
  g.inputs(intTask);
  g.outputs(intTask);
  g.executeGraph();
  g.finishPushingData();
  g.waitForTermination();
}

void graphSimpleDataTransfer(){
  size_t count = 0;
  hh::Graph<1, int, int> g;
  auto intTask = std::make_shared<IntTask>();
  g.inputs(intTask);
  g.outputs(intTask);
  g.executeGraph();
  for(int i = 0; i < 100; ++i){ g.pushData(std::make_shared<int>(i)); }
  g.finishPushingData();

  while(auto res = g.getBlockingResult()){ ++count; }

  g.waitForTermination();

  EXPECT_EQ(count, 100);
}

void graphSimpleDataTransferTaskGroup(){
  int count = 0;
  hh::Graph<1, int, int> g;
  auto intTaskMultiThread = std::make_shared<IntTask>(3);
  g.inputs(intTaskMultiThread);
  g.outputs(intTaskMultiThread);
  g.executeGraph();
  for(int i = 0; i < 100; ++i){ g.pushData(std::make_shared<int>(i)); }
  g.finishPushingData();

  while(auto res = g.getBlockingResult()){ ++count; }

  g.waitForTermination();

  EXPECT_EQ(count, 100);
}

void graphSimpleMultiGroups(){
  int count = 0;
  hh::Graph<1, int, int> g;
  auto intTaskMultiThread1 = std::make_shared<IntTask>(1);
  auto intTaskMultiThread2 = std::make_shared<IntTask>(2);
  auto intTaskMultiThread3 = std::make_shared<IntTask>(3);
  g.inputs(intTaskMultiThread1);
  g.edges(intTaskMultiThread1, intTaskMultiThread2);
  g.edges(intTaskMultiThread1, intTaskMultiThread3);
  g.outputs(intTaskMultiThread2);
  g.outputs(intTaskMultiThread3);
  g.executeGraph();
  for(int i = 0; i < 100; ++i){ g.pushData(std::make_shared<int>(i)); }
  g.finishPushingData();

  while(auto res = g.getBlockingResult()){ ++count; }

  g.waitForTermination();

  EXPECT_EQ(count, 200);
}

#endif //HEDGEHOG_TEST_SIMPLE_GRAPH_H_
