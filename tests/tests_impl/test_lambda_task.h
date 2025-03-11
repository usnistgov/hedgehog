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
#include <memory>
#include "../data_structures/tasks/test_specialized_lambda_task.h"

void lambdaTaskSingleInput(){
  size_t count = 0;
  hh::Graph<1, int, int> g;
  auto intTask = std::make_shared<hh::LambdaTask<1, int, int>>();
  intTask->setLambda<int>([](std::shared_ptr<int> data, auto self) {
      self.addResult(data);
  });
  g.inputs(intTask);
  g.outputs(intTask);
  g.executeGraph();
  for(int i = 0; i < 100; ++i){ g.pushData(std::make_shared<int>(i)); }
  g.finishPushingData();

  while(auto res = g.getBlockingResult()){ ++count; }

  g.waitForTermination();

  EXPECT_EQ(count, 100);
}

void lambdaTaskMultipleInputs(){
  enum InputType { INT, DOUBLE };
  int countInt = 0, countDouble = 0;
  hh::Graph<2, int, double, InputType> g;
  auto task = std::make_shared<hh::LambdaTask<2, int, double, InputType>>();

  task->setLambda<int>([](std::shared_ptr<int>, auto self) {
      self.addResult(std::make_shared<InputType>(INT));
  });
  task->setLambda<double>([](std::shared_ptr<double>, auto self) {
      self.addResult(std::make_shared<InputType>(DOUBLE));
  });

  g.inputs(task);
  g.outputs(task);
  g.executeGraph();
  for(int i = 0; i < 10; ++i){ g.pushData(std::make_shared<int>(i)); }
  for(int i = 0; i < 20; ++i){ g.pushData(std::make_shared<double>(i)); }
  g.finishPushingData();

  while(auto res = g.getBlockingResult()){ 
      switch (*std::get<std::shared_ptr<InputType>>(*res)) {
      case INT: countInt++; break;
      case DOUBLE: countDouble++; break;
      }
  }

  g.waitForTermination();

  EXPECT_EQ(countInt, 10);
  EXPECT_EQ(countDouble, 20);
}

void lambdaTaskSpecializedTask(){
  size_t count = 0;
  hh::Graph<1, int, int> g;
  auto task = std::make_shared<TestSpecializedLambdaTask<1, int, int>>(4);
  task->setLambda<int>([](std::shared_ptr<int> data, auto self) {
      self.addResult(std::make_shared<int>(*data * self->number()));
  });
  g.inputs(task);
  g.outputs(task);
  g.executeGraph();
  for(int i = 0; i < 4; ++i){ g.pushData(std::make_shared<int>(i)); }
  g.finishPushingData();

  while(auto res = g.getBlockingResult()){ 
      count += *std::get<std::shared_ptr<int>>(*res); 
  }

  g.waitForTermination();

  EXPECT_EQ(count, (0*4 + 1*4 + 2*4 + 3*4));
}

void lambdaTaskCaptureContext(){
  size_t count = 0;
  hh::Graph<1, int, int> g;
  auto task = std::make_shared<hh::LambdaTask<1, int, int>>();
  task->setLambda<int>([&count](std::shared_ptr<int> data, auto self) {
      ++count;
      self.addResult(data);
  });
  g.inputs(task);
  g.outputs(task);
  g.executeGraph();
  for(int i = 0; i < 100; ++i){ g.pushData(std::make_shared<int>(i)); }
  g.finishPushingData();
  g.waitForTermination();

  EXPECT_EQ(count, 100);
}
