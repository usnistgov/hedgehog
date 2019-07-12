#include "ep/inner_ep.h"
#include "states/test_state.h"
#include "tasks/input_task.h"
#include "tasks/input_task_2.h"
#include "tasks/output_task.h"
#include "tasks/output_task_2.h"
#include "tasks/task_bda.h"
#include "tasks/task_ic.h"

void testBigExample() {
  for(int r = 0; r < 10; ++r) {
    Graph<A, int, float> graph("Outer Graph");

    auto innerGraph = std::make_shared<Graph<int,
                                             double, A >>("Inner Graph");

    auto inputTask = std::make_shared<InputTask>();
    auto inputTask2 = std::make_shared<InputTask2>();
    auto outputTask = std::make_shared<OutputTask>();
    auto outputTask2 = std::make_shared<OutputTask2>();
    auto taskBDA = std::make_shared<TaskBDA>();
    auto taskBDA2 = std::make_shared<TaskBDA>();
    auto taskIC = std::make_shared<TaskIC>();
    auto taskIC2 = std::make_shared<TaskIC>();
    auto testState = std::make_shared<TestState>();
    auto stateManager = std::make_shared<DefaultStateManager<C, B, A>>("My State Manager", testState);

    int count = 0;

    innerGraph->input(taskBDA);
    innerGraph->input(taskBDA2);
    innerGraph->addEdge(taskBDA, taskIC);
    innerGraph->addEdge(taskBDA2, taskIC);
    innerGraph->output(taskIC);

    innerGraph->addEdge(taskBDA, stateManager);
    innerGraph->addEdge(taskBDA2, stateManager);
    innerGraph->addEdge(stateManager, taskIC2);
    innerGraph->output(taskIC2);

    std::vector<int> v = {0, 1, 2, 3, 4, 5};
    auto execPipeline = std::make_shared<InnerEP>(innerGraph, 6, v);

    graph.input(inputTask);
    graph.input(inputTask2);

    graph.addEdge(inputTask, execPipeline);
    graph.addEdge(inputTask2, execPipeline);

    graph.addEdge(execPipeline, outputTask);
    graph.addEdge(execPipeline, outputTask2);

    graph.output(outputTask);
    graph.output(outputTask2);

    graph.executeGraph();

    for (int i = 0; i < 100; ++i) {
      graph.pushData(std::make_shared<int>(0));
      graph.pushData(std::make_shared<float>(0));
    }

    graph.finishPushingData();

    while (std::shared_ptr<A> result = graph.getBlockingResult()) { count++; }

    ASSERT_EQ(count, 19200);
    graph.waitForTermination();
  }
}


