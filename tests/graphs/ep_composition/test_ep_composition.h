#include "tasks/ii_task.h"
#include "ep/iiep.h"

std::shared_ptr<IITask> task() { return std::make_shared<IITask>(); }

std::shared_ptr<Graph<int, int>> innerGraph() {
  auto g = std::make_shared<Graph<int, int>>();
  auto t = task();
  g->input(t);
  g->output(t);
  return g;
}

std::shared_ptr<Graph<int, int>> wrapperGraph(const std::shared_ptr<Graph<int, int>> &innerGraph) {
  auto g = std::make_shared<Graph<int, int>>();
  auto ep = std::make_shared<IIEP>(innerGraph);
  auto t1 = task(), t2 = task();

  g->input(t1);
  g->addEdge(t1, ep);
  g->addEdge(ep, t2);
  g->output(t2);

  return g;
}

void testEPComposition() {
  std::shared_ptr<Graph<int, int>>
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
}




