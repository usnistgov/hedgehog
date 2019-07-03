#include "tasks/ito_f.h"

void testSmallGraph() {
  Graph<float, int, double, char> g("GraphOutput");
  auto queue = std::make_shared<IToF>();
  g.input(queue);
  g.output(queue);

  g.executeGraph();

  for (uint64_t i = 0; i < 100; ++i) { g.pushData(std::make_shared<int>(i)); }

  g.finishPushingData();
  g.waitForTermination();
}