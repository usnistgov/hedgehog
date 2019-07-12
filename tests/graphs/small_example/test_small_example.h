#include "tasks/ito_f.h"

void testSmallGraph() {
  for(int r = 0; r < 100; ++r) {
    Graph<float, int, double, char> g("GraphOutput");
    auto t = std::make_shared<IToF>();
    size_t count = 0;

    g.input(t);
    g.output(t);

    g.executeGraph();

    for (uint64_t i = 0; i < 100; ++i) { g.pushData(std::make_shared<int>(i)); }

    g.finishPushingData();

    while ((g.getBlockingResult())) { ++count; }

    ASSERT_EQ(count, 0);
    g.waitForTermination();
  }
}