////
//// Created by anb22 on 6/20/19.
////

#include "hedgehog/hedgehog.h"

class IToF : public AbstractTask<float, int, double, char> {
 public:
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    std::cout << "Executing int" << std::endl;
  }
  void execute([[maybe_unused]]std::shared_ptr<double> ptr) override {}
  void execute([[maybe_unused]]std::shared_ptr<char> ptr) override {}
};


void testSmallGraph() {
  for(int r = 0; r < 1; ++r) {
    Graph<float, int, double, char> g("GraphOutput");
    auto t = std::make_shared<IToF>();
    size_t count = 0;

    g.input(t);
    g.output(t);

    g.executeGraph();

    for (uint64_t i = 0; i < 100; ++i) { g.pushData(std::make_shared<int>(i)); }

    g.finishPushingData();

    while ((g.getBlockingResult())) { ++count; }

    g.waitForTermination();
  }
}

int main(){
  testSmallGraph();
  return 0;
}