#include "hedgehog/hedgehog.h"
using namespace std;

class ItoF : public AbstractTask<float, int, double, char> {
 public:
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {}
  void execute([[maybe_unused]]std::shared_ptr<double> ptr) override {}
  void execute([[maybe_unused]]std::shared_ptr<char> ptr) override {}
  //std::cout << "Received int" << std::endl;
//    addResult(std::make_shared<float>(*ptr));
//  }
};

int main() {
  std::chrono::time_point<std::chrono::high_resolution_clock>
      start,
      finish;
  Graph<float, int, double, char> g("GraphOutput");
  auto queue = std::make_shared<ItoF>();
  g.input(queue);
  g.output(queue);

  g.executeGraph();

  g.createDotFile("output.dot");
  start = std::chrono::high_resolution_clock::now();
  for (uint64_t i = 0; i < 100; ++i) {
    g.pushData(std::make_shared<int>(i));
  }

  g.finishPushingData();

  g.waitForTermination();
  finish = std::chrono::high_resolution_clock::now();

  std::cerr << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << std::endl;
}