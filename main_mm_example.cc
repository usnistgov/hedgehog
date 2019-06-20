#include "hedgehog/hedgehog.h"

class floatAllocator : public AbstractAllocator<float> {
 private:
  void initialize() override {}
 private:
  float *allocate([[maybe_unused]]size_t size) override {
    return new float();
  }

  void deallocate(float *data) override {
    delete data;
  }
};

class floatReleaseRule : public AbstractReleaseRule<float> {
 public:
  size_t count = 1;
  void used() override {
    --count;
  }

  bool canRelease() override {
    return count == 0;
  }
};

class ItoF : public AbstractManagedMemoryReceiverTask<float, int, float> {
 public:
  ItoF() :
      AbstractManagedMemoryReceiverTask(
          std::make_shared<StaticMemoryManager<float>>(3, 1, std::make_unique<floatAllocator>()),
          "IToF",
          10) {
  }

  std::shared_ptr<AbstractTask<ManagedMemory<float>, int, float>> copy() override {
    return std::make_shared<ItoF>(this);
  }

  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    auto mem = this->getMemory(std::make_unique<floatReleaseRule>(), 1);
    addResult(mem);
  }
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
    auto mem = this->getMemory(std::make_unique<floatReleaseRule>(), 1);
    addResult(mem);
  }
};

class MFtoF : public AbstractTask<float, ManagedMemory<float>> {
 public:
  MFtoF() : AbstractTask("MFtoF") {}

  void execute([[maybe_unused]]std::shared_ptr<ManagedMemory<float>> ptr) override {
    ptr->release();
    addResult(std::make_shared<float>(2));
  }
};

int main() {
  Graph<float, int, float> g("GraphOutput");
  auto task = std::make_shared<ItoF>();
  auto task2 = std::make_shared<MFtoF>();
  g.input(task);
  g.addEdge(task, task2);
  g.output(task2);
  g.executeGraph();
  g.createDotFile("output.dot");

  for (int i = 0; i < 100; ++i) {
    g.pushData(std::make_shared<int>(i));
    g.pushData(std::make_shared<float>(i));
  }

  g.finishPushingData();

  while (std::shared_ptr<float> result = g.getBlockingResult()) {
    std::cout << "Processing result: " << *result << std::endl;
  }

  g.createDotFile("outputBeforeWait.dot", ColorScheme::EXECUTION);
  g.waitForTermination();
  g.createDotFile("outputAfterWait.dot", ColorScheme::EXECUTION);

}