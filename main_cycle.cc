//
// Created by anb22 on 6/11/19.
//
//
#include "hedgehog/hedgehog.h"

class MyState : public AbstractState<float, float> {
 public:

  MyState() = default;
  void execute(std::shared_ptr<float> ptr) override {
    this->push(ptr);
  }
};

class MyTask : public AbstractTask<float, int, double, float> {
 public:
  MyTask(const std::string_view &name, size_t numberThreads) : AbstractTask(name, numberThreads) {}

  void execute(std::shared_ptr<int> ptr) override {
//    std::cout << "Received int data" << std::endl;
    addResult(std::make_shared<float>(*ptr));
  }

  void execute(std::shared_ptr<double> ptr) override {
//    std::cout << "Received double data" << std::endl;
    addResult(std::make_shared<float>(*ptr));
  }

  void execute(std::shared_ptr<float> ptr) override {
//    std::cout << "Received float data" << std::endl;
    addResult(std::make_shared<float>(*ptr));
  }

  std::shared_ptr<AbstractTask<float, int, double, float>> copy() override {
    return std::make_shared<MyTask>(this->name(), this->numberThreads());
  }
};

class MyTask2 : public AbstractTask<int, float> {
 private:
  int count_ = 0;

 public:
  MyTask2(const std::string_view &name, size_t numberThreads) : AbstractTask(name, numberThreads), count_(0) {}

  void execute(std::shared_ptr<float> ptr) override {

    if (count_ != 30) {
      addResult(std::make_shared<int>(*ptr));

      count_++;
      std::cout << "Received double data -- " << count_ << std::endl;
    }
  }

  std::shared_ptr<AbstractTask<int, float>> copy() override {
    return std::make_shared<MyTask2>(this->name(), this->numberThreads());
  }

  bool canTerminate() override {
    return count_ == 30;
  }

};

int main() {
  std::shared_ptr<MyTask> myTask1 = std::make_shared<MyTask>("myTask1", 10);

  auto myTask2 = std::make_shared<MyTask2>("myTask2", 3);

  auto myGraph = std::make_shared<Graph<float, int, double, float>>();

  std::shared_ptr<MyState> myState = std::make_shared<MyState>();

  std::shared_ptr<DefaultStateManager<float, float>>
      stateManager = std::make_shared<DefaultStateManager<float, float>>(myState);

  myGraph->input(myTask1);

  myGraph->addEdge(myTask1, myTask2);

  myGraph->addEdge(myTask2, myTask1);

  myGraph->addEdge(myTask1, stateManager);

  myGraph->output(stateManager);

  myGraph->executeGraph();

  for (int i = 0; i < 10; i++) {
    myGraph->pushData(std::make_shared<int>(i));
    myGraph->pushData(std::make_shared<double>(i));
    myGraph->pushData(std::make_shared<float>(i));
  }

  myGraph->finishPushingData();

  myGraph->createDotFile("myDotFile.dot");

  int count = 0;

  while (std::shared_ptr<float> graphOutput = myGraph->getBlockingResult()) {
//    std::cout << "Received result: " << *graphOutput << std::endl;
    count++;
  }

  std::cout << "Total data processed = " << count << std::endl;

  myGraph->waitForTermination();

  myGraph->createDotFile("myDotFile.dot");

  return 0;
}