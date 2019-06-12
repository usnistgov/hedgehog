//////#include <iostream>
//////#include <chrono>
//////#include <thread>
////#include "hedgehog/hedgehog.h"
//#include "hedgehog/tools/graph_signal_handler.h"
////////
////////using namespace std;
////////
////////class ItoF : public AbstractTask<float, int, double, char> {
//////// public:
////////  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {}
////////  void execute([[maybe_unused]]std::shared_ptr<double> ptr) override {}
////////  void execute([[maybe_unused]]std::shared_ptr<char> ptr) override {}
////////    //std::cout << "Received int" << std::endl;
//////////    addResult(std::make_shared<float>(*ptr));
//////////  }
////////};
////////
////////int main() {
////////  std::chrono::time_point<std::chrono::high_resolution_clock>
////////      start,
////////      finish;
////////  Graph<float, int, double, char> g("GraphOutput");
////////  auto queue = std::make_shared<ItoF>();
////////  g.input(queue);
//////////  g.output(task);
////////
////////  g.executeGraph();
////////
//////////  g.createDotFile("output.dot");
////////std::this_thread::sleep_for(1000ms);
////////  start = std::chrono::high_resolution_clock::now();
////////  for (uint64_t i = 0; i < 1000000000; ++i) {
////////    g.pushData(std::make_shared<int>(i));
////////  }
////////
////////  g.finishPushingData();
////////
////////  g.waitForTermination();
////////  finish = std::chrono::high_resolution_clock::now();
////////
////////  std::cerr << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << std::endl;
////////}
//////
//////
//////class floatAllocator : public AbstractAllocator<float> {
////// private:
//////  void initialize() override {}
////// private:
//////  float *allocate([[maybe_unused]]size_t size) override {
//////    return new float();
//////  }
//////
//////  void deallocate(float *data) override {
//////    delete data;
//////  }
//////};
//////
//////class floatReleaseRule : public AbstractReleaseRule<float> {
////// public:
//////  size_t count = 1;
//////  void used() override {
//////    --count;
//////  }
//////
//////  bool canRelease() override {
//////    return count == 0;
//////  }
//////};
//////
//////class ItoF : public AbstractManagedMemoryReceiverTask<float, int, float> {
//////
////// public:
//////  ItoF() :
//////      AbstractManagedMemoryReceiverTask(std::make_shared<StaticMemoryManager<float>>(3,
//////                                                                                     1,
//////                                                                                     std::make_unique<floatAllocator>()),
//////                                        "IToF",
//////                                        10) {
//////  }
//////
//////  ItoF(ItoF *rhs) : AbstractManagedMemoryReceiverTask(rhs) {}
//////
//////  std::shared_ptr<AbstractTask<ManagedMemory<float>, int, float>> copy() override {
//////    return std::make_shared<ItoF>(this);
//////  }
//////
//////  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
//////    auto mem = this->getMemory(std::make_unique<floatReleaseRule>(), 1);
//////    addResult(mem);
//////  }
//////  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
//////    auto mem = this->getMemory(std::make_unique<floatReleaseRule>(), 1);
//////    addResult(mem);
//////  }
//////};
//////
//////class MFtoF : public AbstractTask<float, ManagedMemory<float>> {
//////
////// public:
//////  MFtoF() : AbstractTask("MFtoF") {}
//////
//////  void execute([[maybe_unused]]std::shared_ptr<ManagedMemory<float>> ptr) override {
//////    ptr->release();
//////    addResult(std::make_shared<float>(2));
//////  }
//////
//////};
//////
//////int main() {
//////  Graph<float, int, float> g("GraphOutput");
//////  auto task = std::make_shared<ItoF>();
//////  auto task2 = std::make_shared<MFtoF>();
//////  g.input(task);
//////  g.addEdge(task, task2);
//////  g.output(task2);
//////  g.executeGraph();
//////  g.createDotFile("output.dot");
//////
//////  for (int i = 0; i < 100; ++i) {
//////    g.pushData(std::make_shared<int>(i));
//////    g.pushData(std::make_shared<float>(i));
//////  }
//////
//////  g.finishPushingData();
//////
//////  while (std::shared_ptr<float> result = g.getBlockingResult()) {
////////    std::cout << "Processing result: " << *result << std::endl;
//////  }
//////
//////  g.createDotFile("outputBeforeWait.dot", ColorScheme::EXECUTION);
//////  g.waitForTermination();
//////  g.createDotFile("outputAfterWait.dot", ColorScheme::EXECUTION);
//////
//////}
//////
//////
//////class ItoF : public AbstractTask<float, int> {
////// public:
//////  ItoF() : AbstractTask("ItoF", 10) {}
//////
//////  void execute(std::shared_ptr<int> ptr) override {
//////    //std::cout << "Received int Outer Graph " << *ptr << std::endl;
//////    addResult(std::make_shared<float>(*ptr));
//////  }
//////  std::shared_ptr<AbstractTask<float, int>> copy() override {
//////    return std::make_shared<ItoF>();
//////  }
//////};
//////
//////class IFtoF : public AbstractTask<float, int, float> {
////// public:
//////  IFtoF() : AbstractTask("IFtoF", 10) {}
//////
//////  void execute(std::shared_ptr<int> ptr) override {
//////    //std::cout << "Should not come here" << std::endl;
//////
//////    addResult(std::make_shared<float>(*ptr));
//////  }
//////  void execute(std::shared_ptr<float> ptr) override {
//////    //std::cout << "Received float Outer Graph " << *ptr << std::endl;
//////    addResult(std::make_shared<float>(*ptr));
//////  }
//////
//////  std::shared_ptr<AbstractTask<float, int, float>> copy() override {
//////    return std::make_shared<IFtoF>();
//////  }
//////
//////};
//////
//////int main() {
//////  Graph<float, int> outerGraph("outerGraph");
//////
//////  auto innerGraph = std::make_shared<Graph<float, int, float>>("Inner");
//////
//////  auto outerTask = std::make_shared<ItoF>();
//////  auto innerTask = std::make_shared<IFtoF>();
//////
//////  outerGraph.input(outerTask);
//////
//////  innerGraph->input(innerTask);
//////  innerGraph->output(innerTask);
//////
//////  outerGraph.addEdge(outerTask, innerGraph);
//////
//////  outerGraph.output(innerGraph);
//////
//////  GraphSignalHandler<float, int>::registerGraph(&outerGraph);
//////
//////  outerGraph.executeGraph();
//////  outerGraph.createDotFile("output.dot", ColorScheme::EXECUTION);
//////
//////  for (int i = 0; i < 100; ++i) {
//////    outerGraph.pushData(std::make_shared<int>(i));
//////  }
//////
//////
//////  outerGraph.finishPushingData();
//////
//////  while (std::shared_ptr<float> result = outerGraph.getBlockingResult()) {
//////    std::cout << "Processing result: " << *result << std::endl;
//////  }
//////
//////  outerGraph.waitForTermination();
//////  outerGraph.createDotFile("after-wait-output.dot", ColorScheme::EXECUTION);
//////}
//
////int parallelism = 10;
////
////
////class A {
//// private:
////  int taskCount_ = 0;
//// public:
////  A(int taskCount) : taskCount_(taskCount) {}
////
////  int taskCount() const {
////    return taskCount_;
////  }
////
////};
////class B {
//// private:
////  int taskCount_ = 0;
//// public:
////  B(int taskCount) : taskCount_(taskCount) {}
////
////  int taskCount() const {
////    return taskCount_;
////  }
////};
////class C {
//// private:
////  int taskCount_ = 0;
//// public:
////  C(int taskCount) : taskCount_(taskCount) {}
////
////  int taskCount() const {
////    return taskCount_;
////  }
////};
////
////class InputTask : public AbstractTask<double, int, float> {
////
//// public:
////  InputTask() : AbstractTask("InputTask", parallelism) {}
////
////  ~InputTask() {
////    totalCount += this->count;
////    std::cout << "I did " << this->count << " work, Total count = " << totalCount << std::endl;
////  }
////  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
////    addResult(std::make_shared<double>((*input) + 1));
////    count++;
////  }
////  void execute([[maybe_unused]]std::shared_ptr<float> input) override {
////    addResult(std::make_shared<double>((*input) + 1));
////    count++;
////  }
////  void shutdown() override {
////    std::cout << "Task: " << this->name() << " count = " << count << std::endl;
////  }
////
////  std::shared_ptr<AbstractTask<double, int, float>> copy() override {
////    return std::make_shared<InputTask>();
////  }
//// private:
////  static int totalCount;
////  int count = 0;
////};
////
////int InputTask::totalCount = 0;
////
////class InputTask2 : public AbstractTask<A, int, float> {
//// public:
////  InputTask2() : AbstractTask("InputTask2", parallelism) {}
////  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
////    addResult(std::make_shared<A>((*input) + 1));
////    count++;
////  }
////  void execute([[maybe_unused]]std::shared_ptr<float> input) override {
////    addResult(std::make_shared<A>((*input) + 1));
////    count++;
////  }
////  void shutdown() override {
////    //std::cout << "Task: " << this->name() << " count = " << count << std::endl;
////  }
////
////  std::shared_ptr<AbstractTask<A, int, float>> copy() override {
////    return std::make_shared<InputTask2>();
////  }
////
//// private:
////  int count = 0;
////};
////
////class OutputTask : public AbstractTask<A, int> {
//// public:
////  OutputTask() : AbstractTask("OutputTask", parallelism) {}
////  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
////    addResult(std::make_shared<A>((*input) + 1));
////    count++;
////  }
////  void shutdown() override {
////    //std::cout << "Task: " << this->name() << " count = " << count << std::endl;
////  }
////
////  std::shared_ptr<AbstractTask<A, int>> copy() override {
////    return std::make_shared<OutputTask>();
////  }
//// private:
////  int count = 0;
////};
////
////class OutputTask2 : public AbstractTask<A, int> {
//// public:
////  OutputTask2() : AbstractTask("OutputTask2", parallelism) {}
////  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
////    addResult(std::make_shared<A>((*input) + 1));
////    count++;
////  }
////  void shutdown() override {
////    //std::cout << "Task: " << this->name() << " count = " << count << std::endl;
////  }
////
////  std::shared_ptr<AbstractTask<A, int>> copy() override {
////    return std::make_shared<OutputTask2>();
////  }
////
//// private:
////  int count = 0;
////};
////
////class TaskBDA : public AbstractTask<B, double, A> {
//// public:
////  TaskBDA() : AbstractTask("TaskBDA", parallelism) {}
////  void execute([[maybe_unused]]std::shared_ptr<double> input) override {
////    addResult(std::make_shared<B>((*input) + 1));
////    count++;
////  }
////  void execute([[maybe_unused]]std::shared_ptr<A> input) override {
////    addResult(std::make_shared<B>(input->taskCount() + 1));
////    count++;
////  }
////  void shutdown() override {
////    //std::cout << "Task: " << this->name() << " count = " << count << std::endl;
////  }
////
////  std::shared_ptr<AbstractTask<B, double, A>> copy() override {
////    return std::make_shared<TaskBDA>();
////  }
////
//// private:
////  int count = 0;
////};
////
////class TaskIC : public AbstractTask<int, C, B> {
//// public:
////  TaskIC() : AbstractTask("TaskIC", parallelism) {}
////  void execute([[maybe_unused]]std::shared_ptr<C> input) override {
////    addResult(std::make_shared<int>(input->taskCount() + 1));
////    count++;
////  }
////  void execute([[maybe_unused]]std::shared_ptr<B> input) override {
////    addResult(std::make_shared<int>(input->taskCount() + 1));
////    count++;
////  }
////  void shutdown() override {
////    //std::cout << "Task: " << this->name() << " count = " << count << std::endl;
////  }
////  std::shared_ptr<AbstractTask<int, C, B>> copy() override {
////    return std::make_shared<TaskIC>();
////  }
////
//// private:
////  int count = 0;
////};
////
////class TestState : public AbstractState<C, B, A> {
//// public:
////  void execute(std::shared_ptr<B> ptr) override {
////    this->push(std::make_shared<C>(ptr->taskCount() + 1));
////    count++;
////  }
////
////  void execute(std::shared_ptr<A> ptr) override {
////    this->push(std::make_shared<C>(ptr->taskCount() + 1));
////    count++;
////  }
////  ~TestState() {
////    //std::cout << "TestState: " << " count = " << count << std::endl;
////  }
//// private:
////  int count = 0;
////};
////
////class MyStateManager : public AbstractStateManager<C, B, A> {
//// public:
////  MyStateManager(std::string_view const &name,
////                 std::shared_ptr<AbstractState<C, B, A>> const &state,
////                 bool automaticStart = false) : AbstractStateManager(name, state, automaticStart) {}
////
////  void execute(std::shared_ptr<B> input) override {
////
////    this->state()->lock();
////    std::static_pointer_cast<Execute<B>>(this->state())->execute(input);
////    while (!this->state()->readyList()->empty()) {
////      this->addResult(this->state()->frontAndPop());
////    }
////    this->state()->unlock();
////  }
////
////  void execute(std::shared_ptr<A> input) override {
////    this->state()->lock();
////    std::static_pointer_cast<Execute<A>>(this->state())->execute(input);
////    while (!this->state()->readyList()->empty()) {
////      this->addResult(this->state()->frontAndPop());
////    }
////    this->state()->unlock();
////  }
////};
////
////int main() {
////  Graph<A, int, float> graph("Outer Graph");
////
////  auto innerGraph = std::make_shared<Graph<int, double, A>>("Inner Graph");
////
////  auto inputTask = std::make_shared<InputTask>();
////  auto inputTask2 = std::make_shared<InputTask2>();
////  auto outputTask = std::make_shared<OutputTask>();
////  auto outputTask2 = std::make_shared<OutputTask2>();
////  auto taskBDA = std::make_shared<TaskBDA>();
////  auto taskBDA2 = std::make_shared<TaskBDA>();
////  auto taskIC = std::make_shared<TaskIC>();
////  auto taskIC2 = std::make_shared<TaskIC>();
////
////  auto testState = std::make_shared<TestState>();
////
////  auto stateManager = std::make_shared<DefaultStateManager<C, B, A>>("My State Manager", testState);
////
////  innerGraph->input(taskBDA);
////  innerGraph->input(taskBDA2);
////  innerGraph->addEdge(taskBDA, taskIC);
////  innerGraph->addEdge(taskBDA2, taskIC);
////  innerGraph->output(taskIC);
////
////  innerGraph->addEdge(taskBDA, stateManager);
////  innerGraph->addEdge(taskBDA2, stateManager);
////  innerGraph->addEdge(stateManager, taskIC2);
////  innerGraph->output(taskIC2);
////
////  graph.input(inputTask);
////  graph.input(inputTask2);
////
////  graph.addEdge(inputTask, innerGraph);
////  graph.addEdge(inputTask2, innerGraph);
////
////  graph.addEdge(innerGraph, outputTask);
////  graph.addEdge(innerGraph, outputTask2);
////
////  graph.output(outputTask);
////  graph.output(outputTask2);
////
////  GraphSignalHandler<A, int, float>::registerGraph(&graph);
////  GraphSignalHandler<A, int, float>::registerSignal(SIGTERM, false);
////
//////  graph.createDotFile("beforeExecute.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);
////
////  graph.executeGraph();
////
//////  graph.createDotFile("afterExecute.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);
////
////  for (int i = 0; i < 1000; ++i) {
////    graph.pushData(std::make_shared<int>(0));
////    graph.pushData(std::make_shared<float>(0));
////  }
////
//////  graph.createDotFile("afterPush.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);
////
////  graph.finishPushingData();
////
//////  graph.createDotFile("afterFinishingPush.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);
////
////  int count = 0;
////
////  while (std::shared_ptr<A> result = graph.getBlockingResult()) {
//////    std::cout << "Processing result: " << result->taskCount() << std::endl;
////    count++;
////  }
////
////  std::cout << "Total data received: " << count << std::endl;
////
////  graph.waitForTermination();
////
////  graph.createDotFile("N_N.dot", ColorScheme::NONE, StructureOptions::NONE, DebugOptions::DEBUG);
////  graph.createDotFile("N_AT.dot", ColorScheme::NONE, StructureOptions::ALLTHREADING, DebugOptions::DEBUG);
////  graph.createDotFile("N_Q.dot", ColorScheme::NONE, StructureOptions::QUEUE, DebugOptions::DEBUG);
////  graph.createDotFile("N_A.dot", ColorScheme::NONE, StructureOptions::ALL, DebugOptions::DEBUG);
////
////  graph.createDotFile("E_N.dot", ColorScheme::EXECUTION, StructureOptions::NONE);
////  graph.createDotFile("E_AT.dot", ColorScheme::EXECUTION, StructureOptions::ALLTHREADING);
////  graph.createDotFile("E_Q.dot", ColorScheme::EXECUTION, StructureOptions::QUEUE);
////  graph.createDotFile("E_A.dot", ColorScheme::EXECUTION, StructureOptions::ALL);
////
////  graph.createDotFile("W_N.dot", ColorScheme::WAIT, StructureOptions::NONE, DebugOptions::DEBUG);
////  graph.createDotFile("W_AT.dot", ColorScheme::WAIT, StructureOptions::ALLTHREADING, DebugOptions::DEBUG);
////  graph.createDotFile("W_Q.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);
////  graph.createDotFile("W_A.dot", ColorScheme::WAIT, StructureOptions::ALL, DebugOptions::DEBUG);
////}
//
////#include "hedgehog/hedgehog.h"
////
////class mySwitch : public Switch<int, float> {
//// protected:
////  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]] size_t &graphId) override {
////    return false;
////  }
////
////  bool sendToGraph([[maybe_unused]]std::shared_ptr<float> &data, [[maybe_unused]] size_t &graphId) override {
////    return false;
////  }
////
////};
////int main() {
////  std::vector<int> deviceIds = {0, 0, 0};
////
////  ExecutionPipeline<int, int, float> e(
////      std::make_shared<Graph<int, int, float>>(),
////      std::make_unique<mySwitch>(),
////      3,
////      deviceIds);
////  ExecutionPipeline<int, int, float>
////      e2("ExecPipeline", std::make_shared<Graph<int, int, float>>(), std::make_unique<mySwitch>(), 3, deviceIds);
////
////  return 0;
////}
//
//
//#include "hedgehog/hedgehog.h"
//
//class IFToIEP : public AbstractExecutionPipeline<int, int, float> {
// public:
//  IFToIEP(std::string_view const &name,
//          std::shared_ptr<Graph<int, int, float>> const &graph,
//          size_t const &numberGraphDuplications,
//          std::vector<int> const &deviceIds) : AbstractExecutionPipeline(name,
//                                                                         graph,
//                                                                         numberGraphDuplications,
//                                                                         deviceIds) {}
// protected:
//  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
////    std::cout << "Checking switch" << std::endl;
//    return true;
//  }
//  bool sendToGraph([[maybe_unused]]std::shared_ptr<float> &data, [[maybe_unused]]size_t const &graphId) override {
//    return true;
//  }
//
//};
//
//class IFToI : public AbstractTask<int, int, float> {
// public:
//  IFToI(std::string_view const &name) : AbstractTask(name, 3) {}
//  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
////    std::cout << this->name() << "Received data" << std::endl;
//    addResult(std::make_shared<int>(2));
//  }
//  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
//    addResult(std::make_shared<int>(2));
//  }
//  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
//    return std::make_shared<IFToI>("copy");
//  }
//};
//
//class IFToIEpIn : public AbstractTask<int, int, float> {
// public:
//  IFToIEpIn(std::string_view const &name) : AbstractTask(name, 3) {}
//  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
////    std::cout << this->name() << "Received data" << std::endl;
//    addResult(std::make_shared<int>(2));
//  }
//  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
//    addResult(std::make_shared<int>(2));
//  }
//  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
//    return std::make_shared<IFToI>("copy");
//  }
//
////  bool canTerminate() override {
////    bool ret = AbstractTask::canTerminate();
////
//////    std::cout << "EP In can terminate = " << ret << std::endl;
////
////    return ret;
////  }
//};
//
//class EPIFToI : public AbstractTask<int, int, float> {
// public:
//  EPIFToI(std::string_view const &name) : AbstractTask(name, 3) {}
//  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
//    addResult(std::make_shared<int>(2));
//  }
//  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
//    addResult(std::make_shared<int>(2));
//  }
//  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
//    return std::make_shared<IFToI>("copy");
//  }
//};
//
//int main() {
//  std::vector<int> deviceIds = {0, 0, 0};
//
//  auto
//      outterGraph = std::make_shared<Graph<int, int, float>>("Output"),
//      epGraph = std::make_shared<Graph<int, int, float>>("epGRaph");
//
//  auto
//      intputTask = std::make_shared<IFToI>("InputTask"),
//      outputTask = std::make_shared<IFToI>("OutputTask");
//  auto
//      inputEPTask = std::make_shared<IFToIEpIn>("InputEPTask");
//  auto
//      outputEPTask = std::make_shared<EPIFToI>("OutputEPTask");
//
//  epGraph->input(inputEPTask);
//  epGraph->addEdge(inputEPTask, outputEPTask);
//  epGraph->output(outputEPTask);
//
//  auto ep = std::make_shared<IFToIEP>("ExecPipeline",
//                                      epGraph,
//                                      3,
//                                      deviceIds);
//
//  outterGraph->input(intputTask);
//  outterGraph->addEdge(intputTask, ep);
//  outterGraph->addEdge(ep, outputTask);
//  outterGraph->output(outputTask);
//
//
//  outterGraph->createDotFile("creation.dot", ColorScheme::EXECUTION, StructureOptions::ALL, DebugOptions::DEBUG);
//
//  outterGraph->executeGraph();
//
//  GraphSignalHandler<int, int, float>::registerGraph(outterGraph.get());
//  GraphSignalHandler<int, int, float>::registerSignal(SIGTERM, false);
//
//  outterGraph->createDotFile("execute.dot", ColorScheme::EXECUTION, StructureOptions::ALL, DebugOptions::DEBUG);
//
//  for (int i = 0; i < 100; ++i) {
//    outterGraph->pushData(std::make_shared<int>(i));
//  }
//
//  outterGraph->finishPushingData();
//int count = 0;
//  while (auto data = outterGraph->getBlockingResult()){
//    ++count;
//    std:: cout << "Received output data = " << *data << std::endl;
//  }
//
//  std::cout << count << " elements processed." << std::endl;
//
//  outterGraph->waitForTermination();
//
//  outterGraph->createDotFile("test.dot", ColorScheme::EXECUTION, StructureOptions::ALL, DebugOptions::DEBUG);
//
//  return 0;
//}


#include <iostream>

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