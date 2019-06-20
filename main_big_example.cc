#include "hedgehog/hedgehog.h"
#include "hedgehog/tools/graph_signal_handler.h"
int parallelism = 10;

std::mutex lock;

class A {
 private:
  int taskCount_ = 0;
 public:
  A(int taskCount) : taskCount_(taskCount) {}

  int taskCount() const {
    return taskCount_;
  }

};
class B {
 private:
  int taskCount_ = 0;
 public:
  B(int taskCount) : taskCount_(taskCount) {}

  int taskCount() const {
    return taskCount_;
  }
};
class C {
 private:
  int taskCount_ = 0;
 public:
  C(int taskCount) : taskCount_(taskCount) {}

  int taskCount() const {
    return taskCount_;
  }
};

class InputTask : public AbstractTask<double, int, float> {

 public:
  InputTask() : AbstractTask("InputTask", parallelism) {}

  ~InputTask() {
    totalCount += this->count;
//    std::cout << "I did " << this->count << " work, Total count = " << totalCount << std::endl;
  }
  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
    addResult(std::make_shared<double>((*input) + 1));
    count++;
  }
  void execute([[maybe_unused]]std::shared_ptr<float> input) override {
    addResult(std::make_shared<double>((*input) + 1));
    count++;
  }
  void shutdown() override {
    lock.lock();
    std::cout << "Task: " << this->name() << " count = " << count << std::endl;
    lock.unlock();
  }

  std::shared_ptr<AbstractTask<double, int, float>> copy() override {
    return std::make_shared<InputTask>();
  }
 private:
  static int totalCount;
  int count = 0;
};

int InputTask::totalCount = 0;

class InputTask2 : public AbstractTask<A, int, float> {
 public:
  InputTask2() : AbstractTask("InputTask2", parallelism) {}
  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
    addResult(std::make_shared<A>((*input) + 1));
    count++;
  }
  void execute([[maybe_unused]]std::shared_ptr<float> input) override {
    addResult(std::make_shared<A>((*input) + 1));
    count++;
  }
  void shutdown() override {
    lock.lock();
    std::cout << "Task: " << this->name() << " count = " << count << std::endl;
    lock.unlock();

  }

  std::shared_ptr<AbstractTask<A, int, float>> copy() override {
    return std::make_shared<InputTask2>();
  }

 private:
  int count = 0;
};

class OutputTask : public AbstractTask<A, int> {
 public:
  OutputTask() : AbstractTask("OutputTask", parallelism) {}
  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
    addResult(std::make_shared<A>((*input) + 1));
    count++;
  }
  void shutdown() override {
    lock.lock();
    std::cout << "Task: " << this->name() << " count = " << count << std::endl;
    lock.unlock();
  }

  std::shared_ptr<AbstractTask<A, int>> copy() override {
    return std::make_shared<OutputTask>();
  }
 private:
  int count = 0;
};

class OutputTask2 : public AbstractTask<A, int> {
 public:
  OutputTask2() : AbstractTask("OutputTask2", parallelism) {}
  void execute([[maybe_unused]]std::shared_ptr<int> input) override {
    addResult(std::make_shared<A>((*input) + 1));
    count++;
  }
  void shutdown() override {
    lock.lock();
    std::cout << "Task: " << this->name() << " count = " << count << std::endl;
    lock.unlock();
  }

  std::shared_ptr<AbstractTask<A, int>> copy() override {
    return std::make_shared<OutputTask2>();
  }

 private:
  int count = 0;
};

class TaskBDA : public AbstractTask<B, double, A> {
 public:
  TaskBDA() : AbstractTask("TaskBDA", parallelism) {}
  void execute([[maybe_unused]]std::shared_ptr<double> input) override {
    addResult(std::make_shared<B>((*input) + 1));
    count++;
  }
  void execute([[maybe_unused]]std::shared_ptr<A> input) override {
    addResult(std::make_shared<B>(input->taskCount() + 1));
    count++;
  }

  void shutdown() override {
    lock.lock();
    std::cout << "Task: " << this->name() << " count = " << count << std::endl;
    lock.unlock();
  }

  std::shared_ptr<AbstractTask<B, double, A>> copy() override {
    return std::make_shared<TaskBDA>();
  }

 private:
  int count = 0;
};

class TaskIC : public AbstractTask<int, C, B> {
 public:
  TaskIC() : AbstractTask("TaskIC", parallelism) {}
  void execute([[maybe_unused]]std::shared_ptr<C> input) override {
    addResult(std::make_shared<int>(input->taskCount() + 1));
    count++;
  }
  void execute([[maybe_unused]]std::shared_ptr<B> input) override {
    addResult(std::make_shared<int>(input->taskCount() + 1));
    count++;
  }
  void shutdown() override {
    lock.lock();
    std::cout << "Task: " << this->name() << " count = " << count << std::endl;
    lock.unlock();
  }
  std::shared_ptr<AbstractTask<int, C, B>> copy() override {
    return std::make_shared<TaskIC>();
  }

 private:
  int count = 0;
};

class TestState : public AbstractState<C, B, A> {
 public:
  void execute(std::shared_ptr<B> ptr) override {
    this->push(std::make_shared<C>(ptr->taskCount() + 1));
    count++;
  }

  void execute(std::shared_ptr<A> ptr) override {
    this->push(std::make_shared<C>(ptr->taskCount() + 1));
    count++;
  }
  ~TestState() {
    std::cout << "TestState: " << " count = " << count << std::endl;
  }
 private:
  int count = 0;
};

class MyStateManager : public AbstractStateManager<C, B, A> {
 public:
  MyStateManager(std::string_view const &name,
                 std::shared_ptr<AbstractState<C, B, A>> const &state,
                 bool automaticStart = false) : AbstractStateManager(name, state, automaticStart) {}

  void execute(std::shared_ptr<B> input) override {

    this->state()->lock();
    std::static_pointer_cast<Execute<B>>(this->state())->execute(input);
    while (!this->state()->readyList()->empty()) {
      this->addResult(this->state()->frontAndPop());
    }
    this->state()->unlock();
  }

  void execute(std::shared_ptr<A> input) override {
    this->state()->lock();
    std::static_pointer_cast<Execute<A>>(this->state())->execute(input);
    while (!this->state()->readyList()->empty()) {
      this->addResult(this->state()->frontAndPop());
    }
    this->state()->unlock();
  }

};

class InnerEP : public AbstractExecutionPipeline<int, double, A> {
 public:
  InnerEP(std::shared_ptr<Graph<int, double, A>> const &graph,
          size_t const &numberGraphDuplications,
          std::vector<int> const &deviceIds,
          bool automaticStart = false) : AbstractExecutionPipeline(graph,
                                                                   numberGraphDuplications,
                                                                   deviceIds,
                                                                   automaticStart) {}
  bool sendToGraph([[maybe_unused]]std::shared_ptr<double> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
  bool sendToGraph([[maybe_unused]]std::shared_ptr<A> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
};

int main() {
  Graph<A, int, float> graph("Outer Graph");

  auto innerGraph = std::make_shared<Graph<int, double, A>>("Inner Graph");

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

  GraphSignalHandler<A, int, float>::registerGraph(&graph);
  GraphSignalHandler<A, int, float>::registerSignal(SIGTERM, false);

//  graph.createDotFile("beforeExecute.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);

  graph.executeGraph();

//  graph.createDotFile("afterExecute.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);

  for (int i = 0; i < 1; ++i) {
    graph.pushData(std::make_shared<int>(0));
    graph.pushData(std::make_shared<float>(0));
  }

//  graph.createDotFile("afterPush.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);

  graph.finishPushingData();

//  graph.createDotFile("afterFinishingPush.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);

  int count = 0;

  while (std::shared_ptr<A> result = graph.getBlockingResult()) {
//    std::cout << "Processing result: " << result->taskCount() << std::endl;
    count++;
  }

  std::cout << "Total data received: " << count << std::endl;

  graph.waitForTermination();

  graph.createDotFile("N_N.dot", ColorScheme::NONE, StructureOptions::NONE, DebugOptions::DEBUG);
  graph.createDotFile("N_AT.dot", ColorScheme::NONE, StructureOptions::ALLTHREADING, DebugOptions::DEBUG);
  graph.createDotFile("N_Q.dot", ColorScheme::NONE, StructureOptions::QUEUE, DebugOptions::DEBUG);
  graph.createDotFile("N_A.dot", ColorScheme::NONE, StructureOptions::ALL, DebugOptions::DEBUG);

  graph.createDotFile("E_N.dot", ColorScheme::EXECUTION, StructureOptions::NONE, DebugOptions::DEBUG);
  graph.createDotFile("E_AT.dot", ColorScheme::EXECUTION, StructureOptions::ALLTHREADING, DebugOptions::DEBUG);
  graph.createDotFile("E_Q.dot", ColorScheme::EXECUTION, StructureOptions::QUEUE, DebugOptions::DEBUG);
  graph.createDotFile("E_A.dot", ColorScheme::EXECUTION, StructureOptions::ALL, DebugOptions::DEBUG);

  graph.createDotFile("W_N.dot", ColorScheme::WAIT, StructureOptions::NONE, DebugOptions::DEBUG);
  graph.createDotFile("W_AT.dot", ColorScheme::WAIT, StructureOptions::ALLTHREADING, DebugOptions::DEBUG);
  graph.createDotFile("W_Q.dot", ColorScheme::WAIT, StructureOptions::QUEUE, DebugOptions::DEBUG);
  graph.createDotFile("W_A.dot", ColorScheme::WAIT, StructureOptions::ALL, DebugOptions::DEBUG);
}


