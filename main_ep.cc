#include <mutex>
std::mutex mutex;

#include <chrono>
#include "hedgehog/hedgehog.h"
#include "hedgehog/tools/graph_signal_handler.h"

class IFToIEP : public AbstractExecutionPipeline<int, int, float> {
 private:
  int count_ = 0;
 public:
  IFToIEP(std::string_view const &name,
          std::shared_ptr<Graph<int, int, float>> const &graph,
          size_t const &numberGraphDuplications,
          std::vector<int> const &deviceIds) : AbstractExecutionPipeline(name,
                                                                         graph,
                                                                         numberGraphDuplications,
                                                                         deviceIds) {}
  virtual ~IFToIEP() = default;
 protected:
  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
//    std::cout << "Checking switch" << std::endl;
    count_++;
    return true;
  }
  bool sendToGraph([[maybe_unused]]std::shared_ptr<float> &data, [[maybe_unused]]size_t const &graphId) override {
    count_++;
    return true;
  }

  std::string extraPrintingInformation() const override {
    return std::to_string(count_);
  }
};

class IFToI : public AbstractTask<int, int, float> {
 private:
  int count_ = 0;
 public:
  IFToI(std::string_view const &name) : AbstractTask(name, 3) {}
  virtual ~IFToI() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
//    std::cout << this->name() << "Received/ data" << std::endl;
    count_++;
    addResult(std::make_shared<int>(2));
  }
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
//    std::cout << this->name() << "Received data" << std::endl;
    count_++;
    addResult(std::make_shared<int>(2));
  }
  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
    return std::make_shared<IFToI>(this->name());
  }

  void initialize() override {
    mutex.lock();
//      std::cout << "Initialized " << this->name() << ":: x" << this << " tid(" << (int) this->core()->threadId() << ") -- gid(" << this->core()->graphId() << ")" << std::endl;
    mutex.unlock();
  }
  std::string extraPrintingInformation() const override {
    return std::to_string(count_);
  }
};

class IFToIEpIn : public AbstractTask<int, int, float> {
 private:
  int count_ = 0;
 public:
  IFToIEpIn(std::string_view const &name) : AbstractTask(name, 3) {}
  virtual ~IFToIEpIn() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    count_++;
//    std::cout << this->name() << "Received data" << std::endl;
    addResult(std::make_shared<int>(2));
  }
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
    count_++;
//    std::cout << this->name() << "Received data" << std::endl;
    addResult(std::make_shared<int>(2));
  }
  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
    return std::make_shared<IFToIEpIn>(this->name());
  }

  void initialize() override {
    mutex.lock();
//    std::cout << "Initialized " << this->name() << ":: x" << this->core() << " tid(" << (int) this->core()->threadId() << ") -- gid(" << this->core()->graphId() << ")" << std::endl;
    mutex.unlock();
  }

  void shutdown() override {
    mutex.lock();
//    std::cout << "Shutdown " << this->name() << " tid(" << (int) this->core()->threadId() << ") -- gid(" << this->core()->graphId() << ")" << std::endl;
    mutex.unlock();
  }
  std::string extraPrintingInformation() const override {
    return std::to_string(count_);
  }
//  bool canTerminate() override {
//    bool ret = AbstractTask::canTerminate();
//
////    std::cout << "EP In can terminate = " << ret << std::endl;
//
//    return ret;
//  }
};

class EPIFToI : public AbstractTask<int, int, float> {
 private:
  int count_ = 0;
 public:
  EPIFToI(std::string_view const &name) : AbstractTask(name, 3) {}
  virtual ~EPIFToI() = default;
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    count_++;
    addResult(std::make_shared<int>(2));
  }
  void execute([[maybe_unused]]std::shared_ptr<float> ptr) override {
    count_++;
    addResult(std::make_shared<int>(2));
  }
  std::shared_ptr<AbstractTask<int, int, float>> copy() override {
    return std::make_shared<EPIFToI>(this->name());
  }

  void initialize() override {
    mutex.lock();
//    std::cout << "Initialized " << this->name() << ":: x" << this->core() << " tid(" << (int) this->core()->threadId() << ") -- gid(" << this->core()->graphId() << ")" << std::endl;
    mutex.unlock();
  }

  std::string extraPrintingInformation() const override {
    return std::to_string(count_);
  }
};

int main() {

  std::vector<int> deviceIds = {0, 0, 0};

  auto
      outterGraph = std::make_shared<Graph<int, int, float>>("Output"),
      epGraph = std::make_shared<Graph<int, int, float>>("epGRaph");

  auto
      intputTask = std::make_shared<IFToI>("InputTask"),
      outputTask = std::make_shared<IFToI>("OutputTask");

  auto
      inputEPTask = std::make_shared<IFToIEpIn>("InputEPTask");

  auto
      outputEPTask = std::make_shared<EPIFToI>("OutputEPTask");

  epGraph->input(inputEPTask);
  epGraph->addEdge(inputEPTask, outputEPTask);
  epGraph->output(outputEPTask);

  auto ep = std::make_shared<IFToIEP>("ExecPipeline",
                                      epGraph,
                                      3,
                                      deviceIds);

  outterGraph->input(intputTask);
  outterGraph->addEdge(intputTask, ep);
  outterGraph->addEdge(ep, outputTask);
  outterGraph->output(outputTask);

  outterGraph->createDotFile("creation.dot", ColorScheme::EXECUTION, StructureOptions::ALL, DebugOptions::DEBUG);

  outterGraph->executeGraph();

  GraphSignalHandler<int, int, float>::registerGraph(outterGraph.get());
  GraphSignalHandler<int, int, float>::registerSignal(SIGTERM, false);

  outterGraph->createDotFile("execute.dot", ColorScheme::EXECUTION, StructureOptions::ALL, DebugOptions::DEBUG);

  using namespace std::chrono_literals;
//  std::this_thread::sleep_for(2s);

  for (int i = 0; i < 100; ++i) {
    outterGraph->pushData(std::make_shared<int>(i));
  }

  outterGraph->finishPushingData();
  int count = 0;
  while (auto data = outterGraph->getBlockingResult()) { ++count; }

  std::cout << count << " elements processed." << std::endl;

  outterGraph->waitForTermination();

  outterGraph->createDotFile("test.dot", ColorScheme::EXECUTION, StructureOptions::ALL, DebugOptions::DEBUG);

  return 0;
}