#include "hedgehog/hedgehog.h"
#include "hedgehog/tools/graph_signal_handler.h"

int parallel = 5;

template<class T>
class MatrixData : public MemoryData<MatrixData<T>> {
  T *data_ = nullptr;
  size_t
      matrixSize_ = 1024 * 1024;

 public:
  MatrixData() {}
  void data(T *data) {
    data_ = data;
  }
  size_t matrixSize() const { return matrixSize_; }
  virtual ~MatrixData() { delete[] data_; }
  void recycle() override { std::fill_n(data_, matrixSize_, 0); }
};

template<class T>
class DynamicMatrixData : public MemoryData<DynamicMatrixData<T>> {
  T *data_ = nullptr;

 public:
  DynamicMatrixData() {}
  virtual ~DynamicMatrixData() = default;
  void data(T *data) { data_ = data; }
  void recycle() override { delete[] data_; }
};

template<class T>
class StaticMM : public AbstractStaticMemoryManager<MatrixData<T>> {
 public:
  explicit StaticMM(size_t const &poolSize) : AbstractStaticMemoryManager<MatrixData<T>>(poolSize) {}

  std::shared_ptr<AbstractMemoryManager<MatrixData<T>>> copy() override {
    return std::make_shared<StaticMM>(this->poolSize());
  }

  virtual ~StaticMM() {}

  void allocate(std::shared_ptr<MatrixData<T>> &ptr) override { ptr->data(new T[ptr->matrixSize()]); }
  bool canRecycle([[maybe_unused]]std::shared_ptr<MatrixData<T>> const &ptr) override { return true; }
};

template<class T>
class DynamicMM : public AbstractDynamicMemoryManager<DynamicMatrixData<T>> {
 public:
  DynamicMM(size_t const &poolSize) : AbstractDynamicMemoryManager<DynamicMatrixData<T>>(poolSize) {}

  std::shared_ptr<AbstractMemoryManager<DynamicMatrixData<T>>> copy() override {
    return std::make_shared<DynamicMM<T>>(this->poolSize());
  }

  bool canRecycle([[maybe_unused]]std::shared_ptr<DynamicMatrixData<T>> const &ptr) override { return true; }
};

template<class T>
class CudaMM : public AbstractCUDAMemoryManager<MatrixData<T>> {
 public:
  CudaMM(size_t const &poolSize) : AbstractCUDAMemoryManager<MatrixData<T>>(poolSize) {}

  bool canRecycle([[maybe_unused]]std::shared_ptr<MatrixData<T>> const &ptr) override {
    return true;
  }

  std::shared_ptr<AbstractMemoryManager<MatrixData<T>>> copy() override {
    return std::make_shared<CudaMM<T>>(this->poolSize());
  }

  void allocate(std::shared_ptr<MatrixData<T>> &ptr) override {
    ptr->data(new T[ptr->matrixSize()]);
  }
};

class MyCUDATask : public AbstractCUDATask<MatrixData<int>, int> {
 private:
  int count = 0;
 public:
  MyCUDATask() : AbstractCUDATask("CUDA Task") {}
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    ++count;
    addResult(this->getManagedMemory());
  }
  std::string extraPrintingInformation() const override {
    return "Count " + std::to_string(count);
  }
  std::shared_ptr<AbstractTask<MatrixData<int>, int>> copy() override {
    return std::make_shared<MyCUDATask>();
  }
};

class MyStaticTask : public AbstractTask<MatrixData<int>, int> {
 private:
  int count = 0;
 public:
  MyStaticTask() : AbstractTask("Static Task", parallel) {}
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    ++count;
    addResult(this->getManagedMemory());
  }
  std::string extraPrintingInformation() const override {
    return "Count " + std::to_string(count);
  }
  std::shared_ptr<AbstractTask<MatrixData<int>, int>> copy() override {
    return std::make_shared<MyStaticTask>();
  }
};

class MyDynamicTask : public AbstractTask<DynamicMatrixData<int>, int> {
 private:
  int count = 0;
 public:
  MyDynamicTask() : AbstractTask("Dynamic Task", parallel) {}
  void execute([[maybe_unused]]std::shared_ptr<int> ptr) override {
    ++count;
    auto mem = this->getManagedMemory();
    mem->data(new int[30]());
    addResult(mem);
  }
  std::string extraPrintingInformation() const override {
    return "Count " + std::to_string(count);
  }
  std::shared_ptr<AbstractTask<DynamicMatrixData<int>, int>> copy() override {
    return std::make_shared<MyDynamicTask>();
  }
};

class OutputTask : public AbstractTask<int, MatrixData<int>, DynamicMatrixData<int>> {
 public:
  OutputTask() : AbstractTask("output") {}

  void execute(std::shared_ptr<MatrixData<int>> ptr) override {
    ptr->returnToMemoryManager();
    addResult(std::make_shared<int>(1));
  }

  void execute(std::shared_ptr<DynamicMatrixData<int>> ptr) override {
    ptr->returnToMemoryManager();
    addResult(std::make_shared<int>(1));
  }

  std::shared_ptr<AbstractTask<int, MatrixData<int>, DynamicMatrixData<int>>> copy() override {
    return std::make_shared<OutputTask>();
  }
};

class IIEP : public AbstractExecutionPipeline<int, int> {
 public:
  IIEP(std::shared_ptr<Graph<int, int>> const &graph,
       size_t const &numberGraphDuplications,
       std::vector<int> const &deviceIds,
       bool automaticStart) : AbstractExecutionPipeline(graph, numberGraphDuplications, deviceIds, automaticStart) {}
  bool sendToGraph([[maybe_unused]]std::shared_ptr<int> &data, [[maybe_unused]]size_t const &graphId) override {
    return true;
  }
};

int main() {
  std::vector<int> vDevices = {0, 1, 2, 3, 4};

  std::shared_ptr<IIEP> iiep = nullptr;

  auto
      insideGraph = std::make_shared<Graph<int, int>>("MMGraph"),
      outerGraph = std::make_shared<Graph<int, int>>("MMGraph");

  std::shared_ptr<MyStaticTask> staticTask = nullptr;
  std::shared_ptr<StaticMM<int>> staticMM = nullptr;

  std::shared_ptr<MyCUDATask> cudaTask = nullptr;
  std::shared_ptr<CudaMM<int>> cudaMM = nullptr;

  std::shared_ptr<MyDynamicTask> dynamicTask = nullptr;
  std::shared_ptr<DynamicMM<int>> dynMM = nullptr;

  std::shared_ptr<OutputTask> outputTask = nullptr;

  size_t count = 0;
  std::shared_ptr<int> result = nullptr;

  outputTask = std::make_shared<OutputTask>();
  insideGraph->output(outputTask);

  staticTask = std::make_shared<MyStaticTask>();
  staticMM = std::make_shared<StaticMM<int>>(2);
  staticTask->connectMemoryManager(std::static_pointer_cast<AbstractMemoryManager<MatrixData<int>>>(staticMM));
  insideGraph->input(staticTask);
  insideGraph->addEdge(staticTask, outputTask);

  cudaTask = std::make_shared<MyCUDATask>();
  cudaMM = std::make_shared<CudaMM<int>>(2);
  cudaTask->connectMemoryManager(std::static_pointer_cast<AbstractMemoryManager<MatrixData<int>>>(cudaMM));
  insideGraph->input(cudaTask);
  insideGraph->addEdge(cudaTask, outputTask);

  dynamicTask = std::make_shared<MyDynamicTask>();
  dynMM = std::make_shared<DynamicMM<int>>(2);
  dynamicTask->connectMemoryManager(std::static_pointer_cast<AbstractMemoryManager<DynamicMatrixData<int>>>(dynMM));
  insideGraph->input(dynamicTask);
  insideGraph->addEdge(dynamicTask, outputTask);

  iiep = std::make_shared<IIEP>(insideGraph, 5, vDevices, false);

  outerGraph->input(iiep);
  outerGraph->output(iiep);

  outerGraph->executeGraph();

  for (int i = 0; i < 100; ++i) {
    outerGraph->pushData(std::make_shared<int>(i));
  }

  outerGraph->finishPushingData();

  while ((result = outerGraph->getBlockingResult())) { ++count; }
  std::cout << "Received " << std::dec << count << " elements!" << std::endl;

  outerGraph->waitForTermination();
  outerGraph->createDotFile("outputAfterWait.dot", ColorScheme::EXECUTION, StructureOptions::ALL);
}