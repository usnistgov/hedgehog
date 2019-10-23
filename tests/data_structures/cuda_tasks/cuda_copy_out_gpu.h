//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_CUDA_COPY_OUT_GPU_H
#define HEDGEHOG_TESTS_CUDA_COPY_OUT_GPU_H

template<class MatrixType>
class CudaCopyOutGpu
    : public hh::AbstractCUDATask<MatrixBlockData<MatrixType, 'p', Order::Column>,
                                  CudaMatrixBlockData<MatrixType, 'p'>> {
 private:
  size_t blockSize_ = 0;
 public:
  explicit CudaCopyOutGpu(size_t blockSize)
      : hh::AbstractCUDATask<MatrixBlockData<MatrixType, 'p', Order::Column>, CudaMatrixBlockData<MatrixType, 'p'>>
            ("Copy Out GPU", 1, false, false), blockSize_(blockSize) {}
  virtual ~CudaCopyOutGpu() = default;

  void execute(std::shared_ptr<CudaMatrixBlockData<MatrixType, 'p'>> ptr) override {
    auto ret = ptr->convertToCPUMemory(this->stream());
    ptr->returnToMemoryManager();
    this->addResult(ret);
  }

  std::shared_ptr<hh::AbstractTask<MatrixBlockData<MatrixType, 'p', Order::Column>,
                                   CudaMatrixBlockData<MatrixType, 'p'>>> copy() override {
    return std::make_shared<CudaCopyOutGpu>(this->blockSize_);
  }

};

#endif //HEDGEHOG_TESTS_CUDA_COPY_OUT_GPU_H
