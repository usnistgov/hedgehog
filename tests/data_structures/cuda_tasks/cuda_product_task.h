//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_CUDA_PRODUCT_TASK_H
#define HEDGEHOG_TESTS_CUDA_PRODUCT_TASK_H

#include <cuda.h>
#include <cublas.h>
#include "../datas/cuda_matrix_block_data.h"
#include "../../../hedgehog/hedgehog.h"

template<class Type>
class CudaProductTask : public hh::AbstractCUDATask<
    CudaMatrixBlockData<Type, 'p'>,
    std::pair<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>, std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>> {
 private:
  size_t
      countPartialComputation_ = 0;
 private:
  cublasHandle_t
      handle_ = {};

 public:
  explicit CudaProductTask(size_t countPartialComputation, size_t numberThreadsProduct = 1)
      : hh::AbstractCUDATask<
      CudaMatrixBlockData<Type, 'p'>,
      std::pair<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>, std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>
  >("CUDA Product Task", numberThreadsProduct, false, false),
        countPartialComputation_(countPartialComputation) {}
  virtual ~CudaProductTask() = default;

  void initializeCuda() override {
    hh::checkCudaErrors(cublasCreate_v2(&handle_));
    hh::checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
  }

  void shutdownCuda() override {
    hh::checkCudaErrors(cublasDestroy_v2(handle_));
  }
  void execute(std::shared_ptr<
      std::pair<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>, std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>
      >> ptr) override {
    Type
        alpha = 1.,
        beta = 0.;
    auto matA = ptr->first;
    auto matB = ptr->second;
    auto res = this->getManagedMemory();

    res->rowIdx(matA->rowIdx());
    res->colIdx(matB->colIdx());
    res->blockSizeHeight(matA->blockSizeHeight());
    res->blockSizeWidth(matB->blockSizeWidth());
    res->leadingDimension(matA->blockSizeHeight());
    res->ttl(1);

    if constexpr(std::is_same<Type, float>::value) {
      hh::checkCudaErrors(
          cublasSgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), &alpha,
                         (float *) matA->blockData(), matA->leadingDimension(),
                         (float *) matB->blockData(), matB->leadingDimension(), &beta,
                         (float *) res->blockData(), res->leadingDimension())
      );
    } else if (std::is_same<Type, double>::value) {
      hh::checkCudaErrors(
          cublasDgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), &alpha,
                         (double *) matA->blockData(), matA->leadingDimension(),
                         (double *) matB->blockData(), matB->leadingDimension(), &beta,
                         (double *) res->blockData(), res->leadingDimension())
      );
    } else {
      std::cerr << "The matrix can't be multiplied" << std::endl;
      exit(43);
    }
    hh::checkCudaErrors(cudaStreamSynchronize(this->stream()));

    matA->returnToMemoryManager();
    matB->returnToMemoryManager();
    this->addResult(res);
  }
  std::shared_ptr<hh::AbstractTask<CudaMatrixBlockData<Type, 'p'>,
                                   std::pair<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>,
                                         std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>>> copy() override {
    return std::make_shared<CudaProductTask>(countPartialComputation_, this->numberThreads());
  }
};

#endif //HEDGEHOG_TESTS_CUDA_PRODUCT_TASK_H
