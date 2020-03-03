// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.


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
    checkCudaErrors(cublasCreate_v2(&handle_));
    checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
  }

  void shutdownCuda() override {
    checkCudaErrors(cublasDestroy_v2(handle_));
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
      checkCudaErrors(
          cublasSgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                         matA->blockSizeHeight(), matB->blockSizeWidth(), matA->blockSizeWidth(), &alpha,
                         (float *) matA->blockData(), matA->leadingDimension(),
                         (float *) matB->blockData(), matB->leadingDimension(), &beta,
                         (float *) res->blockData(), res->leadingDimension())
      );
    } else if (std::is_same<Type, double>::value) {
      checkCudaErrors(
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
    checkCudaErrors(cudaStreamSynchronize(this->stream()));

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
