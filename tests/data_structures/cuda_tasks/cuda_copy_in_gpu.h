//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_CUDA_COPY_IN_GPU_H
#define HEDGEHOG_TESTS_CUDA_COPY_IN_GPU_H

#include "hedgehog/hedgehog.h"
#include "../../data_structures/datas/data_type.h"
#include "../../data_structures/datas/cuda_matrix_block_data.h"
#include "../../utils/cuda_utils.h"

template<class MatrixType, char Id>
class CudaCopyInGpu : public hh::AbstractCUDATask<CudaMatrixBlockData<MatrixType, Id>,
                                                  MatrixBlockData<MatrixType, Id, Order::Column>> {
 private:
  size_t
      blockTTL_ = 0,
      blockSize_ = 0,
      matrixLeadingDimension_ = 0;

 public:
  CudaCopyInGpu(size_t blockTTL, size_t blockSize, size_t matrixLeadingDimension)
      : hh::AbstractCUDATask<CudaMatrixBlockData<MatrixType, Id>, MatrixBlockData<MatrixType, Id, Order::Column>>
            ("Copy In GPU", 1, false, false),
        blockTTL_(blockTTL),
        blockSize_(blockSize),
        matrixLeadingDimension_(matrixLeadingDimension) {}

  virtual ~CudaCopyInGpu() = default;

  void execute(std::shared_ptr<MatrixBlockData<MatrixType, Id, Order::Column>> ptr) override {
    std::shared_ptr<CudaMatrixBlockData<MatrixType, Id>> block = this->getManagedMemory();
    block->rowIdx(ptr->rowIdx());
    block->colIdx(ptr->colIdx());
    block->blockSizeHeight(ptr->blockSizeHeight());
    block->blockSizeWidth(ptr->blockSizeWidth());
    block->leadingDimension(block->blockSizeHeight());
    block->fullMatrixData(ptr->fullMatrixData());
    block->ttl(blockTTL_);

    if (ptr->leadingDimension() == block->leadingDimension()) {
      hh::checkCudaErrors(cudaMemcpyAsync(block->blockData(), ptr->blockData(),
                                          sizeof(MatrixType) * block->blockSizeHeight() * block->blockSizeWidth(),
                                          cudaMemcpyHostToDevice, this->stream()));
    } else {
      cublasSetMatrixAsync(
          (int) block->blockSizeHeight(), (int) block->blockSizeWidth(), sizeof(MatrixType),
          block->fullMatrixData()
              + IDX2C(block->rowIdx() * blockSize_, block->colIdx() * blockSize_, matrixLeadingDimension_),
          (int) matrixLeadingDimension_, block->blockData(), (int) block->leadingDimension(), this->stream());
    }

    hh::checkCudaErrors(cudaStreamSynchronize(this->stream()));
    this->addResult(block);
  }

  std::shared_ptr<hh::AbstractTask<CudaMatrixBlockData<MatrixType, Id>,
                                   MatrixBlockData<MatrixType, Id, Order::Column>>> copy() override {
    return std::make_shared<CudaCopyInGpu>(blockTTL_, blockSize_, matrixLeadingDimension_);
  }
};

#endif //HEDGEHOG_TESTS_CUDA_COPY_IN_GPU_H
