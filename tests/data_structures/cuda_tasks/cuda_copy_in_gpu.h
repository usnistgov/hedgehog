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
