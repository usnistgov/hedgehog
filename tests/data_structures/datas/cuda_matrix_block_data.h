//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_CUDA_MATRIX_BLOCK_DATA_H
#define HEDGEHOG_TESTS_CUDA_MATRIX_BLOCK_DATA_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "../../../hedgehog/hedgehog.h"
#include "matrix_block_data.h"
#include "data_type.h"

template<class Type, char Id>
class CudaMatrixBlockData : public MatrixBlockData<Type, Id, Order::Column>,
                            public hh::MemoryData<CudaMatrixBlockData<Type, Id>> {
 private:
  size_t
      ttl_ = 0;

  cudaEvent_t
      event_ = {};

 public:
  CudaMatrixBlockData() = default;

  explicit CudaMatrixBlockData(size_t blockSize) {
    hh::checkCudaErrors(cudaMalloc((void **) this->adressBlockData(), sizeof(Type) * blockSize * blockSize));
  }

  CudaMatrixBlockData(
      size_t rowIdx, size_t colIdx,
      size_t blockSizeHeight, size_t blockSizeWidth, size_t leadingDimension,
      Type *fullMatrixData, Type *blockData)
      : MatrixBlockData<Type, Id, Order::Column>(
      rowIdx, colIdx,
      blockSizeHeight, blockSizeWidth, leadingDimension,
      fullMatrixData, blockData) {}

  template<char OldId>
  explicit CudaMatrixBlockData(CudaMatrixBlockData<Type, OldId> &o) {
    this->rowIdx_ = o.rowIdx_;
    this->colIdx_ = o.colIdx_;
    this->blockSizeHeight_ = o.blockSizeHeight_;
    this->blockSizeWidth_ = o.blockSizeWidth_;
    this->leadingDimension_ = o.leadingDimension_;
    this->fullMatrixData_ = o.fullMatrixData_;
    this->blockData_ = o.blockData_;
  }

  template<char OldId>
  explicit CudaMatrixBlockData(std::shared_ptr<CudaMatrixBlockData<Type, OldId>> &o) {
    this->rowIdx_ = o->getRowIdx();
    this->colIdx_ = o->getColIdx();
    this->blockSizeHeight_ = o->getBlockSizeHeight();
    this->blockSizeWidth_ = o->getBlockSizeWidth();
    this->leadingDimension_ = o->getLeadingDimension();
    this->fullMatrixData_ = o->getFullMatrixData();
    this->blockData_ = o->getBlockData();
  }

  virtual ~CudaMatrixBlockData() { hh::checkCudaErrors(cudaFree(this->blockData())); }

  Type **adressBlockData() { return &this->blockData_; }

  void recordEvent(cudaStream_t stream) {
    cudaEventRecord(event_, stream);
  }

  void synchronizeEvent() {
    cudaEventSynchronize(event_);
  }

  std::shared_ptr<MatrixBlockData<Type, Id, Order::Column>> convertToCPUMemory(cudaStream_t stream) {
    auto res = std::make_shared<MatrixBlockData<Type, Id, Order::Column>>();
    res->rowIdx(this->rowIdx());
    res->colIdx(this->colIdx());
    res->blockSizeHeight(this->blockSizeHeight());
    res->blockSizeWidth(this->blockSizeWidth());
    res->leadingDimension(this->leadingDimension());
    res->blockData(new Type[res->blockSizeWidth() * res->blockSizeHeight()]());
    res->fullMatrixData(res->blockData());
    hh::checkCudaErrors(
        cudaMemcpyAsync(res->blockData(), this->blockData(),
                        res->blockSizeHeight() * res->blockSizeWidth() * sizeof(Type), cudaMemcpyDeviceToHost, stream));
    hh::checkCudaErrors(cudaStreamSynchronize(stream));
    return res;
  }

  void ttl(size_t ttl) { ttl_ = ttl; }
  void used() override { --this->ttl_; }
  bool canBeRecycled() override { return this->ttl_ == 0; }

  friend std::ostream &operator<<(std::ostream &os, CudaMatrixBlockData const &data) {
    os << "CudaMatrixBlockData " << Id << " position Grid: (" << data.rowIdx_ << ", " << data.colIdx_ << ")"
       << std::endl;
    os << "Block: (" << data.blockSizeHeight() << ", " << data.blockSizeWidth() << ") leadingDimension="
       << data.leadingDimension_
       << " ttl: " << data.ttl_ << std::endl;
    return os;
  }
};

#endif //HEDGEHOG_TESTS_CUDA_MATRIX_BLOCK_DATA_H
