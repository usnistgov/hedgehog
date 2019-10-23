//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_MATRIX_COLUMN_TRAVERSAL_TASK_H
#define HEDGEHOG_TESTS_MATRIX_COLUMN_TRAVERSAL_TASK_H

#include "../../../hedgehog/hedgehog.h"
#include "../datas/data_type.h"
#include "../datas/matrix_data.h"
#include "../datas/matrix_block_data.h"

template<class Type, char Id = '0', Order Ord = Order::Row>
class MatrixColumnTraversalTask : public hh::AbstractTask<MatrixBlockData<Type, Id, Ord>, MatrixData<Type, Id, Ord>> {
 public:
  MatrixColumnTraversalTask() : hh::AbstractTask<MatrixBlockData<Type, Id, Ord>, MatrixData<Type, Id, Ord>>
                                    ("ColumnTraversal") {}
  virtual ~MatrixColumnTraversalTask() = default;

  void execute(std::shared_ptr<MatrixData<Type, Id, Ord>> ptr) override {
    for (size_t jGrid = 0; jGrid < ptr->numBlocksCols(); ++jGrid) {
      for (size_t iGrid = 0; iGrid < ptr->numBlocksRows(); ++iGrid) {
        if constexpr (Ord == Order::Row) {
          this->addResult(
              std::make_shared<MatrixBlockData<Type, Id, Ord>>(
                  iGrid, jGrid,
                  std::min(ptr->blockSize(), ptr->matrixHeight() - (iGrid * ptr->blockSize())),
                  std::min(ptr->blockSize(), ptr->matrixWidth() - (jGrid * ptr->blockSize())),
                  ptr->leadingDimension(),
                  ptr->matrixData(),
                  ptr->matrixData() + (iGrid * ptr->blockSize()) * ptr->leadingDimension() + jGrid * ptr->blockSize()
              )
          );
        } else {
          this->addResult(
              std::make_shared<MatrixBlockData<Type, Id, Ord>>(
                  iGrid, jGrid,
                  std::min(ptr->blockSize(), ptr->matrixHeight() - (iGrid * ptr->blockSize())),
                  std::min(ptr->blockSize(), ptr->matrixWidth() - (jGrid * ptr->blockSize())),
                  ptr->leadingDimension(),
                  ptr->matrixData(),
                  ptr->matrixData() + (jGrid * ptr->blockSize()) * ptr->leadingDimension() + iGrid * ptr->blockSize()
              )
          );
        }
      }
    }
  }
};

#endif //HEDGEHOG_TESTS_MATRIX_COLUMN_TRAVERSAL_TASK_H
