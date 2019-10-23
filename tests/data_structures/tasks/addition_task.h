//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_ADDITION_TASK_H
#define HEDGEHOG_TESTS_ADDITION_TASK_H

#include "../../../hedgehog/hedgehog.h"
#include "../datas/matrix_block_data.h"

template<class Type, Order Ord = Order::Row>
class AdditionTask : public hh::AbstractTask<
    MatrixBlockData<Type, 'c', Ord>,
    std::pair<
        std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>,
        std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>
    >> {

 public:
  explicit AdditionTask(size_t numberThreads) :
      hh::AbstractTask<
          MatrixBlockData<Type, 'c', Ord>,
          std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>, std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>
          >>("Addition Task", numberThreads) {}

  virtual ~AdditionTask() = default;

 public:
  void execute(std::shared_ptr<std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>,
                                         std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>> ptr) override {
    auto c = ptr->first;
    auto p = ptr->second;
    assert(c->blockSizeWidth() == p->blockSizeWidth());
    assert(c->blockSizeHeight() == p->blockSizeHeight());

    if constexpr (Ord == Order::Row) {
      for (size_t i = 0; i < c->blockSizeHeight(); ++i) {
        for (size_t j = 0; j < c->blockSizeWidth(); ++j) {
          c->blockData()[i * c->leadingDimension() + j] += p->blockData()[i * p->leadingDimension() + j];
        }
      }
    } else {
      for (size_t j = 0; j < c->blockSizeWidth(); ++j) {
        for (size_t i = 0; i < c->blockSizeHeight(); ++i) {
          c->blockData()[j * c->leadingDimension() + i] += p->blockData()[j * p->leadingDimension() + i];
        }
      }
    }

    delete[] p->blockData();
    this->addResult(c);
  }

  std::shared_ptr<hh::AbstractTask<MatrixBlockData<Type, 'c', Ord>,
                                   std::pair<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>,
                                         std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>>> copy() override {
    return std::make_shared<AdditionTask>(this->numberThreads());
  }
};

#endif //HEDGEHOG_TESTS_ADDITION_TASK_H
