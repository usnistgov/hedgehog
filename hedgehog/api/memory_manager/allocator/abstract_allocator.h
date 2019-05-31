//
// Created by anb22 on 5/24/19.
//

#ifndef HEDGEHOG_ABSTRACT_ALLOCATOR_H
#define HEDGEHOG_ABSTRACT_ALLOCATOR_H

#include <cstdio>

template<class Data>
class AbstractAllocator {
 public:
  virtual ~AbstractAllocator() = default;
  virtual void initialize() = 0;
  virtual Data *allocate(size_t) = 0;
  virtual void deallocate(Data *) = 0;
};

#endif //HEDGEHOG_ABSTRACT_ALLOCATOR_H
