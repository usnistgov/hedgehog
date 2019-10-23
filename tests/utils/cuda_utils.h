//
// Created by anb22 on 9/25/19.
//

#ifndef HEDGEHOG_TESTS_CUDA_UTILS_H
#define HEDGEHOG_TESTS_CUDA_UTILS_H

#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

// Utils macro to convert row based index to column based
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

#endif //HEDGEHOG_TESTS_CUDA_UTILS_H
