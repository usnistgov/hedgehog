//
// Created by tjb3 on 2/28/20.
//

#ifndef HEDGEHOG_CUDA_DEBUGGING_H
#define HEDGEHOG_CUDA_DEBUGGING_H
#include <iostream>
#include <cublas.h>
#include <cuda_runtime.h>

#ifndef checkCudaErrors
/// @brief Inline helper function for all of the SDK helper functions, to catch and show CUDA Error, in case of error,
/// the device is reset (cudaDeviceReset) and the program exit
/// @param err Error to manage
/// @param file File generating the error
/// @param line File's line generating the error
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    std::cerr << "checkCudaErrors() Cuda error = "
              << err
              << "\"" << cudaGetErrorString(err) << " \" from "
              << file << ":" << line << std::endl;
    exit(43);
  }
}

/// @brief Inline helper function for all of the SDK helper functions, to catch and show CUDA Status, in case of error,
/// the device is reset (cudaDeviceReset) and the program exit
/// @param status Status to manage
/// @param file File generating the error
/// @param line File's line generating the error
inline void __checkCudaErrors(cublasStatus_t status, const char *file, const int line) {
  if (CUBLAS_STATUS_SUCCESS != status) {
    std::cerr << "checkCudaErrors() Status Error = "
              << status << " from "
              << file << ":" << line << std::endl;
    exit(44);
  }
}
#ifndef HH_DISABLE_CHECK_CUDA
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
#else
#define checkCudaErrors(err) err
#endif

#endif
#endif //HEDGEHOG_CUDA_DEBUGGING_H
