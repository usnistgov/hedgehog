
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

#ifndef HEDGEHOG_CUDADEBUGGING_H
#define HEDGEHOG_CUDADEBUGGING_H

#ifdef HH_USE_CUDA

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
    std::cerr
        << "checkCudaErrors() Cuda error = "
        << err
        << " \"" << cudaGetErrorString(err) << "\" from "
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
    std::cerr
        << "checkCudaErrors() Status Error = "
        << status << " from "
        << file << ":" << line << std::endl;
    exit(44);
  }
}

#ifdef HH_ENABLE_CHECK_CUDA
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
#else //HH_ENABLE_CHECK_CUDA
#define checkCudaErrors(err) err
#endif //HH_ENABLE_CHECK_CUDA

#endif //checkCudaErrors

#endif //HH_USE_CUDA

#endif //HEDGEHOG_CUDADEBUGGING_H
