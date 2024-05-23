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

#ifndef HEDGEHOG_HEDGEHOG_H
#define HEDGEHOG_HEDGEHOG_H

#include <version>

#ifdef HH_ENABLE_HH_CX
#if !__cpp_lib_constexpr_string || !__cpp_lib_constexpr_vector
#error The compiler has not the features for Hedgehog CX
#undef HH_ENABLE_HH_CX
#endif // !__cpp_lib_constexpr_string || !__cpp_lib_constexpr_vector
#endif // HH_ENABLE_HH_CX

#include "src/api/task/abstract_task.h"
#include "src/api/task/abstract_mixed_task.h"
#include "src/api/task/abstract_atomic_task.h"
#include "src/api/task/abstract_limited_atomic_task.h"
#include "src/api/graph/graph.h"
#include "src/api/graph/graph_signal_handler.h"
#include "src/api/execution_pipeline/abstract_execution_pipeline.h"

#include "src/api/memory_manager/manager/memory_manager.h"
#include "src/api/memory_manager/managed_memory.h"
#include "src/api/memory_manager/manager/static_memory_manager.h"

#include "src/api/state_manager/state_manager.h"

#ifdef HH_USE_CUDA
#include "src/api/task/abstract_cuda_task.h"
#include "src/tools/cuda_debugging.h"
#endif //HH_USE_CUDA

#ifdef HH_ENABLE_HH_CX
#include "hedgehog_cx/hedgehog_cx.h"
#endif //HH_ENABLE_HH_CX

#endif //HEDGEHOG_HEDGEHOG_H
