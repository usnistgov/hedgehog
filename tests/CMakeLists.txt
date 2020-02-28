# NIST-developed software is provided by NIST as a public service.
# You may use, copy and distribute copies of the  software in any  medium,
# provided that you keep intact this entire notice. You may improve,
# modify and create derivative works of the software or any portion of the
# software, and you may copy and distribute such modifications or works.
# Modified works should carry a notice stating that you changed the software
# and should note the date and nature of any such change. Please explicitly
# acknowledge the National Institute of Standards and Technology as the
# source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY
# OF ANY KIND, EXPRESS, IMPLIED, IN FACT  OR ARISING BY OPERATION OF LAW,
# INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST
# NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION  OF THE SOFTWARE WILL
# BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST
# DOES NOT WARRANT  OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE
# SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE
# CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using
# and distributing the software and you assume  all risks associated with
# its use, including but not limited to the risks and costs of program
# errors, compliance  with applicable laws, damage to or loss of data,
# programs or equipment, and the unavailability or interruption of operation.
# This software is not intended to be used in any situation where a failure
# could cause risk of injury or damage to property. The software developed
# by NIST employees is not subject to copyright protection within
# the United States.

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
link_libraries(${CMAKE_THREAD_LIBS_INIT})

add_definitions(-D_GLIBCXX_ASSERTIONS)

# Add Compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -pedantic -pthread")
endif (NOT MSVC)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_COMPILER_IS_CLANGXX 1)
    set(use-libclang ON)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

if (NOT APPLE AND NOT CMAKE_COMPILER_IS_CLANGXX AND NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi")
endif (NOT APPLE AND NOT CMAKE_COMPILER_IS_CLANGXX AND NOT MSVC)

# Set specific flags for filesystem [experimental] library
if (CMAKE_COMPILER_IS_CLANGXX)
    string(REGEX REPLACE bin.* "" CLANG_FOLDER ${CMAKE_CXX_COMPILER})
    string(CONCAT FS_LIB_PATH ${CLANG_FOLDER} "lib/")
    message("Try to find Filesystem library in " ${FS_LIB_PATH})
    link_directories(${FS_LIB_PATH})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstandalone-debug")

    if (APPLE)
        link_libraries(c++fs)
    else (APPLE)
        link_libraries(stdc++fs)
    endif (APPLE)
elseif (NOT MSVC)
    link_libraries(stdc++fs)
endif (CMAKE_COMPILER_IS_CLANGXX)

set(TEST_SRC
        ${HEDGEHOG_TESTS_SRC}
        testAll.cc
        tests/test_memory_manager.h
        tests/test_small_graph.h
        tests/test_complex_graph.h
        tests/test_cuda.h
        tests/test_execution_pipeline.h
        tests/test_execution_pipeline_composition.h
        tests/test_cycles.h
        tests/test_simple_partial_input.h
        data_structures/tasks/int_double_char_to_float.h
        data_structures/tasks/int_float_to_double.h
        data_structures/tasks/int_float_to_int.h
        data_structures/tasks/int_double_float.h
        data_structures/tasks/int_to_int.h
        data_structures/execution_pipelines/execution_pipeline_int_double_to_int.h
        data_structures/states/state_int_float_to_int.h
        data_structures/cuda_tasks/cuda_copy_in_gpu.h
        data_structures/cuda_tasks/cuda_copy_out_gpu.h
        data_structures/cuda_tasks/cuda_product_task.h
        data_structures/datas/cuda_matrix_block_data.h
        data_structures/datas/data_type.h
        data_structures/datas/matrix_block_data.h
        data_structures/datas/matrix_data.h
        utils/cuda_utils.h
        data_structures/states/cuda_input_block_state.h
        data_structures/states/output_state.h
        data_structures/states/partial_computation_state.h
        data_structures/state_managers/partial_computation_state_manager.h
        data_structures/tasks/addition_task.h
        data_structures/tasks/matrix_column_traversal_task.h
        data_structures/tasks/matrix_row_traversal_task.h
        data_structures/tasks/float_to_int_cycle.h
        data_structures/execution_pipelines/execution_pipeline_int_to_int.h
        data_structures/execution_pipelines/execution_pipeline_int_to_static_memory_managed_int.h
        data_structures/datas/static_memory_manage_data.h
        data_structures/datas/dynamic_memory_manage_data.h
        data_structures/memory_managers/dynamic_memory_manager.h
        data_structures/tasks/int_to_static_memory_managed_int.h
        data_structures/tasks/int_to_dynamic_memory_managed_int.h
        data_structures/cuda_tasks/cuda_int_to_memory_managed_int.h
        data_structures/tasks/static_memory_managed_int_dynamic_memory_managed_int_to_static_memory_managed_int.h
        data_structures/cuda_tasks/cuda_link_example.cpp
        data_structures/cuda_tasks/cuda_link_example.h
        tests/test_link.cpp
        tests/test_link.h
        tests/test_link2.cpp
        tests/test_link2.h
        data_structures/cuda_tasks/cuda_link2_example.cpp
        data_structures/cuda_tasks/cuda_link2_example.h)

if (CUDA_FOUND)
    find_package(NVTX QUIET)
    cuda_add_executable(runAllTests ${TEST_SRC})
    target_compile_definitions(runAllTests PUBLIC -DHH_USE_CUDA)
    target_link_libraries(runAllTests ${CUDA_LIBRARIES})
    target_include_directories(runAllTests PUBLIC ${CUDA_INCLUDE_DIR} cuda)
    CUDA_ADD_CUBLAS_TO_TARGET(runAllTests)

    if (NVTX_FOUND)
        target_include_directories(runAllTests PUBLIC ${NVTX_INCLUDE_DIR})
        target_link_libraries(runAllTests ${NVTX_LIBRARIES})
        target_compile_definitions(runAllTests PUBLIC -DHH_USE_NVTX)
    endif()

else (CUDA_FOUND)
    add_executable(runAllTests ${TEST_SRC})
endif ()

target_include_directories(runAllTests PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(runAllTests ${GTEST_LIBRARY})


add_dependencies(runAllTests googletest)
