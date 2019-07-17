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

set(TEST_SRC
        testAll.cc
        graphs/big_example/types/a.h
        graphs/big_example/types/b.h
        graphs/big_example/types/c.h
        graphs/big_example/tasks/input_task.h
        graphs/big_example/tasks/input_task_2.h
        graphs/big_example/tasks/output_task.h
        graphs/big_example/tasks/output_task_2.h
        graphs/big_example/tasks/task_bda.h
        graphs/big_example/tasks/task_ic.h
        graphs/big_example/states/test_state.h
        graphs/big_example/ep/inner_ep.h
        graphs/cycle/states/my_state.h
        graphs/cycle/tasks/my_task.h
        graphs/cycle/tasks/my_task_2.h
        graphs/ep/ep/if_to_iep.h
        graphs/ep/tasks/if_to_i.h
        graphs/ep/tasks/if_to_i_ep_in.h
        graphs/ep/tasks/epif_to_i.h
        graphs/ep_composition/tasks/ii_task.h
        graphs/ep_composition/ep/iiep.h
        graphs/memory_manager/data/matrix_data.h
        graphs/memory_manager/data/dynamic_matrix_data.h
        graphs/memory_manager/memory_manager/static_mm.h
        graphs/memory_manager/memory_manager/dynamic_mm.h
        graphs/memory_manager/memory_manager/cuda_mm.h
        graphs/memory_manager/task/my_cuda_task.h
        graphs/memory_manager/task/my_static_task.h
        graphs/memory_manager/task/my_dynamic_task.h
        graphs/memory_manager/task/output_md_task.h
        graphs/memory_manager/ep/iiep_mm.h
        graphs/small_example/tasks/ito_f.h
        graphs/partial_input/task/task_ii_partial_input.h graphs/partial_input/task/task_fi_partial_input.h graphs/partial_input/ep/iifep_partial_input.h graphs/partial_input/test_partial_input.h)


if (CUDA_FOUND)
    cuda_add_executable(runAllTests ${TEST_SRC})
    target_compile_definitions(runAllTests PUBLIC -DHH_USE_CUDA)
else (CUDA_FOUND)
    add_executable(runAllTests ${TEST_SRC})
endif ()

target_link_libraries(runAllTests gtest gtest_main)
