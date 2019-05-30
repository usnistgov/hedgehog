cmake_minimum_required(VERSION 3.13)
project(hedgehog)


add_definitions(-D_GLIBCXX_ASSERTIONS)

# Ensure C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_COMPILER_IS_CLANGXX 1)
    set(use-libclang ON)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif (MSVC)

#Options
option(LOG "Activate logging" OFF)

if (LOG)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
    find_package(GLOG REQUIRED)
    include_directories(${GLOG_INCLUDE_DIRS})
    link_libraries(${GLOG_LIBRARIES})
    message(STATUS "GLOG imported")
    add_definitions(-DHLOG_ENABLED)
    add_definitions(-DHLOG_LEVEL=0)
endif (LOG)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Set targets
option(BUILD_DOXYGEN "Creates the doxygen documentation of the API" OFF)
option(RUN_GTEST "Downloads google unit test API and runs google test scripts to test Fast Image core and api" OFF)
option(BUILD_MAIN "Compiles main function for testing changes to API" OFF)

set(BUILD_DOXYGEN ON)
set(RUN_GTEST ON)
set(BUILD_MAIN ON)

# Cuda
find_package(CUDA REQUIRED)

if (CUDA_FOUND)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS_RELEASE -O3; -DNDEBUG)
endif (CUDA_FOUND)

if (RUN_GTEST)
    # Download and unpack googletest at configure time
    configure_file(tests/CMakeLists.txt.gtest googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download)
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download)

    # Prevent GoogleTest from overriding our compiler/linker options
    # when building with Visual Studio
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This adds
    # the following targets: gtest, gtest_main, gmock
    # and gmock_main
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
            ${CMAKE_BINARY_DIR}/googletest-build)

    # The gtest/gmock targets carry header search path
    # dependencies automatically when using CMake 2.8.11 or
    # later. Otherwise we have to add them here ourselves.
    if (CMAKE_VERSION VERSION_LESS 2.8.11)
        include_directories("${gtest_SOURCE_DIR}/include")
    endif ()
    #	enable_testing()
    add_subdirectory(tests)

endif (RUN_GTEST)


if (BUILD_DOXYGEN)
    find_package(Doxygen)

    if (DOXYGEN_FOUND)
        add_definitions(-DPROFILE)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen/hedgehog.cfg ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc ALL
                ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API documentation with Doxygen" VERBATIM
                )

        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc/Hedgehog)

        remove_definitions(-DPROFILE)
    endif (DOXYGEN_FOUND)
endif (BUILD_DOXYGEN)

# Add Compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Werror -pedantic -pthread")
endif (NOT MSVC)

if (NOT APPLE AND NOT CMAKE_COMPILER_IS_CLANGXX AND NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi")
endif (NOT APPLE AND NOT CMAKE_COMPILER_IS_CLANGXX AND NOT MSVC)

# Set specific flags for filesystem [experimental] library
if (CMAKE_COMPILER_IS_CLANGXX)
    link_directories(/usr/local/Cellar/llvm/8.0.0/lib/)
    link_libraries(c++fs)
elseif (NOT MSVC)
    link_libraries(stdc++fs)
endif (CMAKE_COMPILER_IS_CLANGXX)


# Set main executable to be moved to a specific target
add_executable(main
        main.cpp

        src/hedgehog.h

        src/api/graph.h
        src/api/abstract_task.h
        src/api/abstract_state_manager.h
        src/api/abstract_state.h

        src/behaviour/execute.h
        src/core/node/core_node.h
        src/behaviour/threadable.h
        src/behaviour/node.h
        src/behaviour/io/sender.h
        src/behaviour/io/multi_receivers.h

        src/core/scheduler/abstract_scheduler.h
        src/core/node/core_task.h
        src/core/node/core_graph.h
        src/core/io/base/sender/core_sender.h
        src/core/io/base/sender/core_notifier.h
        src/core/io/task/sender/core_task_sender.h
        src/core/io/task/sender/core_task_notifier.h
        src/core/io/graph/sender/core_graph_source.h
        src/core/io/base/receiver/core_receiver.h
        src/core/io/base/receiver/core_multi_receivers.h
        src/core/io/base/receiver/core_slot.h
        src/core/io/task/receiver/core_task_receiver.h
        src/core/io/task/receiver/core_task_multi_receivers.h
        src/core/io/task/receiver/core_task_slot.h
        src/core/io/graph/receiver/core_graph_multi_receivers.h
        src/core/io/graph/receiver/core_graph_receiver.h
        src/core/io/graph/receiver/core_graph_sink.h

        src/tools/helper.h
        src/tools/traits.h
        src/tools/graph_signal_handler.h
        src/tools/printers/abstract_printer.h
        src/tools/printers/dot_printer.h
        src/tools/scheduler/default_scheduler.h src/api/memory_manager/memory/managed_memory.h src/api/memory_manager/memory_manager/abstract_memory_manager.h src/api/memory_manager/memory_manager/static_memory_manager.h src/api/memory_manager/memory_manager/dynamic_memory_manager.h src/tools/data_structure/pool.h src/api/memory_manager/allocator/abstract_allocator.h src/api/memory_manager/allocator/abstract_cuda_allocator.h src/api/memory_manager/memory/abstract_release_rule.h src/api/memory_manager/receiver/managed_memory_abstract_task.h src/api/execution_pipeline/execution_pipeline.h src/api/execution_pipeline/execution_pipeline_rule.h src/api/execution_pipeline/execution_pipeline_switch.h)
