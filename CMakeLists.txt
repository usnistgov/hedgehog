cmake_minimum_required(VERSION 3.10)
project(hedgehog VERSION 1.0.3 LANGUAGES CXX)

include(ExternalProject)

# Ensure C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif (MSVC)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Create version.h file in api folder
string(TIMESTAMP DATE "%m/%d/%Y")
file(WRITE hedgehog/version.h "#include <string> \n\n")
file(APPEND hedgehog/version.h "/// @file version.h\n")
file(APPEND hedgehog/version.h "/// @author Alexandre Bardakoff - Timothy Blattner\n")
file(APPEND hedgehog/version.h "/// @date ${DATE}\n")
file(APPEND hedgehog/version.h "/// @brief Hedgehog version\n\n")
file(APPEND hedgehog/version.h "namespace hh{\n")
file(APPEND hedgehog/version.h "/// \\namespace hh hedgehog namespace\n\n")
file(APPEND hedgehog/version.h "// hedgehog version number\n")
file(APPEND hedgehog/version.h "\tconst int MAJOR_VERSION = ${PROJECT_VERSION_MAJOR}; ///< Hedgehog major version\n")
file(APPEND hedgehog/version.h "\tconst int MINOR_VERSION = ${PROJECT_VERSION_MINOR}; ///< Hedgehog minor version\n")
file(APPEND hedgehog/version.h "\tconst int PATCH_VERSION = ${PROJECT_VERSION_PATCH}; ///< Hedgehog patch version\n")
file(APPEND hedgehog/version.h "\tconst std::string FULL_VERSION = \"${PROJECT_VERSION}\"; ///< Hedgehog full version\n")
file(APPEND hedgehog/version.h "}\n")


install(DIRECTORY hedgehog
        DESTINATION include
        COMPONENT headers)

# Set targets
option(LOG "Activate logging with GLOG" OFF)
option(BUILD_DOXYGEN "Creates the doxygen documentation of the API" OFF)
option(RUN_GTEST "Downloads google unit test API and runs google test scripts to test Hedgehog core and api" ON)
option(BUILD_MAIN "Compiles main function for testing changes to API" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")


# Cuda
find_package(CUDA QUIET)
if (CUDA_FOUND)
    message(STATUS "CUDA found")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS_RELEASE -O3; -DNDEBUG)
    add_definitions(-DHH_USE_CUDA)
endif (CUDA_FOUND)

# NVTX
find_package(NVTX QUIET)

# Targets
if (LOG)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
    find_package(GLOG REQUIRED)
    include_directories(${GLOG_INCLUDE_DIRS})
    link_libraries(${GLOG_LIBRARIES})
    message(STATUS "GLOG imported")
    add_definitions(-DHLOG_ENABLED)
    add_definitions(-DHLOG_LEVEL=1)
endif (LOG)

if (RUN_GTEST)
    ExternalProject_Add(googletest
            GIT_REPOSITORY "https://github.com/google/googletest.git"
            CMAKE_ARGS
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/deps
                -DINSTALL_GTEST:BOOL=OFF
                -DBUILD_GMOCK:BOOL=OFF
            UPDATE_COMMAND ""
            INSTALL_COMMAND ""
            )

    ExternalProject_Get_Property(googletest source_dir)
    ExternalProject_Get_Property(googletest binary_dir)

    set(GTEST_INCLUDE_DIRS ${source_dir}/googletest/include)
    set(GTEST_LIBRARY_PATH ${binary_dir}/lib/)
    set(GTEST_LIBRARY gtest)

    message(STATUS "Binrary dir for gtest = ${binary_dir}")

    # Prevent GoogleTest from overriding our compiler/linker options
    # when building with Visual Studio
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    add_subdirectory(tests)
endif (RUN_GTEST)


if (BUILD_DOXYGEN)
    find_package(Doxygen REQUIRED)
    message(STATUS "Doxygen found")
    if (DOXYGEN_FOUND)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen/hedgehog.cfg ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc ALL
                ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API documentation with Doxygen" VERBATIM
                )

        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc/Hedgehog)
    endif (DOXYGEN_FOUND)
endif (BUILD_DOXYGEN)


if (BUILD_MAIN)
    message(STATUS "Build main")

    add_definitions(-D_GLIBCXX_ASSERTIONS)

    # Add Compile flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

    if (NOT MSVC)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Werror -pedantic -pthread")
    endif (NOT MSVC)

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        set(CMAKE_COMPILER_IS_CLANGXX 1)
        set(use-libclang ON)
    endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")

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

        if (NOT APPLE)
            link_libraries(stdc++fs)
        endif (NOT APPLE)
    elseif (NOT MSVC)
        link_libraries(stdc++fs)
    endif (CMAKE_COMPILER_IS_CLANGXX)

    set(HEDGEHOG_SRC
            hedgehog/hedgehog.h

            hedgehog/api/graph.h
            hedgehog/api/abstract_task.h
            hedgehog/api/state_manager/abstract_state.h
            hedgehog/api/abstract_execution_pipeline.h
            hedgehog/api/abstract_cuda_task.h

            hedgehog/behavior/execute.h
            hedgehog/behavior/node.h
            hedgehog/behavior/io/sender.h
            hedgehog/behavior/io/multi_receivers.h

            hedgehog/core/node/core_node.h
            hedgehog/core/node/core_task.h
            hedgehog/core/node/core_graph.h
            hedgehog/core/io/base/sender/core_sender.h
            hedgehog/core/io/base/sender/core_notifier.h
            hedgehog/core/io/base/receiver/core_receiver.h
            hedgehog/core/io/base/receiver/core_multi_receivers.h
            hedgehog/core/io/base/receiver/core_slot.h
            hedgehog/core/io/queue/sender/core_queue_sender.h
            hedgehog/core/io/queue/sender/core_queue_notifier.h
            hedgehog/core/io/queue/receiver/core_queue_receiver.h
            hedgehog/core/io/queue/receiver/core_queue_multi_receivers.h
            hedgehog/core/io/queue/receiver/core_queue_slot.h
            hedgehog/core/io/graph/receiver/core_graph_multi_receivers.h
            hedgehog/core/io/graph/receiver/core_graph_receiver.h
            hedgehog/core/io/graph/receiver/core_graph_sink.h
            hedgehog/core/io/graph/sender/core_graph_source.h
            hedgehog/api/scheduler/abstract_scheduler.h

            hedgehog/tools/helper.h
            hedgehog/tools/traits.h
            hedgehog/api/tools/graph_signal_handler.h
            hedgehog/api/printer/abstract_printer.h
            hedgehog/api/printer/dot_printer.h
            hedgehog/api/scheduler/default_scheduler.h
            hedgehog/tools/nvtx_profiler.h
            hedgehog/behavior/switch_rule.h
            hedgehog/core/node/execution_pipeline/core_switch.h

            hedgehog/core/behavior/core_execute.h
            hedgehog/core/defaults/core_default_task.h
            hedgehog/api/state_manager/state_manager.h
            hedgehog/core/node/execution_pipeline/core_execution_pipeline.h
            hedgehog/core/defaults/core_default_execution_pipeline.h

            hedgehog/api/memory_manager/abstract_memory_manager.h
            hedgehog/api/memory_manager/memory_data.h
            hedgehog/behavior/memory_manager/pool.h
            hedgehog/api/memory_manager/static_memory_manager.h)

    # Set main executable to be moved to a specific target
    if (CUDA_FOUND)
        message(STATUS "CUDA enabled")
        cuda_add_executable(main_cuda main.cc ${HEDGEHOG_SRC})
        target_compile_definitions(main_cuda PUBLIC -DHH_USE_CUDA)
    endif ()

    add_executable(main main.cc ${HEDGEHOG_SRC})
    if (NVTX_FOUND)
        message(STATUS "NVTX Enabled")
        target_compile_definitions(main PUBLIC -DHH_USE_NVTX)
        target_include_directories(main PUBLIC ${NVTX_INCLUDE_DIR})
        target_link_libraries(main ${NVTX_LIBRARIES})
    endif()
endif (BUILD_MAIN)