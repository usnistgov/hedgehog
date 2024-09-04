cmake_minimum_required(VERSION 3.16)

project(hedgehog VERSION 3.3.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(FetchContent) # Used to download GTEST

option(TEST_HEDGEHOG "Downloads google unit test API and runs google test scripts to test Hedgehog core and api" ON)
option(HH_CX "Enable compile-time library for Hedgehog" OFF)
option(ENABLE_CHECK_CUDA "Enable extra checks for CUDA library if found" ON)
option(ENABLE_NVTX "Enable NVTX if CUDA is found" OFF)
option(BUILD_MAIN "Compiles main function for testing changes to API" ON)

set(CACHE_LINE_SIZE 64)

find_package(TBB QUIET COMPONENTS tbb)
find_package(Threads REQUIRED)
find_package(CUDAToolkit QUIET)

link_libraries(Threads::Threads)

if (NOT CACHE_LINE_SIZE)
    set(CACHE_LINE_SIZE 64)
endif (NOT CACHE_LINE_SIZE)

# Create version.h file in api folder
configure_file(inputs_cmake/version.h.in "${PROJECT_SOURCE_DIR}/hedgehog/version.h")
configure_file(inputs_cmake/constants.h.in "${PROJECT_SOURCE_DIR}/hedgehog/constants.h")

if (HH_CX)
    message("Hedgehog CX enabled")
    add_definitions(-DHH_ENABLE_HH_CX)
endif (HH_CX)

if (CUDAToolkit_FOUND)
    message("CUDA found: ${CUDAToolkit_INCLUDE_DIRS}")
    if (ENABLE_CHECK_CUDA) 
        message("CUDA check enabled")
        add_definitions(-DHH_ENABLE_CHECK_CUDA)
    else (ENABLE_CHECK_CUDA)
        message("CUDA check disabled")
    endif (ENABLE_CHECK_CUDA)
    include_directories(CUDAToolkit_INCLUDE_DIRS)
    link_libraries(CUDA::cudart CUDA::cuda_driver CUDA::cupti)
    add_definitions(-DHH_USE_CUDA)
    if (ENABLE_NVTX)
        link_libraries(CUDA::nvToolsExt)
        link_libraries(dl)
        add_definitions(-DHH_USE_NVTX)
    endif (ENABLE_NVTX)
else (CUDAToolkit_FOUND)
    message("CUDA not found, all features will not be available.")
endif (CUDAToolkit_FOUND)

# Gtest
if (TEST_HEDGEHOG)
    message("Fetch Gtest")
    if (POLICY CMP0135)
        FetchContent_Declare(
                googletest
                URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
                DOWNLOAD_EXTRACT_TIMESTAMP true
        )
    else ()
        FetchContent_Declare(
                googletest
                URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
        )
    endif ()

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    file(GLOB_RECURSE test_hedgehog_header tests/*.h)
    add_executable(gtest_hedgehog ${test_hedgehog_header} tests/test_hedgehog.cc)

    target_link_libraries(gtest_hedgehog gtest_main)

    if(TBB_FOUND)
        target_link_libraries(gtest_hedgehog tbb)
    endif (TBB_FOUND)

    include(GoogleTest)
    gtest_discover_tests(gtest_hedgehog)
endif (TEST_HEDGEHOG)


if (BUILD_MAIN)
    message(STATUS "Build main")

    if (MSVC)
        set(CMAKE_CXX_FLAGS "-Wall")
    else (MSVC)
        set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -Werror -pedantic -pipe -v")
    endif (MSVC)

    file(GLOB_RECURSE hedgehog_header hedgehog/*.h)

    add_executable(hedgehog main.cc ${hedgehog_header})

    if(TBB_FOUND)
        target_link_libraries(hedgehog tbb)
    endif (TBB_FOUND)

endif (BUILD_MAIN)

