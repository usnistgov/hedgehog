# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


# - Find Hedgehog includes and required compiler flags and library dependencies
# Dependencies: C++20 support and threading library
#
# The Hedgehog_CXX_FLAGS should be added to the CMAKE_CXX_FLAGS
#
# This module defines
#  Hedgehog_INCLUDE_DIRS
#  Hedgehog_FOUND
#  CACHE_LINE_SIZE
#  HH_ENABLE_HH_CX (OPTIONAL)
#  CMAKE_CXX_FLAGS (OPTIONAL)
#

# Ensure C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TEST_HEDGEHOG "Downloads google unit test API and runs google test scripts to test Hedgehog core and api" ON)
option(HH_CX "Enable compile-time library for Hedgehog" OFF)
option(ENABLE_CHECK_CUDA "Enable extra checks for CUDA library if found" ON)
option(ENABLE_NVTX "Enable NVTX if CUDA is found" OFF)

if (NOT CACHE_LINE_SIZE)
    set(CACHE_LINE_SIZE 64)
endif (NOT CACHE_LINE_SIZE)

if (HH_CX)
    message("Hedgehog CX enabled")
    add_definitions(-DHH_ENABLE_HH_CX)
endif (HH_CX)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20")
endif (MSVC)

# Try to found Hedgehog
SET(Hedgehog_FOUND ON)

FIND_PATH(Hedgehog_INCLUDE_DIR hedgehog.h
        /usr/include/hedgehog
        /usr/local/include/hedgehog
        )

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

find_package(CUDAToolkit QUIET)

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

find_package(TBB QUIET COMPONENTS tbb)
if(TBB_FOUND)
    link_libraries(tbb)
endif (TBB_FOUND)

IF (NOT Hedgehog_INCLUDE_DIR)
    SET(Hedgehog_FOUND OFF)
    MESSAGE(STATUS "Could not find Hedgehog includes. Hedgehog_FOUND now off")
ELSE(NOT Hedgehog_INCLUDE_DIR)
    list(APPEND Hedgehog_INCLUDE_DIRS ${Hedgehog_INCLUDE_DIR})
ENDIF ()

IF (Hedgehog_FOUND)
    IF (NOT Hedgehog_FIND_QUIETLY)
        MESSAGE(STATUS "Found Hedgehog include: ${Hedgehog_INCLUDE_DIR}")
    ENDIF (NOT Hedgehog_FIND_QUIETLY)
ELSE (Hedgehog_FOUND)
    IF (Hedgehog_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find Hedgehog header files, please set the cmake variable Hedgehog_INCLUDE_DIR")
    ENDIF (Hedgehog_FIND_REQUIRED)
ENDIF (Hedgehog_FOUND)


include_directories(${Hedgehog_INCLUDE_DIRS})

MARK_AS_ADVANCED(Hedgehog_INCLUDE_DIR)
