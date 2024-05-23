# Hedgehog (hh) : A library to generate heterogeneous dataflow graphs

Hedgehog is a header-only API that is designed to aid in creating dataflow graphs for algorithms to obtain performance across CPUs and multiple co-processors. This library represents the successor of the Hybrid Task Graph Scheduler ([HTGS](https://github.com/usnistgov/HTGS)).

## Dependencies

1) C++20 compiler (tested with gcc 11.1+, clang 10, and MSVC 14.33)

2) pthread

3) CUDA (https://developer.nvidia.com/cuda-zone) [optional]

4) GTEST (https://github.com/google/googletest) [optional / test]

5) For the static analysis (hh_cx) a compiler with the constexpr std::vector (P1004R2) and constexpr std::string (P0980R1) is needed, tested with gcc 12.1.0 +

## Building Hedgehog
**CMake Options:**

CMAKE_INSTALL_PREFIX - Where to install Hedgehog (and documentation)

TEST_HEDGEHOG - Compiles and runs google unit tests for Hedgehog ('make run-test' to re-run)

ENABLE_CHECK_CUDA - Enable extra checks for CUDA library if found

ENABLE_NVTX - Enable NVTX if CUDA is found

BUILD_MAIN - Build main file

```
 :$ cd <Hedgehog_Directory>
 :<Hedgehog_Directory>$ mkdir build && cd build
 :<Hedgehog_Directory>/build$ ccmake ../ (or cmake-gui)

 'Configure' and setup cmake parameters
 'Configure' and 'Build'

 :<Hedgehog_Directory>/build$ make
 :<Hedgehog_Directory>/build$ [sudo] make install
```

## Tutorials

[Hedgehog Tutorials](https://github.com/usnistgov/hedgehog-Tutorials)

# Credits

Alexandre Bardakoff

Timothy Blattner

Walid Keyrouz

Bruno Bachelet

Loïc Yon

Mary Brady

# Special thanks

We would like to thank Prof. Joel Falcou (https://www.lri.fr/~falcou/ / https://github.com/jfalcou) and Jules Penuchot (https://github.com/JPenuchot)
for their advice!

# Contact us

<a target="_blank" href="mailto:timothy.blattner@nist.gov">Timothy Blattner (timothy.blattner ( at ) nist.gov</a>
