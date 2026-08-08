# SPDX-License-Identifier: Apache-2.0
#
# The C++ core. Header-only except for the C ABI shim.
#
# The test suite does not need this file: tests/test_cpp_parity.py invokes the
# compiler directly so that `python -m pytest` works on a clone with nothing but a
# compiler installed. CMake is here for consuming the library from a real build and
# for the cross-compilation the target floor implies.
#
# Cross-compiling for the declared floor (Cortex-M4F, see DESIGN.md), which is how the
# flash and RAM figures should be obtained rather than by interpolating from the host:
#
#   cmake -B build-m4 -G Ninja \
#     -DCMAKE_SYSTEM_NAME=Generic \
#     -DCMAKE_C_COMPILER=arm-none-eabi-gcc \
#     -DCMAKE_CXX_COMPILER=arm-none-eabi-g++ \
#     -DARGMIND_EMBEDDED_FLAGS=ON
#   cmake --build build-m4
#
# That produces the static library only; the host tools and tests need a hosted
# runtime and are skipped automatically when cross-compiling.
#
# This recipe was written before any cross toolchain existed on the machine and was
# therefore unrun. It has now been run. `tests/test_target_build.py` drives the same flags
# directly and links four firmware-shaped configurations against `target/cortex-m4.ld`, so
# that flash and RAM come from a link map rather than from a sum of object sizes.
#
# What that found, and what an integrator therefore has to supply: the core calls `memcpy`
# and `memset`. GCC is permitted to emit calls to those even in a freestanding build, and
# it does. Beyond them it needs only libgcc's soft-float double helpers, which are the cost
# `align.hpp` already documents for a single-precision FPU.

cmake_minimum_required(VERSION 3.20)

# Must come BEFORE project(), which is what runs the compiler check. On a bare-metal
# target that check fails: it tries to link a full executable and newlib's stdio wants
# _read, _write, _sbrk and friends that no firmware has yet supplied. Compiling to a
# static library instead is the standard answer and checks the thing that matters.
#
# This is here because the cross recipe above was written before any cross toolchain
# existed on this machine, and a recipe nobody has run is a recipe that does not work:
# it did not, and this is what it was missing.
if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
endif()

project(argmind_core LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ARGMIND_EMBEDDED_FLAGS
       "Use the reference Cortex-M4F flags from DESIGN.md" OFF)
option(ARGMIND_VECTOR_REPORT
       "Emit -fopt-info-vec so auto-vectorisation is verified rather than assumed"
       OFF)

# -fno-exceptions -fno-rtti are not tuning. The core has to be usable in a
# freestanding environment, and DESIGN.md pays those costs deliberately.
add_library(argmind STATIC src/abi.cpp)
target_include_directories(argmind PUBLIC include)
target_compile_options(argmind PRIVATE
    -fno-exceptions -fno-rtti -Wall -Wextra -Wpedantic)

if(ARGMIND_EMBEDDED_FLAGS)
    # The declared floor: ARMv7E-M with a single-precision FPU. See "Target range"
    # in DESIGN.md for why the FPU is the boundary rather than memory or speed.
    #
    # -fno-math-errno is not a fast-math option and changes no result. Without it GCC
    # will not use VSQRT.F32, because it must keep the promise that sqrtf sets errno on
    # a negative argument -- so every norm() becomes a call into libm, three times per
    # RLS update. Measured with a real cross build on 2026-08-09; the values norm() takes
    # a root of are sums of squares, so the promise was never meaningful here.
    target_compile_options(argmind PRIVATE
        -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
        -ffreestanding -fno-math-errno -ffunction-sections -fdata-sections -Os)
endif()

if(ARGMIND_VECTOR_REPORT)
    # DESIGN.md: auto-vectorisation is not guaranteed, so it must be verified rather
    # than assumed. This is how, not a promise that it happened.
    target_compile_options(argmind PRIVATE -fopt-info-vec-optimized)
endif()

if(NOT CMAKE_CROSSCOMPILING)
    add_executable(argmind_test_core tests/test_core.cpp src/abi.cpp)
    target_include_directories(argmind_test_core PRIVATE include)
    target_compile_options(argmind_test_core PRIVATE
        -fno-exceptions -fno-rtti -Wall -Wextra -Wpedantic)

    add_executable(argmind_xcheck tools/xcheck.cpp)
    target_include_directories(argmind_xcheck PRIVATE include)
    target_compile_options(argmind_xcheck PRIVATE -fno-exceptions -fno-rtti)

    # The record-format parity tool. Every byte-identity claim rests on it, and it was
    # missing here while the pytest path built it directly with g++ — so the checked-in
    # build description had stopped covering the tree.
    add_executable(argmind_pcheck tools/pcheck.cpp)
    target_include_directories(argmind_pcheck PRIVATE include)
    target_compile_options(argmind_pcheck PRIVATE -fno-exceptions -fno-rtti)

    add_executable(argmind_scheck tools/scheck.cpp)
    target_include_directories(argmind_scheck PRIVATE include)
    target_compile_options(argmind_scheck PRIVATE -fno-exceptions -fno-rtti)

    add_executable(argmind_pipecheck tools/pipecheck.cpp)
    target_include_directories(argmind_pipecheck PRIVATE include)
    target_compile_options(argmind_pipecheck PRIVATE -fno-exceptions -fno-rtti)

    enable_testing()
    add_test(NAME core COMMAND argmind_test_core)
endif()
