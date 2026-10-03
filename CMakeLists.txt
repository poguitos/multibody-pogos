cmake_minimum_required(VERSION 3.25)

project(MultibodyCore
    VERSION 0.1.0
    LANGUAGES CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Build concurrency -------------------------------------------------------
#
# Compiling this project is memory-hungry: one compiler process needs well over
# a gigabyte (see "Build limits" in README.md). The cap on simultaneous
# compiler and linker processes therefore lives here, in the build system, so
# that a plain `cmake --build` cannot overload the machine whatever -j value
# Ninja picks by itself. Raise it with -DMBD_COMPILE_JOBS=<n>.
#
# Job pools are a Ninja feature; other generators ignore these settings.
# This block must stay above every target, including the fetched dependencies.
set(MBD_COMPILE_JOBS 1 CACHE STRING
    "Maximum number of compiler processes running at once (Ninja only)")
set(MBD_LINK_JOBS 1 CACHE STRING
    "Maximum number of linker processes running at once (Ninja only)")
set_property(GLOBAL PROPERTY JOB_POOLS
    mbd_compile=${MBD_COMPILE_JOBS}
    mbd_link=${MBD_LINK_JOBS})
set(CMAKE_JOB_POOL_COMPILE mbd_compile)
set(CMAKE_JOB_POOL_LINK mbd_link)

# --- Options ---
option(MBD_BUILD_PYTHON "Build the Python bindings" ON)
option(MBD_WARNINGS "Compile the project's own code at a high warning level" ON)

include(FetchContent)

# Every dependency is pinned to an exact archive hash or commit, so that a
# fresh checkout builds the same code as this one.

#  Eigen: header-only, no tests, no subproject CMake
# We *only* want the headers, not Eigen's own CMake/CTest setup.
# So we download the sources and create our own INTERFACE target.

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    # Point at a directory without a CMakeLists.txt: the archive is downloaded
    # and unpacked, but Eigen's own build is never entered.
    SOURCE_SUBDIR not-a-cmake-project
)
FetchContent_MakeAvailable(eigen)

# Create a header-only imported target for Eigen
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE
    "${eigen_SOURCE_DIR}"
)

#  pybind11 (v2.12.0)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG 3e9dfa2866941655c56877882565e7577de6fc7b
    SYSTEM
)
FetchContent_MakeAvailable(pybind11)

#  Catch2 (v3.6.0, our test framework)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG 4e8d92bf02f7d1c8006a0e7a5ecabd8e62d98502
    SYSTEM
)
FetchContent_MakeAvailable(Catch2)

#  spdlog (v1.13.0, logging)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG 7c02e204c92545f869e2f04edaab1f19fe8b19fd
    SYSTEM
)
FetchContent_MakeAvailable(spdlog)

# --- Warning level for our own targets (never for the dependencies) ---
add_library(mbd_warnings INTERFACE)
if (MBD_WARNINGS)
    if (MSVC)
        target_compile_options(mbd_warnings INTERFACE /W4)
    else()
        target_compile_options(mbd_warnings INTERFACE -Wall -Wextra -Wpedantic)
    endif()
endif()

#  Library + Python bindings
add_subdirectory(src)
if (MBD_BUILD_PYTHON)
    add_subdirectory(python)
endif()

#  Testing setup (only *our* tests)
include(CTest)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
