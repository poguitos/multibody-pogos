cmake_minimum_required(VERSION 3.25)

project(MultibodyCore
    VERSION 0.1.0
    LANGUAGES CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

#  Eigen: header-only, no tests, no subproject CMake
# We *only* want the headers, not Eigen's own CMake/CTest setup.
# So we download the sources and create our own INTERFACE target.

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    # You can add URL_HASH here later if you want reproducible builds
)

FetchContent_GetProperties(eigen)
if (NOT eigen_POPULATED)
    FetchContent_Populate(eigen)

    # Create a header-only imported target for Eigen
    add_library(Eigen3::Eigen INTERFACE IMPORTED)

    target_include_directories(Eigen3::Eigen INTERFACE
        "${eigen_SOURCE_DIR}"
    )
endif()

#  pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

#  Catch2 (our test framework)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(Catch2)

#  spdlog (logging)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)

#  Testing setup (only *our* tests)
include(CTest)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()

#  Library + Python bindings
add_subdirectory(src)
add_subdirectory(python)
