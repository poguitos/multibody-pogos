cmake_minimum_required(VERSION 3.25)

project(MultibodyCore
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# --- Eigen: treat as pure header-only, don't run its CMake, no tests ----
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)

    # Define the standard Eigen target name used in examples/docs:
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
endif()

# --- pybind11 ---
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# --- Catch2 ---
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(Catch2)

# --- spdlog ---
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)

# ---- CTest / testing ----
include(CTest)

# Optionally wrap tests behind BUILD_TESTING (CTest sets this option)
if (BUILD_TESTING)
    add_subdirectory(tests)
endif()

add_subdirectory(src)
add_subdirectory(python)
