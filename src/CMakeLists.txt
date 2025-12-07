# CMakeLists.txt in src/

add_library(multibody_core
    core.cpp
)

target_include_directories(multibody_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        # Eigen is header-only, target is "Eigen3::Eigen"
)

target_link_libraries(multibody_core
    PUBLIC
        Eigen3::Eigen
        spdlog::spdlog
)
