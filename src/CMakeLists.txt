# 

add_library(multibody_core
    core.cpp
)

# Public headers live in the top-level include directory
target_include_directories(multibody_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Link to Eigen and spdlog (header-only is enough for spdlog)
target_link_libraries(multibody_core
    PUBLIC
        Eigen3::Eigen
        spdlog::spdlog_header_only
)
