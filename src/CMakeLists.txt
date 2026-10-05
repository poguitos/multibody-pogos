# The engine, as one compiled library. Headers under include/mbd/ hold
# declarations, small inline functions and templates; everything else is
# compiled here once (plan task 2.8).

add_library(multibody_core
    core.cpp
    kernel/joint_models.cpp
    kernel/model.cpp
    kernel/algorithms.cpp
    kernel/constraints.cpp
    kernel/constrained_dynamics.cpp
    kernel/forces.cpp
    kernel/simulator.cpp
    kernel/checks.cpp
    kernel/validate.cpp
    kernel/joint_forces.cpp
    kernel/outputs.cpp
    forces/force_element.cpp
    forces/curve.cpp
    forces/spring_damper.cpp
    forces/rotational_spring_damper.cpp
    forces/bushing.cpp
    forces/user_force.cpp
    forces/pacejka.cpp
    forces/tire.cpp
    forces/aerodynamics.cpp
    forces/anti_roll_bar.cpp
    vehicle/suspension/double_wishbone.cpp
    vehicle/suspension/mcpherson.cpp
    vehicle/suspension/multilink.cpp
    vehicle/vehicle.cpp
    vehicle/vehicle_template.cpp
    vehicle/drivetrain.cpp
    analysis/position_kinematics.cpp
    analysis/bicycle_model.cpp
    analysis/track.cpp
    analysis/lap_vehicle.cpp
    analysis/lap_speed_profile.cpp
    analysis/optimization.cpp
    analysis/recorder.cpp
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
    PRIVATE
        mbd_warnings
)

# Each source file would otherwise parse Eigen and the kernel headers again,
# about ten seconds per file with MSVC.
if (MBD_PCH)
    target_precompile_headers(multibody_core PRIVATE pch.hpp)
endif()
